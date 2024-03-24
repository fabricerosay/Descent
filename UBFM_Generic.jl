using CUDA
using Flux
using Base.Iterators: partition
using Statistics
using Distributions
using DataStructures
using Random
using StatsBase
using Base.Threads: @threads, @spawn
using LinearAlgebra
using ArgParse
using BSON
using JLD2
using ProgressMeter
using MLUtils
using StaticArrays
#using SimpleChains

module UBFM

using ..CUDA
using ..Flux
using ..Base.Iterators: partition
using ..Statistics
using ..Distributions
using ..DataStructures
using ..Random
using ..ProgressMeter
using ..StatsBase
using ..JLD2
using ..BSON: @save, @load
using ..LinearAlgebra
using ..ArgParse
using ..MLUtils
#using SimpleChains
using OMEinsum

include("/home/fabrice/Julia_Files/UBFM/UBFM-clean/UTTT/UTTT.jl")
include("net.jl")
include("train.jl")
#include("train_simplechains.jl")
using .Game
const NAME = Game.Game_Name

const StateShape = Game.StateShape
const NN = Game.BOARDSIZE_square
const maxmove = 81

struct Sample
    state::GameState
    value::Float32
end

function ordinal_selection(pos, T, ϵ)
    n = length(T[pos].actions)
    if pos.player == 1
        rev = true
    else
        rev = false
    end
    I = sortperm(T[pos].actions; rev=rev)#by=x ->(x.solved x.value
    for j in 0:(n-1)
        if rand() <= (ϵ * (n - j - 1) + 1) / (n - j)
            return T[pos].moves[I[j+1]]
        end
    end
end

function OneGame(actor, T, S, τ, batch, lmp)
    f = false
    position = GameState()
    len = 0
    p = Progress(round(Int, lmp))
    while !f
        action = ubfm(position, T, S, actor, batch, τ)
        #if T[position].value.resolved == 0
        action = ordinal_selection(position, T, rand())
        #end
        position = play(position, action)
        len += 1
        f, r = iswon(position)
        next!(p)
    end
    return len
end

function duel(actor1, actor2, entries)
    testmode!(actor1, true)
    testmode!(actor2, true)
    n = length(entries)
    position = Game.Game[]
    for entrie in entries
        pos = Game.Game()
        for m in entrie
            Game.play(pos, m)
        end
        push!(position, pos)
    end
    sposition = [pos.player for pos in position]
    batch = [zeros(Float32, (StateShape..., maxmove)) for k in 1:n]
    p = 0
    rem = n
    v1 = 0
    v2 = 0
    round = 0
    player = 1

    while true
        answer = [Channel(1) for k in 1:rem]
        T = [Dict() for k in 1:rem]
        @sync begin
            if player == 1
                req = server(rem, actor1)
            else
                req = server(rem, actor2)
            end
            player *= -1
            for k in 1:rem
                @async begin
                    for j in 1:1
                        UBFMComp(
                            position[k], T[k], Evaluator(req, answer[k]), batch[k], 1.1
                        )
                    end
                    put!(req, (0, answer[k]))
                end
            end
        end
        # @threads for k in 1:rem
        #     @views batch[:,:,:,k].=getboard(position[k])[:,:,:,1]
        # end
        # if position[1].player==1
        #     evals,pol=actor1(batch|>gpu,true)|>cpu
        # else
        #     evals,pol=actor2(batch|>gpu,true)|>cpu
        # end
        todelete = Int[]
        for k in 1:rem
            #push_buffer(pool,reshape(getboard(position[k]),84),evals[1,k],0)
            # actions=gen_moves(position[k])
            # if round<7
            #     p=softmax([0.5*ac.value for ac in  T[k][Game.gethash(position[k])].actions])
            #     action=sample(actions,weights(p))
            #
            #     #action=rand(actions)
            # else
            action = bestaction(position[k], T[k])
            action = T[k][Game.gethash(position[k])].moves[action]

            #end
            Game.play(position[k], action)
            # println("position: ",position[k].round)
            # println("evaluation: ",T[k][Game.gethash(position[k])].value)
            # tsamples+=1
            # if tsamples>=0.1*nsamples
            #     p+=10
            #     println("$p% accomplis")
            #     tsamples=0
            #
            # end
            f, r = Game.iswon(position[k])
            if f
                push!(todelete, k)
                #println("victoire: $f")
                rem -= 1
                if r == sposition[k]
                    v1 += 1
                elseif r == -sposition[k]
                    v2 += 1
                end
            end
        end
        round += 1
        position = [pos for (k, pos) in enumerate(position) if !(k in todelete)]
        sposition = [player for (k, player) in enumerate(sposition) if !(k in todelete)]
        if isempty(position)
            break
        end
    end
    return v1, v2
end

function gen_entries(n)
    entries = Vector{Vector{Game.Move}}()
    for k in 1:n
        pos = Game.Game()
        e = Game.Move[]
        for k in 1:4
            move = rand(Game.gen_moves(pos))
            push!(e, move)
            Game.play(pos, move)
        end
        push!(entries, e)
    end
    return entries
end

function gen_entries_lvl(lvl)
    if lvl == 1
        entries = Vector{Vector{Game.Move}}()
        for k in 1:81
            e = Game.Move[]
            push!(e, k)
            push!(entries, e)
        end
    else
        entries = Vector{Vector{Int}}()
        for k in 1:81
            pos = Game.Game()

            Game.play(pos, k)
            for move in Game.gen_moves(pos)
                e = [k]
                push!(e, move)
                push!(entries, e)
            end
        end
    end
    return entries
end

function fair_duel(actor1, actor2, n)
    entries = gen_entries(n)#sample(gen_entries_lvl(2),100,replace=false)
    n = length(entries)
    v = 0
    d = 0
    for k in 1:10:n
        v1, v2 = duel(actor1, actor2, entries[k:min(n, k + 9)])
        v3, v4 = duel(actor2, actor1, entries[k:min(n, k + 9)])
        v += v1 + v4
        d += v2 + v3
    end
    return v / n * 50, d / n * 50
end

const R = [reshape(rotr90(reshape(collect(1:9), (3, 3)), r), 9) for r in 1:4]
const RH = [7, 8, 9, 4, 5, 6, 1, 2, 3]

function rotstateh(state)
    NCol = size(state)[2]
    NRow = size(state)[1]
    answer = similar(state)
    for i in 1:NCol
        for j in 1:NRow
            answer[i, j, :, :] = state[i, NCol-j+1, :, :]
        end
    end
    return answer
end

function rotstate(state, i)
    NCol = size(state)[2]
    NRow = size(state)[1]
    answer = zeros(Int8, NRow, NCol, 3, 1)

    answer[:, :, 1, 1] .= rotr90(state[:, :, 1, 1], i)
    answer[:, :, 2, 1] .= rotr90(state[:, :, 2, 1], i)
    answer[:, :, 3, 1] .= state[:, :, 3, 1]

    return answer
end
## state_fusion()
function state_fusion(buffer)
    T = Dict{GameState,Tuple{Float32,Int}}()
    trainingbuffer = Sample[]
    for games in buffer
        for s in games
            if haskey(T, s.state)
                T[s.state] = (T[s.state][1] + s.value, T[s.state][2] + 1)
            else
                T[s.state] = (s.value, 1)
            end
        end
    end
    for g in keys(T)
        push!(trainingbuffer, Sample(g, T[g][1] / T[g][2]))
    end
    return trainingbuffer
end

function selfplay(n, actor, τ; start=0)
    buffer = CircularBuffer{Vector{Sample}}(100)
    if start != 0
        for k in 1:100
            push!(buffer, Sample[])
        end
        Threads.@threads for k in 0:99
            id = start - k
            buff = JLD2.load(
                "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/samples_uttt/data$id.jld2",
                "data",
            )
            append!(buffer[100-k], buff)
        end
    end
    nsamples = 0
    len_mean_partie = 0
    time0 = time()
    k = 1
    nsamplestot = 0
    batch = zeros(Float32, (StateShape..., 200))
    think = τ
    lr = 0.0001 #### changed lr from 0.001 to 0.0001
    S = Dict{GameState,Bool}()
    T = Dict{GameState,StateInfo}()
    len_partie = 0
    solved = 0
    len_mean_partie = 100
    while time() - time0 < n * 3600
        println("epoque: $k")
        t = time()

        len = OneGame(actor, T, S, think, batch, len_mean_partie)

        len_partie += len
        len_mean_partie = len_partie / k
        this_game = Vector{Sample}()
        to_save = Vector()
        for key in keys(S)
            erase = T[key].value.resolved == 0
            value = T[key].value.value
            push!(this_game, Sample(key, value))
            if !erase
                solved += 1
            else
                delete!(T, key)
            end
        end
        nsamplestot += length(this_game)
        push!(buffer, this_game)
        samples = div(nsamplestot, k)
        println("total de samples acquis: $nsamplestot")
        println("moyenne de samples acquis: $samples")
        println("total de samples terminaux: $solved")
        println("samples acquis ce tour: ", length(this_game))
        println("longueur de la partie: $len")
        println("longueur moyenne des parties: $len_mean_partie")
        println("taille de l'arbre: ", length(T))

        if length(T) > 2_000_000
            empty!(T)
        end
        empty!(S)
        training_state = reduce(vcat, buffer)#state_fusion(buffer)
        buffer_size = length(training_state)
        ntsamples = round(Int, buffer_size / 20)

        traininPipe(1024, actor, training_state, ntsamples; lr=lr, epoch=1)

        sampatt = 3600 * (n) * (nsamplestot) / (time() - time0)
        println("samples attendus: ", sampatt)
        reseau = cpu(actor)
        saving_index = k + start
        jldsave(
            "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/samples_uttt/data$saving_index.jld2";
            data=this_game
        )
        @save "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/nets_uttt/reseau$k.json" reseau
        m = time() - t
        println(
            "temps de l'époque: ",
            floor(m / 60),
            "minutes ",
            m - 60 * floor(m / 60),
            "secondes",
        )
        ts = n * 3600 - time() + time0

        h = floor(ts / 3600)
        ts = ts - 3600 * h
        m = floor(ts / 60)
        ts = round(ts - 60 * m)

        println("temps restant: $h heures, $m minutes, $ts secondes")
        k += 1
    end
end

function retrain(n, actor, τ; start=0)
    buffer = CircularBuffer{Vector{Sample}}(100)
    if start != 0
        for k in 1:100
            push!(buffer, Sample[])
        end
        Threads.@threads for k in 0:99
            id = start - k
            buff = JLD2.load(
                "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/samples/data$id.jld2",
                "data",
            )
            append!(buffer[100-k], buff)
        end
    end

    nsamplestot = 0

    lr = 0.001

    for k in 1:n
        println("epoque: $k")
        t = time()
        id = start + k
        this_game = JLD2.load(
            "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/samples/data$id.jld2", "data"
        )
        nsamplestot += length(this_game)
        push!(buffer, this_game)
        training_state = reduce(vcat, buffer)#state_fusion(buffer)
        buffer_size = length(training_state)
        ntsamples = round(Int, buffer_size / 20)

        traininPipe(1024, actor, training_state, ntsamples; lr=lr, epoch=1)

        reseau = cpu(actor)

        @save "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/nets/reseau128x10id$k.json" reseau
        m = time() - t
        println(
            "temps de l'époque: ",
            floor(m / 60),
            "minutes ",
            m - 60 * floor(m / 60),
            "secondes",
        )
    end
end

mutable struct SValues
    resolved::Int8
    solved::Int8
    value::Float32
    n::Int32
end

import Base: isless

function isless(a::SValues, b::SValues)
    return isless(a.solved, b.solved) || ((a.solved == b.solved) & isless(a.value, b.value))
end

mutable struct StateInfo
    value::SValues
    actions::Vector{SValues}
    moves::Vector{Move}
end

function update(sv1::SValues, sv2::SValues)
    sv1.resolved = sv2.resolved
    sv1.solved = sv2.solved
    sv1.value = sv2.value
    return nothing
end

function bestaction(position, T)
    if position.player == 1
        k = argmax(T[position].actions)
        return k
    else
        k = argmin(T[position].actions)
        return k
    end
end

function bestaction_final(position, T)
    if position.player == 1
        k = argmax([
            (action.solved, action.n, action.value) for action in T[position].actions
        ])
        return k
    else
        k = argmin([
            (action.solved, -action.n, action.value) for action in T[position].actions
        ])
        return k
    end
end

function update(position, T::Dict{V}) where {V}
    k = bestaction(position, T)
    T[position].value.solved = T[position].actions[k].solved
    T[position].value.value = T[position].actions[k].value
    if T[position].value.solved != 0
        T[position].value.resolved = 1
    else
        T[position].value.resolved = minimum(
            action.resolved for action in T[position].actions
        )
    end
end

function UBFMDesc(position, T, S, evaluation, batch)
    f, r = Game.iswon(position)
    S[position] = true
    if f
        v = Game.score(position) * r
        T[position] = StateInfo(SValues(1, r, v, 0), SValues[], Game.Move[])
    else
        if !haskey(T, position)
            movestoeval = Vector{Tuple{Int,Int}}()
            moves = Move[]
            nmoves = genmoves(position, moves)
            actions = [SValues(0, 0, 0, 0) for k in 1:nmoves]
            cpt = 0
            for (ind, move) in enumerate(moves)
                newposition = play(position, move)
                if haskey(T, newposition)
                    update(actions[ind], T[newposition].value)
                else
                    f, r = Game.iswon(newposition)
                    if f
                        v = Game.score(newposition) * r
                        sv = SValues(1, r, v, 0)
                        T[newposition] = StateInfo(sv, SValues[], Game.Move[])
                        S[newposition] = true
                        update(actions[ind], sv)
                    else
                        cpt += 1
                        push!(movestoeval, (ind, cpt))
                        decode(newposition, batch, cpt)
                    end
                end
            end
            if cpt != 0
                eval = cpu(evaluation(gpu(batch[:, :, :, 1:cpt])))
                for (ind, cpt) in movestoeval
                    actions[ind].value = eval[1, cpt]
                end
            end
            T[position] = StateInfo(SValues(0, 0, 0.0, 0), actions, moves)
            update(position, T)
        end
        if T[position].value.resolved == 0
            candidats = [
                k for
                k in 1:length(T[position].moves) if T[position].actions[k].resolved == 0
            ]

            if position.player == 1
                M = maximum(T[position].actions[k].value for k in candidats)
            else
                M = minimum(T[position].actions[k].value for k in candidats)
            end
            filter!(x -> T[position].actions[x].value == M, candidats)
            k = rand(candidats)

            move = T[position].moves[k]
            T[position].actions[k].n += 1
            update(
                T[position].actions[k],
                UBFMDesc(play(position, move), T, S, evaluation, batch),
            )
            update(position, T)
        end
    end
    T[position].value.n += 1
    return T[position].value
end

function PV(pos, T)
    moves = Game.Move[]
    sizehint!(moves, 45)
    position = deepcopy(pos)
    cpt = 0

    while haskey(T, Game.gethash(position)) && T[Game.gethash(position)].value.resolved == 0
        k = bestaction(position, T)
        m = T[Game.gethash(position)].moves[k]
        Game.play(position, m)
        cpt += 1
        push!(moves, k)
    end
    if !haskey(T, Game.gethash(position))
        f = false
    else
        f = T[Game.gethash(position)].value.resolved == 1
    end
    return moves, f
end

function ubfm(position, T, S, evaluation, batch, τ)
    t = time()
    while true
        UBFMDesc(position, T, S, evaluation, batch)
        (T[position].value.resolved != 0 || time() - t >= τ) && break
    end
    k = bestaction(position, T)
    return T[position].moves[k]
end

resh(v, α) = abs(v) >= 0.5 + 0.5 * α ? Int8(sign(v)) : Int8(0)

const VV = Float32.(collect(-40:40))

function UBFMComp(position, T, evaluation, batch, timer, gpu_device)
    f, r = Game.iswon(position)
    if f
        v = Game.score(position) * r
        T[position] = StateInfo(SValues(1, r, v, 0), SValues[], Game.Move[])
    else
        if !haskey(T, position)
            evaluate = true
            movestoeval = Vector{Tuple{Int,Int}}()
            moves = Move[]
            nmoves = genmoves(position, moves)
            actions = [SValues(0, 0, 0, 0) for k in 1:nmoves]
            cpt = 0
            for (ind, move) in enumerate(moves)
                newposition = play(position, move)
                if haskey(T, newposition)
                    update(actions[ind], T[newposition].value)
                else
                    f, r = Game.iswon(newposition)
                    if f
                        v = Game.score(newposition) * r
                        sv = SValues(1, r, v, 0)
                        # if position.player * r == 1
                        #     evaluate = false
                        #     break
                        # end
                        T[newposition] = StateInfo(sv, SValues[], Game.Move[])
                        update(actions[ind], sv)
                    else
                        cpt += 1
                        push!(movestoeval, (ind, cpt))
                        decode(newposition, batch, cpt)
                    end
                end
            end
            if cpt != 0 && evaluate
                t = time()
                if gpu_device
                    eval = cpu(evaluation(gpu(batch[:, :, :, 1:cpt])))
                else
                    eval = evaluation(batch[:, :, :, 1:cpt])
                end
                timer[2] += length(eval)
                timer[3] += 1
                timer[1] += time() - t
                for (ind, cpt) in movestoeval
                    actions[ind].value = eval[1, cpt]
                end
            end
            T[position] = StateInfo(SValues(0, 0, 0.0, 0), actions, moves)
            update(position, T)
        else
            if T[position].value.resolved == 0
                candidats = [
                    k for
                    k in 1:length(T[position].moves) if T[position].actions[k].resolved == 0
                ]

                if position.player == 1
                    M = maximum(T[position].actions[k].value for k in candidats)
                else
                    M = minimum(T[position].actions[k].value for k in candidats)
                end
                filter!(x -> T[position].actions[x].value == M, candidats)
                k = rand(candidats)

                move = T[position].moves[k]
                T[position].actions[k].n += 1
                update(
                    T[position].actions[k],
                    UBFMComp(play(position, move), T, evaluation, batch, timer, gpu_device),
                )
                update(position, T)
            end
        end
    end
    T[position].value.n += 1
    return T[position].value
end

function UBFMVs(position, T, evaluation, batch, τ; gpu_device=true, verbose=false)
    t = time()
    timer = [0.0, 0.0, 0.0]
    while true
        UBFMComp(position, T, evaluation, batch, timer, gpu_device)
        (T[position].value.resolved != 0 || (time() - t >= τ)) && break
    end
    k = bestaction_final(position, T)
    a = T[position].moves[k]
    # println("ratio temps eval/temps total", timer[1] / (time() - t))
    if verbose
        println(
            "leaves evaluated: ", timer[2], " average number of leaves: ", timer[2] / timer[3]
        )
    end
    return a
end

function affiche_moves(moves)
    _, Dic_coups = generate_dict()
    for move in moves
        println(Dic_coups[move])
        readline()
    end
end

function test(actor)
    res = zeros(1000)
    for k in 0:999
        batch = rand(Float32, (9, 9, 3, k + 1500))
        t = time()
        r = cpu(actor(gpu(batch)))
        res[k+1] = (time() - t) / (k + 1500)
        GC.gc(true)
        CUDA.reclaim()
    end
    return argmin(res)
end

# generation = parsed_args["generation"]

function extract_dat_ubfm(buff)
    newbuff = Vector{Tuple{GameState,Float32}}()
    for s in buff
        push!(newbuff, (s.state, s.value))
    end
    return newbuff
end

function add_data_ubfm(buffer, a, b, ϕ=0)
    newbuff = [Vector{eltype(buffer)}() for k in 1:Threads.nthreads()]
    Threads.@threads for k in a:b
        buff = JLD2.load(
            "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/samples/data$k.jld2", "data"
        )
        append!(newbuff[Threads.threadid()], buff)
    end
    for buff in newbuff
        append!(buffer, buff)
    end
end

function trainnnue_ubfm(ev, n, ϕ, Nfile=10, maxF=198; start=0, lr=0.00001)
    buffer = Vector{Sample}()

    ndatas = 0
    for k in 1:n
        epochs = div(maxF - 1 - start, Nfile)
        for j in 0:epochs
            # if j == div(epochs, 2)
            #     lr = lr / 5
            # end
            println("Epoch $k/$n, dataset $j/$epochs phase $ϕ")

            fin = Nfile + Nfile * j + start
            if j == epochs
                fin = maxF - 2
            end
            add_data_ubfm(buffer, 1 + Nfile * j + start, fin, ϕ)
            ndatas += length(buffer)
            println("datas so far: $ndatas")
            traininPipe(1024, ev, buffer, length(buffer); lr=lr)
            #test=Main.sample(check,10000)
            empty!(buffer)
        end
    end
end

function trainnnue_ubfm_simple(ev, n, ϕ, Nfile=10, maxF=198; start=0, lr=0.00001)
    buffer = Vector{Sample}()

    ndatas = 0
    for k in 1:n
        epochs = div(maxF - 1 - start, Nfile)
        for j in 0:epochs
            # if j == div(epochs, 2)
            #     lr = lr / 5
            # end
            println("Epoch $k/$n, dataset $j/$epochs phase $ϕ")

            fin = Nfile + Nfile * j + start
            if j == div(maxF - 1, Nfile)
                fin = maxF - 2
            end
            add_data_ubfm(buffer, 1 + Nfile * j + start, fin, ϕ)
            ndatas += length(buffer)
            println("datas so far: $ndatas")
            train_simple(1024, ev, buffer, length(buffer); lr=lr)
            #test=Main.sample(check,10000)
            empty!(buffer)
        end
    end
end

####################alpha beta####################

function init_reduction_table()
    lmrtable = zeros(Int, (64, 200))
    for depth in 1:64
        for played in 1:200
            @inbounds lmrtable[depth, played] = floor(
                Int, 0.2 + 0.6 * log(depth) * log(played) * log(played)
            )
        end
    end
    return lmrtable
end

const Reduction = init_reduction_table()

function add_node(position, T, evaluation, batch)
    movestoeval = Vector{Tuple{Int,Int}}()
    moves = Move[]
    nmoves = genmoves(position, moves)
    actions = [SValues(0, 0, 0, 0) for k in 1:nmoves]
    cpt = 0
    for (ind, move) in enumerate(moves)
        newposition = play(position, move)
        if haskey(T, newposition)
            update(actions[ind], T[newposition].value)
        else
            f, r = Game.iswon(newposition)
            if f
                v = Game.score(newposition) * r
                sv = SValues(1, r, v, 0)
                T[newposition] = StateInfo(sv, SValues[], Game.Move[])
                S[newposition] = true
                update(actions[ind], sv)
            else
                cpt += 1
                push!(movestoeval, (ind, cpt))
                decode(newposition, batch, cpt)
            end
        end
    end
    if cpt != 0
        eval = cpu(evaluation(gpu(batch[:, :, :, 1:cpt])))
        for (ind, cpt) in movestoeval
            actions[ind].value = eval[1, cpt]
        end
    end
    T[position] = StateInfo(SValues(0, 0, 0.0, 0), actions, moves)
    return update(position, T)
end

# struct TC
#     finish::Float64
# end

# function TCstart(t)
#     st = time()
#     return TC(st + t)
# end

# function over(tc::TC)
#     return time() >= tc.finish
# end

# struct Entry
#     key::UInt
#     move::Move
#     depth::UInt8
#     value::Int16
#     flag::UInt8
# end

# struct HashTable
#     size::Int
#     data::Vector{Entry}
# end

# HashTable(n) = HashTable(n, fill(Entry(0, GAME.αNONEMOVE, 0, 0, 0), n))

# index(tt::HashTable, key) = (key - 1) % tt.size + 1
# HashTable() = HashTable(8388593)

# function put(ht, key, depth, move, value, flag)
#     return ht.data[index(ht, key)] = Entry(key, move, depth, value, flag)
# end

# function retrieve(ht, key)
#     entry = ht.data[index(ht, key)]
#     if entry.key == key
#         move = entry.move
#         value = entry.value
#         depth = entry.depth
#         flag = entry.flag
#     else
#         move = NONEMOVE
#         value = 0
#         depth = 0x0
#         flag = 0x0
#     end
#     return move, value, flag, depth
# end

# mutable struct Stack
#     nodes::Int
#     tt::HashTable
# end

# Stack(n) = Stack(0, HashTable())
# const CM = 10000
# const basemarge = round(Int, 200 * CM / 5000)
# const δ = round(Int, 100 * CM / 5000)
# const Δ = [0, 0, 500, 1000, 1500, 1500, 1500, 500, 100, 0, 0] ### tuned for weights 156000 et CM=10000
# const MARG = [0, basemarge, 2 * basemarge, 3 * basemarge, 4 * basemarge]
# @inbounds function abbase(game, eval, α, β, depth, ply, moves, stack, tc)
#     if over(tc)
#         return 0
#     end
#     f, r = GAME.isOver(game)
#     if f
#         return 10000 * r * game.player
#     end
#     if depth <= 1
#         return round(Int, clamp(eval(game) * CM, -9999, 9999))
#     end
#     ##### Sans transpo proche autaxx mais gros gain en profondeur######
#     move, value, flag, d = retrieve(stack.tt, game.hash)
#     #move=GAME.αNONEMOVE
#     if flag == 2 && d >= depth
#         α = max(α, value)
#         static_eval = value
#         if α >= β
#             return α
#         end
#     elseif flag == 1 && d >= depth
#         β = min(β, value)
#         static_eval = value
#         move = GAME.αNONEMOVE
#         if α >= β
#             return α
#         end
#     elseif flag == 3 && d >= depth
#         return value
#     end
#     nmoves = GAME.αgen_moves(game, moves[ply + 1], move)
#     bestmove = move
#     bestvalue = -Inf
#     prevα = α

#     for k in 1:nmoves
#         cmove = popfirst!(moves[ply + 1])

#         u = GAME.play(game, cmove.move)
#         v = -abbase(game, eval, -β, -α, depth - 1, ply + 1, moves, stack, tc, CM, T)

#         GAME.undo(game, cmove.move, u)

#         if v > bestvalue
#             bestvalue = v
#             bestmove = cmove
#             if v > α
#                 # stack.pv[1,ply+1]=bestmove
#                 # for k in 1:ply
#                 #     stack.pv[k+1,ply+1]=stack.pv[k,ply+2]
#                 # end
#                 α = v
#                 if α >= β
#                     empty!(moves[ply + 1])
#                     break
#                 end
#             end
#         end
#     end
#     if bestvalue <= prevα
#         flag = 1
#     elseif bestvalue >= β
#         flag = 2
#     else
#         flag = 3
#     end
#     if T != nothing
#         T[deepcopy(game)] = bestvalue
#     end
#     put(stack.tt, game.hash, depth, bestmove, bestvalue, flag)

#     return bestvalue
# end

# function abbase_iterative(
#     game, t, ev, stack=nothing, moves=nothing; maxdepth=255, CM=CM, T=nothing
# )
#     if moves == nothing
#         moves = [GAME.αMoves() for k in 1:maxdepth]
#     end
#     if stack == nothing
#         stack = Stack(maxdepth)
#     end
#     local bestmove = GAME.NONEMOVE
#     local bestvalue = 0
#     v = -32768
#     depth = 0
#     tc = TCstart(t)
#     tcs = TCstart(1)
#     v = abbase(game, ev, -30000, 30000, 1, 0, moves, stack, tcs, CM, T)
#     bestvalue = v
#     PV = stack.pv

#     bestmove, _, _, _ = retrieve(stack.tt, game.hash)#stack.pv[1,1].move
#     depth = 1
#     for d in 2:maxdepth
#         timeiter = time()
#         v = abbase(game, ev, -30000, 30000, d, 0, moves, stack, tc, CM, T)
#         timeiter = time() - timeiter
#         timeleft = tc.finish - time()
#         over(tc) && break
#         bestvalue = v
#         bestmove, _, _, _ = retrieve(stack.tt, game.hash)#stack.pv[1,1].move
#         depth = d
#         timeleft <= 1.5 * timeiter && break
#     end
#     return bestmove.move, bestvalue, depth
# end
function game_vs_net(actor, τ=1)
    game = GameState()
    T = Dict{GameState,StateInfo}()
    batch = rand(Float32, 3, 3, 12, 81)
    f, r = iswon(game)
    while !f
        if game.player == 1
            move = UBFMVs(game, T, actor, batch, τ)
            println("state value ", T[game].value)
            midx = move.to
            x = div(midx - 1, 9) + 1
            y = (midx - 1) % 9 + 1
            println("computer move $x $y")
        else
            println("enter you move (zone nextboard)")
            str = readline()
            x = parse(Int, str[1])
            y = parse(Int, str[3])
            move = Move(9 * (x - 1) + y)
        end
        game = play(game, move)
        f, r = iswon(game)
    end
end

function game_vs_rand(actor)
    game = GameState()
    T = Dict()
    batch = rand(Float32, 3, 3, 12, 81)
    f, r = iswon(game)
    while !f
        if game.player == 1
            move = UBFMVs(game, T, actor, batch, 0)
            #println("state value ", T[game].value)
            #midx=move.to
            #x=div(midx-1,9)+1
            #y=(midx-1)%9+1
            #println("computer move $x $y")
        else
            moves = Move[]
            genmoves(game, moves)
            move = rand(moves)
        end
        game = play(game, move)
        f, r = iswon(game)
    end
    return r
end

function test_net(actor, ngames=100)
    v = 0
    n = 0
    d = 0
    for k in 1:ngames
        r = game_vs_rand(actor)
        if r == 1
            v += 1
        elseif r == 0
            n += 1
        else
            d += 1
        end
        if k % 100 == 0
            println("victoires: $v, nulles: $n, défaites: $d sur $k matchs")
        end
    end
    println("victoires: $v, nulles: $n, défaites: $d")
end


function read_game(fileName::String, max=10)
    io = open(fileName, "r")
    games = []
    cpt = 0
    rolling_mean = [0.0f0, 0.0f0]
    while !eof(io)
        player = read(io, Int8)
        nextboard = read(io, Int8)
        score = read(io, Int32)
        result = read(io, Int8)
        gsquare = read(io, Int8)
        lsquare = read(io, Int8)


        boardx = zeros(UInt16, 10)
        boardo = zeros(UInt16, 10)

        for i in 1:10
            boardx[i] = read(io, Int16)
            boardo[i] = read(io, Int16)
        end

        batch = zeros(Int8, 3, 3, 12)
        for i in 1:9
            for j in 0:8
                x = div(j, 3) + 1
                y = j % 3 + 1
                if boardx[i] & (1 << j) != 0
                    batch[x, y, i] = 1
                end
                if boardo[i] & (1 << j) != 0
                    batch[x, y, i] = -1
                end
            end
        end

        for j in 0:8
            x = div(j, 3) + 1
            y = j % 3 + 1
            if boardx[10] & (1 << j) != 0
                batch[x, y, 10] = 1
            elseif boardo[10] & (1 << j) != 0
                batch[x, y, 10] = -1
            end
        end
        if 8 >= nextboard >= 0
            x = div(nextboard, 3) + 1
            y = nextboard % 3 + 1
            batch[x, y, 11] = 1
        end
        if player == 0
            batch[:, :, 12] .= 1
        else
            batch[:, :, 12] .= -1
        end

        if -1 <= nextboard <= 8
            if player == 1
                score = -score
            end
            push!(
                games,
                (
                    batch, tanh(score / 7600)
                )
            )
            cpt += 1
            if cpt >= max * 10^6
                break
            end
        end
    end
    close(io)
    return games
end




function add_data_Cpp(buffer, a, b)
    newbuff = [Vector{eltype(buffer)}() for k in 1:Threads.nthreads()]
    Threads.@threads for k in a:b
        buff = read_game("/home/fabrice/C++/CodinGame/Data/data$k.dat")
        append!(newbuff[Threads.threadid()], buff)
    end
    for buff in newbuff
        append!(buffer, buff)
    end
end

function retrain(actor, n, lr=0.001)
    for k in 1:n
        buff = read_game("/home/fabrice/C++/CodinGame/Data/data$k.dat")
        traininPipee(1024, actor, buff; lr=lr, epoch=1)
    end
end

function load_data(a, b)
    buffer = []
    for id in a:b
        buff = JLD2.load(
            "/home/fabrice/Julia_Files/UBFM/UBFM-clean/data/samples_uttt/data$id.jld2",
            "data"; typemap=Dict("Main.UBFM.Sample" => Sample, "Main.UBFM.Game.GameState" => GameState)
        )
        for s in buff
            push!(buffer, (vectorize(s.state)..., s.value * s.state.player))
        end

    end
    buffer
end

function trainnnue(net, n, lr=0.001, batch_size=1024)
    ndatas = 0
    for k in 1:100:n
        buffer = load_data(k, min(k + 99, n))
        ndatas += length(buffer)
        loss = sgdtrain(net, buffer, lr, batch_size)
        println("datas so far: $ndatas")
        println("loss: $loss")

    end
end

function weights_range(net)
    wmin = 0
    wmax = 0
    for w in net
        wmin = min(wmin, minimum(w.weight), minimum(w.bias))
        wmax = max(wmax, maximum(w.weight), maximum(w.bias))
    end
    return wmin, wmax
end

function compress(w)
    comp = UInt16[]
    for k in 1:2:length(w)
        push!(comp, (w[k] << 8) | w[k+1])
    end
    return comp
end

function weight_to_float16(d)
    range = weights_range(d)
    m = round(Int, abs(range[1]) + 1)
    mm = 0
    println("range: $m")
    w = round.(UInt16, (127 * d[1].weight .+ 10000))
    ws = transcode(String, reshape(w, length(w)))
    b = round.(UInt16, (127 * d[1].bias .+ 10000))
    bs = transcode(String, reshape(b, length(b)))
    mm = max(mm, maximum(w), maximum(b))
    open("/home/fabrice/C++/CodinGame/weights.dat", "w") do io
        write(io, "wstring w0=L\"")
        write(io, ws)
        write(io, "\";\n")
        write(io, "wstring b0 =L\"")
        write(io, bs)
        write(io, "\";\n")
    end

    for i in 2:2
        w = round.(UInt16, 5000 * (d[i].weight .+ m))
        b = round.(UInt16, 5000 * (d[i].bias .+ m))
        mm = max(mm, maximum(w), maximum(b))

        ws = transcode(String, reshape(transpose(w), length(w)))

        bs = transcode(String, reshape(b, length(b)))

        id = i - 1
        open("/home/fabrice/C++/CodinGame/weights.dat", "a") do io
            write(io, "wstring w$id =L\"")
            write(io, ws)
            write(io, "\";\n")

            write(io, "wstring b$id =L\"")
            write(io, bs)
            write(io, "\";\n")
        end
    end
    println("max: $mm")
end

function weight_to_float16_comp(d)
    range = weights_range(d)
    m = round(Int, abs(range[1]) + 1)
    mm = 0
    println("range: $m")
    w = round.(UInt16, (127 * d[1].weight .+ 128))
    ws = transcode(String, compress(reshape(w, length(w))))
    b = round.(UInt16, (127 * d[1].bias .+ 128))
    bs = transcode(String, compress(reshape(b, length(b))))
    mm = max(mm, maximum(w), maximum(b))
    open("/home/fabrice/C++/CodinGame/weights.dat", "w") do io
        write(io, "wstring w0=L\"")
        write(io, ws)
        write(io, "\";\n")
        write(io, "wstring b0 =L\"")
        write(io, bs)
        write(io, "\";\n")
    end

    for i in 2:2
        w = round.(UInt16, 5000 * (d[i].weight .+ m))
        b = round.(UInt16, 5000 * (d[i].bias .+ m))
        mm = max(mm, maximum(w), maximum(b))

        ws = transcode(String, reshape(transpose(w), length(w)))

        bs = transcode(String, reshape(b, length(b)))

        id = i - 1
        open("/home/fabrice/C++/CodinGame/weights.dat", "a") do io
            write(io, "wstring w$id =L\"")
            write(io, ws)
            write(io, "\";\n")

            write(io, "wstring b$id =L\"")
            write(io, bs)
            write(io, "\";\n")
        end
    end
    println("max: $mm")
end

end
# function main(ttime, timein, tkingtime, generation)
#     if generation != 0
#         @load "/home/fabrice/Julia_Files/UBFM/Games/" * NAME * "/Data/reseau11x11_mul$generation.json" reseau
#         actor = reseau |> gpu
#         #@load "/home/fabrice/Julia_Files/UBFM/Games/NAME/Data/reseau11x11_test31.json" reseau
#         #actor2=resnetb_2H(128,10,128)
#         #actor2=reseau|>gpu
#         actor2 = deepcopy(actor)
#         testmode!(actor, true)
#         testmode!(actor2, true)
#     else
#         @load "/home/fabrice/Julia_Files/UBFM/Games/" * NAME * "/Data/reseau_bsf.json" reseau
#         actor = reseau |> gpu
#         #actor=resnetb(96,6,128)
#         actor2 = deepcopy(actor)
#         #actor=densenetwork()
#         #actor=mobilenet(128,48,5,32)
#         testmode!(actor, true)
#     end
#     #actor2=resnet(83,4,74)
#     selfplay(ttime, timein, actor, actor2, tkingtime)
# end

# #main(parsed_args["ttime"],parsed_args["timein"],parsed_args["tkingtime"],parsed_args["generation"])

# function test()
#     S = [Dict()]
#     T = [Dict()]
#     batch = [zeros(Float32, 6, 6, 3, 36)]
#     actor = resnetb_2H_gen(64, 5, 128)
#     l = OneGame(actor, T, S, 1, 0, 1, batch)
#     return l
# end
