module MCTS
using BSON: @save
using Lux, NNlib, Zygote, Optimisers, Functors, StatsBase, DataStructures
using StaticArrays, Accessors, Random, ArraysOfArrays, Unrolled
using Infiltrator 
using ThreadTools
using VisdomLog

using Plots
using StatsBase: mean

export test, simulate, start_state, Rand,
  AlphaBeta

const VALIDATE=false;

include("rules.jl")
include("util.jl")
include("searches.jl")
include("groupops.jl")
include("tests.jl")
include("nn.jl")

const discount = 0.99f0
const inv_discount = 1/discount

mutable struct Neural{NN, P, S}
  net::NN
  ps::P
  st::S
  temp::Float32
end

struct Greedy end

function (b::Greedy)(st::State)
  acts = actions(st)
  choices = ValuedAction[apply_hueristic(st, a) for a in acts]
  player = st.player == 1 ? 1 : -1
  probs = [player * c.value for c in choices]
  ix = argmax(probs)
  mask = (probs .== probs[ix])
  rand(choices[mask])
end

const greedy_players = (Greedy(), Greedy())

function (b::Neural)(st::State)
  choices = actions(st)
  batch = cat(as_pic.(apply_action.(st, choices)); dims=4)
  player = st.player == 1 ? 1 : -1
  values, _ = Lux.apply(b.net, batch, b.ps, b.st)
  normalized_values = player .* values .* b.temp
  softmax!(normalized_values)
  ix = sample(1:length(choices), Weights(normalized_values))
  ValuedAction(choices[ix], values[ix])
end

apply_action(st::State, a::ValuedAction) = apply_action(st, a.action)

function apply_action(st::State, (pieceix, pos)::Action)
  if pieceix < 6
    pieces = (st.positions[st.player].pieces)::SMatrix{2, 5, Int8}
    ix = LinearIndices(pieces)
    pmat = setindex(setindex(pieces, pos[1], ix[1, pieceix]), pos[2], ix[2, pieceix])
    @set st.positions[st.player].pieces = pmat
  else
    @set st.positions[st.player].ball = pos
  end
end

struct EndState
  winner::Union{Int8, Nothing}
  st::State
  steps::Int
  states::Vector{State}
end

function simulate(st::State, players; steps=300, log=false, track=false)
  player_ixs = st.player == 1 ? (0=>1,1=>2) : (0=>2,1=>1)
  simulate_(st, players, player_ixs, steps, log, track)
end

@unroll function simulate_(st::State, players, player_ixs, steps, log, track)
  states = Vector{State}()
  for nsteps in 0:steps
    @unroll for (substep, player) in player_ixs
      st = @set st.player = player
      if is_terminal(st)
        return EndState(next_player(player), st, 2 * nsteps + substep, states)
      end
      action = players[player](st)
      if log
        log_action(st, action)
      end
      st = apply_action(st, action)
      if track
        push!(states, st)
      end
    end
  end
  return EndState(nothing, st, steps, states)
end

mutable struct Edge
  action::Action
  q::Float32
  n::Int
end

ValuedAction(e::Edge) = ValuedAction(e.action, e.q)

struct BackEdge
  ix::Int8
  state::State
  trans::Transformation
end

Base.hash(a::BackEdge, h::UInt) = hash(a.state, h)

Base.:(==)(a::BackEdge, b::BackEdge) = a.state == b.state

mutable struct Node
  last_access::Int
  counts::Int
  edges::Vector{Edge}
  parents::Set{BackEdge}
end

ucb(n::Int, e::Edge) = e.q + sqrt(2) * sqrt(log(n) / e.n)

Base.@kwdef mutable struct MC{P}
  players::P
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, Node} = Dict{State, Node}()
  steps::Int = 100
end

function gc!(mc::MC)
  to_delete = Vector{State}()
  for (k,v) in mc.cache
    if v.last_access < mc.last_move_time
      push!(to_delete, k)
    end
  end
  for k in to_delete
    delete!(mc.cache, k)
  end
end

function expand_leaf!(mcts, nst::State)
  parent_key = nothing
  mcts.time += 1
  while true
    if haskey(mcts.cache, nst)
      c = mcts.cache[nst]
      if c.last_access == mcts.time
        return
      end
      c.last_access = mcts.time
      if !isnothing(parent_key)
        push!(c.parents, parent_key)
      end
      ix = argmax([ucb(c.counts, e) for e in c.edges])
      # print("Traversing Edge ")
      # log_action(nst, ValuedAction(c.edges[ix].action, 0))
      next_st = @set apply_action(nst, c.edges[ix].action).player = 2
      if is_terminal(next_st)
        return
      end
      trans, new_nst = normalized(next_st)
      parent_key = BackEdge(ix, nst, trans)
      nst = new_nst
    else
      # println("Expanding Leaf")
      # indent!()
      edges = Edge[rollout(nst, a, mcts.players) for a in actions(nst)]
      # dedent!()
      max_q = maximum(e.q for e in edges)
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = Node(mcts.time, 1, edges, parents)
      backprop(mcts, nst, discount * max_q, length(edges))
      return
    end
  end
end

function backprop(mcts::MC, st::State, q::Float32, n::Int)
  to_process = DefaultDict{State, Float32}(0f0)
  to_process[st] = q
  while length(to_process) > 0
    (st, q) = pop!(to_process)
    node = mcts.cache[st]
    for p in node.parents 
      if !haskey(mcts.cache, p.state)
        delete!(node.parents, p.state)
      else 
        newq = discount * p.trans.value_map * q
        edge = mcts.cache[p.state].edges[p.ix]
        edge.n += n
        edge.q = max(edge.q, newq)
        to_process[p.state] = max(to_process[p.state], newq)
      end
    end
  end
end

function rollout(st::State, a::Action, players)
  # printindent("Starting Rollout of ")
  # log_action(st, a)
  next_st = @set apply_action(st, a).player = 2
  endst = simulate(next_st, players; steps=20)
  if isnothing(endst.winner)
    endq = 0f0
  elseif endst.winner == 1
    endq = 1f0
  else
    endq = -1f0
  end
  q = (discount ^ endst.steps) * endq
  # printindent("$(endst.steps) step rollout: ")
  # log_action(st, ValuedAction(a, q))
  Edge(a, q, 1)
end

function (mcts::MC)(st::State)
  mcts.last_move_time = mcts.time
  trans, nst = normalized(st)
  for _ in 1:mcts.steps
    expand_leaf!(mcts, nst)
  end
  gc!(mcts)
  edges = mcts.cache[nst].edges
  # println("Options:")
  # indent!()
  # for e in edges
  #   printindent("")
  #   log_action(st, ValuedAction(e))
  # end
  # dedent!()
  trans(ValuedAction(edges[argmax([e.q for e in edges])]))
end

# function rand_game_lengths()
#   results = tmap(_->simulate(start_state, rand_players), 8, 1:10)
#   println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
#   histogram([r.steps for r in results if !isnothing(r.winner)])
# end

# function ab_game_lengths()
#   results = tmap(_->simulate(start_state, (AlphaBeta(3), Rand())), 8, 1:10)
#   println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
#   win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
#   println("Average winner was $win_avg") 
#   histogram([r.steps for r in results if !isnothing(r.winner)])
# end

# function mc_game_lengths()
#   results = tmap(_->simulate(start_state, (MC(), Rand()); steps=300), 8, 1:10)
#   println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
#   win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
#   println("Average winner was $win_avg") 
#   histogram([r.steps for r in results if !isnothing(r.winner)])
# end

# function blah()
#   simulate(start_state, (MC(steps=50), AlphaBeta(3)); steps=300, log=true)
# end


# Potential things:
# The GC could prune earler, by removing everything not touched by the chosen action
# We could only update backwards on the actual path taken rather than all parents. 
# Or, a middle ground: only keep around the past k parents. 
# Could also not do state sharing.


# TODO
# Double Q learning (rather than the simple Q learning you're doing here)
# AlphaGo setup

# Hueristics:
# Position of the ball? Highest player position?

# Optimizations:
# Encode and use bitvec instead of Dict for ball passing
# Disable bounds checks
# Do stuff in parallel (and benchmark it)

# Fun:
# Add GUI for human player 

end # module MCTS
