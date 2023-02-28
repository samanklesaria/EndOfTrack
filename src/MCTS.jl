module MCTS
using Infiltrator 
using StaticArrays
using Accessors
using Random
using Functors
using ArraysOfArrays
using ThreadTools
using StatsBase
using LogExpFunctions
using Unrolled

using Plots
using StatsBase: mean

export test, ab_game_lengths, simulate, start_state, Rand,
  AlphaBeta

const VALIDATE=true;

include("rules.jl")
include("util.jl")
include("searches.jl")
include("groupops.jl")
include("tests.jl")

const discount = 0.99f0
const inv_discount = 1/discount

function actions(st)
  [apply_hueristic(st, a) for a in Iterators.flatten(
    (piece_actions(st), ball_actions(st)))]
end

function ordered_actions(st)
  sort(actions(st); by=a-> abs(a.value), rev=true) 
end

function apply_hueristic(st, a)
  term = is_terminal(unchecked_apply_action(st, a))
  if !term
    return ValuedAction(a, 0)
  else
    return ValuedAction(a, st.player == 1 ? 1 : -1)
  end
end

function shuffled_actions(st)
  acts = actions(st)
  Random.shuffle!(acts)
  sort(acts; by=a-> abs.(a.value), rev= true, alg=MergeSort) 
end

struct Rand end

function (::Rand)(st::State)
  choices = actions(st)
  choices[rand(1:length(choices))]
end

struct Boltzmann
  temp::Float32
end

function (b::Boltzmann)(st::State)
  choices = actions(st)
  player = st.player == 1 ? 1 : -1
  probs = [player .* c.value ./ b.temp for c in choices]
  softmax!(probs)
  sample(choices, Weights(probs))
end

struct Greedy end

function (b::Greedy)(st::State)
  choices = actions(st)
  player = st.player == 1 ? 1 : -1
  probs = [player * c.value for c in choices]
  ix = argmax(probs)
  mask = (probs .== probs[ix])
  rand(choices[mask])
end

apply_action(st::State, a::Action) = unchecked_apply_action(st, a)

function apply_action(st::State, va::ValuedAction)
  new_st = unchecked_apply_action(st, va.action)
  if VALIDATE
    try
      assert_valid_state(new_st)
    catch exc
      println(st)
      log_action(st, va)
      rethrow() 
    end
  end
  new_st
end

function unchecked_apply_action(st::State, a::Action)
  if a[1] < 6
    pieces = st.positions[st.player].pieces
    pmat = MMatrix{2,5}(pieces) 
    pmat[:, a[1]] .= a[2]
    @set st.positions[st.player].pieces = SMatrix{2,5}(pmat)
  else
    @set st.positions[st.player].ball = a[2]
  end
end

struct EndState
  winner::Union{Int8, Nothing}
  st::State
  steps::Int
end

# function simulate(st::State, players; steps=300, log=false)
#   # if log
#   #   println("Simulating on thread $(Threads.threadid())")
#   # end
#   for nsteps in 0:steps
#     if is_terminal(st)
#       return EndState(next_player(st.player), st, nsteps)
#     end
#     action = players[st.player](st)
#     if log
#       log_action(st, action)
#     end
#     st = apply_action(st, action)
#     st = @set st.player = next_player(st.player) 
#   end
#   return EndState(nothing, st, steps)
# end

function simulate(st::State, players; steps=300, log=false)
  player_ixs = st.player == 1 ? (0=>1,1=>2) : (0=>2,1=>1)
  if log
    println("Simulating on thread $(Threads.threadid())")
  end
  simulate_(st, players, player_ixs, steps, log)
end

@unroll function simulate_(st::State, players, player_ixs, steps, log)
  for nsteps in 0:steps
    @unroll for (substep, player) in player_ixs
      st = @set st.player = player
      if is_terminal(st)
        return EndState(next_player(player), st, 2 * nsteps + substep)
      end
      action = players[player](st)
      if log
        log_action(st, action)
      end
      st = apply_action(st, action)
    end
  end
  return EndState(nothing, st, steps)
end

mutable struct Edge
  action::Action
  q::Float32
  n::Int
end

ValuedAction(e::Edge) = ValuedAction(e.action, e.q / e.n)

struct BackEdge
  ix::Int8
  state::State
  trans::Transformation
end

mutable struct Node
  last_access::Int
  counts::Int
  edges::Vector{Edge}
  parents::Set{BackEdge}
end

ucb(n::Int, e::Edge) = (e.q / e.n) + sqrt(2) * sqrt(log(n) / e.n)

Base.@kwdef mutable struct MC
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, Node} = Dict{State, Node}()
  cache_lim::Int = 100
  steps::Int = 100
end

# TODO: need to invalidate parent pointers too
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

# ALSO: What if we just did the simple AlphaGo heuristic instead?

# In a multi-threaded context, can use relativistic time: (thread, time) tuple. 

# Why is it only 0.93 for a move to (4,8)?
# We should have it hard coded to always prefer moves that end the game


# TODO: The first time the parent size goes above 1, pause to check that
# this makes sense. 

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
      edges = [rollout(nst, a) for a in actions(nst)]
      # dedent!()
      total_q = sum(e.q for e in edges)
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = Node(mcts.time, 1, edges, parents)
      backprop(mcts, nst, discount * total_q, length(edges))
      return
    end
  end
end

# It also seems like batching backprop might be useful.
# Many of your paths will go through the same node, so
# rather than doing them individually, do them all at once.
# Instead of a queue, you'll need to use a hashtable. 


# But also: what's really taking so much time?
# How long are the backprop steps? What's the branching factor?



# It seems like this is taking a very long time. 
# Why? are there loops in the back-edges?
# Doing gc would help matters, so there's less parents to follow.

function backprop(mcts::MC, st::State, q::Float32, n::Int)
  node = mcts.cache[st]
  to_process = [(p,q,node) for p in node.parents]
  while length(to_process) > 0
    b, q, child = pop!(to_process)
    if !haskey(mcts.cache, b.state)
      delete!(child.parents, b.state)
    end
    node = mcts.cache[b.state]
    edge = node.edges[b.ix]
    trans_q = b.trans.value_map * q
    edge.q += trans_q
    edge.n += n
    # if length(node.parents) > 9
    #   println("Parent size ", length(node.parents))
    # end
    for p in node.parents
      push!(to_process, (p, discount * trans_q, node))
    end
  end
end

const rand_players = (Rand(), Rand())
const greedy_players = (Greedy(), Greedy())

function rollout(st::State, a::ValuedAction)
  # printindent("Starting Rollout of ")
  # log_action(st, a)
  next_st = @set apply_action(st, a).player = 2
  endst = simulate(next_st, greedy_players; steps=10)
  if isnothing(endst.winner)
    endq = 0f0
  elseif endst.winner == 1
    endq = 1f0
  else
    endq = -1f0
  end
  q = (discount ^ endst.steps) * endq
  # printindent("$(endst.steps) step rollout: ")
  # log_action(st, ValuedAction(a.action, q))
  Edge(a.action, q, 1)
end

function (mcts::MC)(st::State)
  mcts.last_move_time = mcts.time
  trans, nst = normalized(st)
  for _ in 1:mcts.steps
    expand_leaf!(mcts, nst)
  end
  if true # length(mcts.cache) > mcts.cache_lim
    gc!(mcts)
  end
  edges = mcts.cache[nst].edges
  # println("Options:")
  # indent!()
  # for e in edges
  #   printindent("")
  #   log_action(st, ValuedAction(e))
  # end
  # dedent!()
  trans(ValuedAction(edges[argmax([e.q / e.n for e in edges])]))
end

function rand_game_lengths()
  results = tmap(_->simulate(start_state, rand_players), 8, 1:10)
  println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
  histogram([r.steps for r in results if !isnothing(r.winner)])
end

function ab_game_lengths()
  results = tmap(_->simulate(start_state, (AlphaBeta(3), Rand())), 8, 1:10)
  println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
  win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
  println("Average winner was $win_avg") 
  histogram([r.steps for r in results if !isnothing(r.winner)])
end

function mc_game_lengths()
  results = tmap(_->simulate(start_state, (MC(), Rand()); steps=300), 8, 1:10)
  println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
  win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
  println("Average winner was $win_avg") 
  histogram([r.steps for r in results if !isnothing(r.winner)])
end

function blah()
  simulate(start_state, (MC(steps=20), AlphaBeta(3)); steps=300, log=true)
end

# Potential things:
# The GC could prune earler, by removing everything not touched by the chosen action
# We could only update backwards on the actual path taken rather than all parents. 
# Or, a middle ground: only keep around the past k parents. 
# Could also not do state sharing.


# TODO
# Double Q learning (rather than the simple Q learning you're doing here)
# AlphaGo setup

# For MCTS, no need to expand all the nodes. Initialize them all
# with their hueristic values, and go from there

# Hueristics:
# Position of the ball? Highest player position?

# Optimizations:
# Encode and use bitvec instead of Dict for ball passing
# Make simulate know the types of the players (by unrolling the loop and using tuples)
# Disable bounds checks
# Do stuff in parallel (and benchmark it)
# Benchmark
# Look at tools for type stability warnings in Julia

# Fun:
# Add GUI for human player 

end # module MCTS
