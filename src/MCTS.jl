module MCTS
using Infiltrator 
using StaticArrays
using Accessors
using Random
using Functors
using ArraysOfArrays
using ThreadTools

using Plots
using StatsBase: mean

export test, ab_game_lengths, simulate, start_state, rand_policy,
  AlphaBeta, precomp

include("util.jl")

const indent_level = Ref(0)

const VALIDATE=true;

indent!() = indent_level[] += 1

function dedent!()
  @assert indent_level[] > 0
  indent_level[] -= 1
end

function printindent(a)
  print(String(fill(' ', 4 * indent_level[])))
  print(a)
end

const limits = @SVector [7, 8]

const discount = 0.99f0
const inv_discount = 1/discount

const Pos = SVector{2, Int8}

struct PlayerState
  ball::Pos
  pieces::SMatrix{2,5, Int8}
end
@functor PlayerState

struct State
  player::Int
  positions::SVector{2, PlayerState}
end
@functor State (positions,)

const start_state = State(1, 
  SVector{2}([
    PlayerState(
      SVector{2}([4,1]),
      SMatrix{2,5}(Int8[collect(2:6) fill(1, 5)]')),
    PlayerState(
      SVector{2}([4,8]),
      SMatrix{2,5}(Int8[collect(2:6) fill(8, 5)]'))
      ]))
 
function is_terminal(state)
  state.positions[1].ball[2] == limits[2] || state.positions[2].ball[2] == 1
end

function occupied(state, y)
  inner(p::SMatrix) = any(all(p .== y[:, na]; dims=1))
  inner(_) = false
  foldmap(inner, Base.:|, false, state)
end

struct PotentialMove
  y::Pos
  norm::Int8
end

function lub(a::PotentialMove, b::PotentialMove)
  if a.norm < b.norm
    return a
  end
  if a.norm == b.norm
    error("Two pieces at the same position: ", a.y, ", ", b.y)
  end
  return b
end

const PieceIx = UInt8
const Action = Tuple{PieceIx, Pos}

struct ValuedAction
  action::Union{Action, Nothing}
  value::Float32
end

function piece_actions(st::State)
  actions = Vector{Action}()
  ball_pos = st.positions[st.player].ball
  for i in 1:5
    x = st.positions[st.player].pieces[:, i]
    if any(x .!= ball_pos)
      for move in (SVector{2}([1,2]), SVector{2}([2,1]))
        for d1 in (-1, 1)
          for d2 in (-1, 1)
            pos = x .+ move .* (@SVector [d1, d2])
            if all(pos .>= 1) && all(pos .<= limits) && !occupied(st, pos)
              push!(actions, (i, pos))
            end
          end
        end
      end
    end
  end
  actions
end

function ball_actions(st::State)
  y = st.positions[st.player].ball
  passes = Set{Pos}([y])
  balls = [y]
  while length(balls) > 0
    x = pop!(balls)
    for move in pass_actions(st, x)
      yy = move.y
      if yy ∉ passes 
        push!(passes, yy)
        push!(balls, yy)
      end
    end
  end
  pop!(passes, y)
  [(UInt8(6), p) for p in passes]
end

next_player(player::Int) = (1 ⊻ (player - 1)) + 1

const Dir = Pos

function encode(pieces)
  (pieces[2,:] .- 1) .* limits[1] .+ pieces[1,:]
end

function assert_valid_state(st::State)
  for i in (1,2)
    encoded = encode(st.positions[i].pieces)
    @assert length(encoded) == length(unique(encoded))
    ball_pos = encode(st.positions[i].ball[:, na])
    @assert ball_pos[1] in encoded
    @assert all(st.positions[i].pieces .>= 1)
    @assert all(st.positions[i].pieces .<= limits[:, na])
  end
end

function pass_actions(st::State, x::Pos)
  if VALIDATE
    assert_valid_state(st)
  end
  moves = Dict{Dir, PotentialMove}()
  for player in (st.player, next_player(st.player))
    vecs = st.positions[player].pieces .- x[:, na]
    absvecs = abs.(vecs)
    diagonals = absvecs[1, :] .== absvecs[2, :]
    verticals = vecs[1, :] .== 0
    horizontals = vecs[1, :] .== 0
    norms = maximum(absvecs; dims=1)[1, :]
    nz = norms .> 0
    valid = (diagonals .| verticals .| horizontals) .& nz
    norms = norms[valid]
    units = vecs[:, valid] .÷ norms[na, :]
    y = st.positions[player].pieces[:, valid]
    if player == st.player
      for i in 1:size(units, 2)
        u = SVector{2}(units[:, i])
        if haskey(moves, u)
          moves[u] = lub(moves[u], PotentialMove(y[:,i], norms[i]))
        else
          moves[u] = PotentialMove(y[:,i], norms[i])
        end
      end
    else
      for i in 1:size(units, 2)
        u = SVector{2}(units[:, i])
        if haskey(moves, u) && norms[i] < moves[u].norm
          pop!(moves, u)
        end
      end
    end
  end
  values(moves)
end

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


function rand_policy(st::State)
  choices = actions(st)
  choices[rand(1:length(choices))]
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

function log_action(st, action_val::ValuedAction)
  action = action_val.action 
  val = action_val.value
  if action[1] < 6
    old_pos = st.positions[st.player].pieces[:, action[1]]
    println("[$val] $(st.player) moves from $old_pos to $(action[2])")
  else
    old_pos = st.positions[st.player].ball
    println("[$val] $(st.player) kicks from $old_pos to $(action[2])")
  end
end

struct EndState
  winner::Union{Int8, Nothing}
  st::State
  steps::Int
end

function simulate(st::State, players; steps=300, log=false)
  if log
    println("Simulating on thread $(Threads.threadid())")
  end
  for nsteps in 0:steps
    if is_terminal(st)
      return EndState(next_player(st.player), st, nsteps)
    end
    action = players[st.player](st)
    if log
      log_action(st, action)
    end
    st = apply_action(st, action)
    st = @set st.player = next_player(st.player) 
  end
  return EndState(nothing, st, steps)
end

larger_q(a, b) = a.value > b.value ? a : b

function cached_max_action(st::State, depth::Int, cache::Dict)
  trans, nst = normalized(st)
  if is_terminal(nst)
    trans(ValuedAction(nothing, -1))
  elseif haskey(cache, nst)
    println("Reusing cache")
    trans(cache[nst])
  elseif depth == 0
    chosen = rand_policy(nst)
    trans(ValuedAction(chosen.action, 0))
  else
    best_child = mapreduce(larger_q, shuffled_actions(nst)) do a
      next_st = @set apply_action(nst, a).player = 2
      child_val = cached_max_action(next_st, depth - 1, cache).value
      ValuedAction(a.action, discount * child_val)
    end
    cache[nst] = best_child
    trans(best_child)
  end
end

function plot_state(st::State)
  scatter(st.positions[1].pieces[1, :],
    st.positions[1].pieces[2, :])
  scatter!(st.positions[1].ball[1:1], st.positions[1].ball[2:2])
  scatter!(st.positions[2].pieces[1,:],
    st.positions[2].pieces[2, :])
  scatter!(st.positions[2].ball[1:1], st.positions[2].ball[2:2])
end

struct CachedMinimax
  depth::Int
end

function (mm::CachedMinimax)(st::State)
  cache = Dict{State, ValuedAction}()
  cached_max_action(st, mm.depth, cache)
end

const eps = 1e-3
const no_min_action = ValuedAction(nothing, 1 + eps)
const no_max_action = ValuedAction(nothing, -1 - eps)

Base.:*(a::Number, b::ValuedAction) = ValuedAction(b.action, a * b.value)

function min_action(st, alpha::ValuedAction, beta::ValuedAction, depth)
  if depth == 0
    return ValuedAction(rand_policy(st).action, 0)
  end
  for a in shuffled_actions(st) 
    next_st = @set apply_action(st, a).player = next_player(st.player)
    if is_terminal(next_st)
      return ValuedAction(a.action, -1)
    else
      lb = discount * max_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1)
      if lb.value < beta.value
        beta = ValuedAction(a.action, lb.value)
        if alpha.value > beta.value
          return alpha
        end
        if alpha.value == beta.value
          return beta
        end
      end
    end
  end
  beta
end

function max_action(st, alpha, beta, depth)
  if depth == 0
    return ValuedAction(rand_policy(st).action, 0)
  end
  for a in shuffled_actions(st)
    next_st = @set apply_action(st, a).player = next_player(st.player)
    if is_terminal(next_st)
      return ValuedAction(a.action, 1)
    else
      ub = discount * min_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1)
      the_action = ValuedAction(a.action, ub.value)
      if ub.value > alpha.value
        alpha = the_action
        if alpha.value > beta.value
          return beta
        end
        if alpha.value == beta.value
          return alpha
        end
      end
    end
  end
  alpha
end

struct AlphaBeta
  depth::Int
end

function (ab::AlphaBeta)(st)
  if st.player == 1
    max_action(st, no_max_action, no_min_action, ab.depth)
  else
    min_action(st, no_max_action, no_min_action, ab.depth)
  end
end

abstract type GroupElt end

struct TokenPerm <: GroupElt
  perm::SVector{5, UInt8}
end

struct FlipHor <: GroupElt end

struct FlipVert <: GroupElt end

const mean_vert = SVector{2, Int8}([4, 0])
const flip_vert = SVector{2, Int8}([-1, 1])

const mean_hor = SVector{2}([0, 4.5])
const flip_hor = SVector{2}([1, -1])

flip_pos_vert(a::Pos) = flip_vert .* (a .- mean_vert) .+ mean_vert
flip_pos_vert(a::SMatrix) = flip_vert[:, na] .* (a .- mean_vert[:, na]) .+ mean_vert[:, na]

flip_pos_hor(a::Pos) = Pos(flip_hor .* (a .- mean_hor) .+ mean_hor)
flip_pos_hor(a::SMatrix) = SMatrix{2,5, Int8}(flip_hor[:, na] .* (a .- mean_hor[:, na]) .+ mean_hor[:, na])

function group_op(::FlipVert, a::Action)
  @set a[2] = flip_pos_vert(a[2])
end

function group_op(::FlipHor, a::Action)
  @set a[2] = flip_pos_hor(a[2])
end

function group_op(p::TokenPerm, a::Action) 
  if a[1] < 6
    @set a[1] = p.perm[a[1]]
  else
    a
  end
end

ismat(::SMatrix) = true
ismat(_) = false

struct Transformation
  action_map::Vector{GroupElt}
  value_map::Int
end

function normalized(st)
  action_map = GroupElt[]
  value_map = 1
  
  # Swap players so that the current player is 1.
  if st.player == 2
    st = fmap(flip_pos_hor, st)
    st = State(1, reverse(st.positions))
    value_map = -1
    push!(action_map, FlipHor())
  end
  
  # # Flip the board left or right
  st2 = fmap(flip_pos_vert, st)
  if hash(st2) > hash(st)
    st = st2
    push!(action_map, FlipVert())
  end
  
  # # Sort tokens of each player
  ixs = fmapstructure(a->sortperm(nestedview(a)), st; exclude=ismat)
  st = fmap(st, ixs) do a, ix
    ix === (()) ? a : a[:, ix]
  end
  push!(action_map, TokenPerm(invperm(ixs.positions[1].pieces)))
  
  if VALIDATE
    assert_valid_state(st)
  end
  Transformation(action_map, value_map), st
end

function (t::Transformation)(a::ValuedAction)
  ValuedAction(t(a.action), t.value_map * a.value)
end

(t::Transformation)(a::Action) = foldr(group_op, t.action_map; init=a)
(t::Transformation)(::Nothing) = nothing

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
  counts::Int
  edges::Vector{Edge}
  parents::Set{BackEdge}
end

ucb(n::Int, e::Edge) = (e.q / e.n) + sqrt(2) * sqrt(log(n) / e.n)

struct MC
  cache::Dict{State, Node}
  steps::Int
end

MC(steps::Int) = MC(Dict{State,Node}(), steps)

function expand_leaf!(mcts, nst::State)
  parent_key = nothing
  while true
    if haskey(mcts.cache, nst)
      c = mcts.cache[nst]
      if !isnothing(parent_key)
        push!(c.parents, parent_key)
      end
      ix = argmax([ucb(c.counts, e) for e in c.edges])
      next_st = @set apply_action(nst, c.edges[ix].action).player = 2
      if is_terminal(next_st)
        return
      end
      trans, new_nst = normalized(next_st)
      parent_key = BackEdge(ix, nst, trans)
      nst = new_nst
    else
      edges = [rollout(nst, a) for a in actions(nst)]
      total_q = sum(e.q for e in edges)
      parents = isnothing(parent_key) ? Set() : Set([parent_key])
      mcts.cache[nst] = Node(1, edges, parents)
      backprop(mcts, nst, discount * total_q, length(edges))
      return
    end
  end
end

function backprop(mcts::MC, st::State, q::Float32, n::Int)
  node = mcts.cache[st]
  to_process = [(p,q) for p in node.parents]
  while length(to_process) > 0
    b, q = pop!(to_process)
    edge = mcts.cache[b.state].edges[b.ix]
    edge.q += b.trans.value_map * q
    edge.n += n
    for p in mcts.cache[b.state].parents
      push!(to_process, (p, discount * q))
    end
  end
end

const rand_players = fill(rand_policy, 2)

function rollout(st::State, a::ValuedAction)
  next_st = @set apply_action(st, a).player = 2
  endst = simulate(next_st, rand_players)
  if isnothing(endst.winner)
    endq = 0f0
  elseif endst.winner == 1
    endq = 1f0
  else
    endq = -1f0
  end
  q = (discount ^ endst.steps) * endq
  # print("Rollout result ")
  # log_action(st, ValuedAction(a.action, q))
  Edge(a.action, q, 1)
end

function (mcts::MC)(st::State)
  trans, nst = normalized(st)
  for _ in 1:mcts.steps
    expand_leaf!(mcts, nst)
  end
  edges = mcts.cache[nst].edges
  trans(ValuedAction(edges[argmax([e.q for e in edges])]))
end

function rand_game_lengths()
  results = tmap(_->simulate(start_state, [rand_policy, rand_policy]), 8, 1:10)
  println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
  histogram([r.steps for r in results if !isnothing(r.winner)])
end

function ab_game_lengths()
  results = tmap(_->simulate(start_state, [AlphaBeta(3), rand_policy]), 8, 1:10)
  println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
  win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
  println("Average winner was $win_avg") 
  histogram([r.steps for r in results if !isnothing(r.winner)])
end

function mc_game_lengths()
  results = tmap(_->simulate(start_state, [MC(20), rand_policy]; steps=300), 8, 1:10)
  println("$(sum(isnothing(r.winner) for r in results) / 10) were nothing")
  win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
  println("Average winner was $win_avg") 
  histogram([r.steps for r in results if !isnothing(r.winner)])
end


function precomp()
  simulate(start_state, [AlphaBeta(2), rand_policy])
end

# TODO: 
# For CachedMinimax, we can reuse the cache over multiple rounds.
# Tag every node with which of the original actions used it.
# After a round, sweep away nodes that were not used by the chosen action 
# Depth is roughly 45. So in 4 steps, that's 4 million positions

# For MCTS, no need to expand all the nodes. Initialize them all
# with their hueristic values, and go from there

# Hueristics:
# Position of the ball? Highest player position?

# Optimizations:
# Encode and use bitvec instead of Dict for ball passing
# Disable all the validation checks


include("tests.jl")

end # module MCTS
