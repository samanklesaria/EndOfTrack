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
  AlphaBeta

include("util.jl")

const limits = @SVector [7, 8]

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
      SVector{2}([4,7]),
      SMatrix{2,5}(Int8[collect(2:6) fill(7, 5)]'))
      ]))
 
function is_terminal(state)
  state.positions[1].ball[2] == limits[2] || state.positions[2].ball[2] == 0
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

# Hueristic value is dy
function piece_actions(st::State)
  actions = Vector{ValuedAction}()
  ball_pos = st.positions[st.player].ball
  mlt = st.player == 1 ? 0.2 : -0.2
  for i in 1:5
    x = st.positions[st.player].pieces[:, i]
    if any(x .!= ball_pos)
      for move in (SVector{2}([1,2]), SVector{2}([2,1]))
        for d1 in (-1, 1)
          for d2 in (-1, 1)
            dx = move .* (@SVector [d1, d2])
            pos = x .+ dx
            if all(pos .>= 1) && all(pos .<= limits) && !occupied(st, pos)
              push!(actions, ValuedAction((i, pos), mlt * dx[2]))
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
  mlt = st.player == 1 ? 0.8 : -0.8
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
  [ValuedAction((UInt8(6), p), mlt * (p - y)[2]) for p in passes]
end

next_player(player::Int) = (1 ⊻ (player - 1)) + 1

const Dir = Pos

function pass_actions(st::State, x::Pos)
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
  pa = piece_actions(st)
  append!(pa, ball_actions(st))
  pa
end

function ordered_actions(st)
  acts = actions(st)
  Random.shuffle!(acts)
  sort!(acts; by=a-> a.value, rev=true) 
  acts
end

function rand_policy(st::State)
  choices = actions(st)
  choices[rand(1:length(choices))]
end

function apply_action(st::State, va::ValuedAction)
  a = va.action
  if a[1] < 6
    pieces = st.positions[st.player].pieces
    pmat = Matrix(pieces) 
    pmat[:, a[1]] .= a[2]
    @set st.positions[st.player].pieces = SMatrix{2,5}(pmat)
  else
    @set st.positions[st.player].ball = a[2]
  end
end

function log_action(st, action)
  if action[1] < 6
    old_pos = st.positions[st.player].pieces[:, action[1]]
    println("$(st.player) moves from $old_pos to $(action[2])")
  else
    old_pos = st.positions[st.player].ball
    println("$(st.player) kicks from $old_pos to $(action[2])")
  end
end

function simulate(st::State, players; steps=600)
  println("Simulating on thread $(Threads.threadid())")
  for nsteps in 1:steps
    action = players[st.player](st)
    # log_action(st, action)
    st = apply_action(st, action)
    if is_terminal(st)
      return (st.player, nsteps)
    end
    st = @set st.player = next_player(st.player) 
  end
  return (nothing, steps)
end

larger_q(a, b) = a.value > b.value ? a : b

function cached_max_action(st, depth, cache)
  action_map, value_map, nst = normalized(st)
  if is_terminal(st)
    ValuedAction(nothing, value_map * -1)
  elseif haskey(cache, nst)
    cached = cache[nst]
    ValuedAction(group_ops(cached[1], action_map), value_map * cached[2])
  elseif depth == 0
    chosen = ValuedAction(rand_policy(nst), 0)
    cached[nst] = chosen
    ValuedAction(group_ops(chosen[1], action_map), 0)
  else
    mapreduce(larger_q, ordered_actions(nst)) do a
      next_st = @set apply_action(nst, a).player = 2
      cached_max_action(next_St, depth - 1, cache)
    end
  end
end

struct CachedMinimax
  depth::Int
end

function (mm::CachedMinimax)(st::State)
  cache = Dict{State, ValuedAction}()
  cached_max_action(st, mm.depth, cache)
end

const no_min_action = ValuedAction(nothing, 1f0)
const no_max_action = ValuedAction(nothing, -1f0)

function min_action(st, alpha, beta, depth)
  if is_terminal(st)
    ValuedAction(nothing, 1)
  elseif depth == 0
    rand_policy(st)
  else
    for a in ordered_actions(st) 
      next_st = @set apply_action(st, a).player = next_player(st.player)
      lb = max_action(next_st, alpha, beta, depth - 1)
      if lb.value <= beta.value
        beta = ValuedAction(a.action, lb.value)
        if alpha.value > beta.value
          return alpha
        end
        if alpha.value == beta.value
          return beta
        end
      end
    end
    beta
  end
end

function max_action(st, alpha, beta, depth)
  if is_terminal(st)
    ValuedAction(nothing, -1)
  elseif depth == 0
    rand_policy(st)
  else
    for a in ordered_actions(st)
      next_st = @set apply_action(st, a).player = next_player(st.player)
      ub = min_action(next_st, alpha, beta, depth - 1)
      if ub.value >= alpha.value
        alpha = ValuedAction(a.action, ub.value)
        if alpha.value > beta.value
          return beta
        end
        if alpha.value == beta.value
          return alpha
        end
      end
    end
    alpha
  end
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
  perm::SVector{6, Int8}
end

struct Flip <: GroupElt
end

const mean_pos = SVector{2, Int8}([4, 0])
const flip = SVector{2, Int8}([-1, 1])

flip_vec(a::Pos) = flip .* (a .- mean_pos) .+ mean_pos
flip_vec(a::SMatrix) = flip[:, na] .* (a .- mean_pos[:, na]) .+ mean_pos[:, na]

function group_op(a::Action, ::Flip)
  @set a[2] = flip_vec(a[2])
end

function group_op(a::Action, p::TokenPerm) 
  @set a[1] = p.perm[a[1]]
end

ismat(::SMatrix) = true
ismat(_) = false

function normalized(st)
  action_map = GroupElt[]
  value_map = 1
  
  # Swap players so that the current player is 0.
  # Action map is unchanged, but value functions flip sign
  if st.player == 2
    st = State(1, reverse(st.positions))
    value_map = -1
  end
  
  # Flip the board left or right
  st2 = fmap(flip_vec, st)
  if hash(st2) > hash(st)
    st = st2
    push!(action_map, Flip())
  end
  
  # Sort tokens of each player
  ixs = fmapstructure(a->sortperm(nestedview(a)), st; exclude=ismat)
  st = fmap(st, ixs) do a, ix
    a[:, ix]
  end
  push!(action_map, TokenPerm(invperm(ixs.positions[1].pieces)))
  
  (action_map, value_map, st)
end

group_ops(a::Action, ops::Vector{GroupElt}) = foldl(group_op, ops; init=a)

struct Edge
  action::Action
  q::Float64
  n::Int
end

struct Node
  counts::Int
  edges::Vector{Edge}
  parents::Set{Tuple{Int8, State}}
end

ucb(n::Int, e::Edge) = (e.q / e.n) + sqrt(2) * sqrt(log(n) / e.n)

struct MC
  cache::Dict{State, Node}
  steps::Int
end

function expand_leaf!(mcts, st::State)
  # TODO
end

function (mcts::MC)(st::State)
  for _ in 1:steps
    expand_leaf!(mcts, st)
  end
  map_action, nst = normalized(st)
  edges = mcts.cache[nst].edges
  edges[argmax([e.q for e in edges])].action
end

function rand_game_lengths()
  results = tmap(_->simulate(start_state, [rand_policy, rand_policy]), 8, 1:10)
  println("$(sum(isnothing(r[1]) for r in results) / 10) were nothing")
  histogram([r[2] for r in results if !isnothing(r[1])])
end

function ab_game_lengths()
  results = tmap(_->simulate(start_state, [AlphaBeta(3), rand_policy]), 8, 1:10)
  println("$(sum(isnothing(r[1]) for r in results) / 10) were nothing")
  win_avg = mean([r[1] for r in results if !isnothing(r[1])])
  println("Average winner was $win_avg") 
  histogram([r[2] for r in results if !isnothing(r[1])])
end

include("tests.jl")

end # module MCTS
