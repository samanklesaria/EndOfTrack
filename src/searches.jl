include("hueristics.jl")

Base.:*(a::Number, b::ValuedAction) = ValuedAction(b.action, a * b.value)

const eps = 1e-3
const fake_action = (0, @SVector zeros(2))
const no_min_action = ValuedAction(fake_action, 1 + eps)
const no_max_action = ValuedAction(fake_action, -1 - eps)

function min_action(st, alpha::ValuedAction, beta::ValuedAction, depth, hueristic)
  if depth == 0
    if isnothing(hueristic)
      return ValuedAction(fake_action, 0f0)
    else
      trans, nst = normalized(st)
      return trans(ValuedAction(fake_action, approx_val(hueristic, nst)))
    end
  end
  for a in shuffled_actions(st) 
    next_st = @set apply_action(st, a).player = next_player(st.player)
    if is_terminal(next_st)
      return ValuedAction(a.action, -1)
    else
      lb = discount * max_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, hueristic)
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

function max_action(st, alpha, beta, depth, hueristic)
  if depth == 0
    if isnothing(hueristic)
      return ValuedAction(fake_action, 0f0)
    else
      trans, nst = normalized(st)
      return trans(ValuedAction(fake_action, approx_val(hueristic, nst)))
    end
  end
  for a in shuffled_actions(st)
    next_st = @set apply_action(st, a).player = 2
    if is_terminal(next_st)
      return ValuedAction(a.action, 1)
    else
      ub = discount * min_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, hueristic)
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

struct AlphaBeta{H}
  depth::Int
  hueristic::H
end

AlphaBeta(depth) = AlphaBeta(depth, nothing)

function (ab::AlphaBeta)(st)
  if st.player == 1
    max_action(st, no_max_action, no_min_action, ab.depth, ab.hueristic)
  else
    min_action(st, no_max_action, no_min_action, ab.depth, ab.hueristic)
  end
end

larger_q(a, b) = a.value > b.value ? a : b

function cached_max_action(st::State, depth::Int, cache::Dict, heuristic)
  trans, nst = normalized(st)
  if is_terminal(nst)
    trans(ValuedAction(fake_action, -1))
  elseif haskey(cache, nst)
    trans(cache[nst])
  elseif depth == 0
    trans(ValuedAction(fake_action, approx_val(heuristic, nst)))
  else
    best_child = mapreduce(larger_q, smart_actions(nst)) do a
      next_st = @set apply_action(nst, a).player = 2
      child_val = cached_max_action(next_st, depth - 1, cache, heuristic).value
      ValuedAction(a.action, discount * child_val)
    end
    cache[nst] = best_child
    trans(best_child)
  end
end

struct CachedMinimax{H}
  depth::Int
  heuristic::H
end

CachedMinimax(depth) = CachedMinimax(depth, nothing)

function (mm::CachedMinimax)(st::State)
  cache = Dict{State, ValuedAction}()
  cached_max_action(st, mm.depth, cache, mm.heuristic)
end

function apply_hueristic(st::State, a::Action)::ValuedAction
  new_st = apply_action(st, a)
  term = is_terminal(new_st)
  if !term
    return ValuedAction(a, fast_hueristic(new_st))
  else
    return ValuedAction(a, st.player == 1 ? 1 : -1)
  end
end

function shuffled_actions(st)
  acts = actions(st)
  Random.shuffle!(acts)
  vacts = ValuedAction[apply_hueristic(st, a) for a in acts]
  sort(vacts; by=a-> abs.(a.value), rev= true, alg=MergeSort) 
end

# This gives some of the benefit of AlphaBeta to minimax
function smart_actions(st)
  acts = shuffled_actions(st)
  if abs(acts[1].value) == 1f0
    acts[1:1]
  else
    acts 
  end
end

# Some variation of this would be useful
# if we wanted to use an expensive hueristic. 
# function get_actions(st, reward)
#   raw_choices = actions(st)
#   k = min(length(raw_choices), 5)
#   choices = Vector{ValuedAction}(undef, k)
#   sofar = 0
#   for a in raw_choices
#     new_st = apply_action(st, a)
#     if is_terminal(new_st)
#       return [ValuedAction(a, reward)]
#     end
#     if sofar < k
#       sofar += 1
#       choices[sofar] = ValuedAction(a, 0)
#     else
#       choices[rand(1:k)] = ValuedAction(a, 0)
#     end
#   end
#   @assert sofar == k
#   choices
# end

struct Rand end

const rand_players = (Rand(), Rand())

function (::Rand)(st::State)
  choices = actions(st)
  ValuedAction(choices[rand(1:length(choices))], 0f0)
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

# struct Boltzmann
#   temp::Float32
# end

# function (b::Boltzmann)(st::State)
#   choices = actions(st)
#   player = st.player == 1 ? 1 : -1
#   probs = [player .* c.value ./ b.temp for c in choices]
#   softmax!(probs)
#   sample(choices, Weights(probs))
# end
