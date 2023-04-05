include("hueristics.jl")

Base.:*(a::Number, b::ValuedAction) = ValuedAction(b.action, a * b.value)

const eps = 1e-3
const fake_action = (0, @SVector zeros(2))
const no_min_action = ValuedAction(fake_action, 1 + eps)
const no_max_action = ValuedAction(fake_action, -1 - eps)

approx_val(::Nothing, _) = 0f0

function approx_q_val(heuristic, st::State, a::Action)
  new_st = apply_action(st, a)
  if is_terminal(new_st) return 1.0 end
  new_st = @set new_st.player = next_player(new_st.player)
  trans, nst = normalized(new_st)
  trans.value_map * approx_val(heuristic, nst)
end

function min_action(st, alpha::ValuedAction, beta::ValuedAction, depth, hueristic)
  if depth == 0
    return ValuedAction(fake_action, approx_val(hueristic, st))
  end
  for a in shuffled_actions(st) 
    next_st = @set apply_action(st, a).player = 1
    if is_terminal(next_st)
      return ValuedAction(a.action, -1)
    else
      # printindent("Min considering  ")
      # log_action(st, a)
      # indent!()
      recurse = max_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, hueristic)
      # dedent!()
      lb = discount * recurse
      if lb.value < beta.value
        beta = ValuedAction(a.action, lb.value)
        if alpha.value > beta.value
          # printindent("Parent would discard ")
          # log_action(next_st, recurse)
          return alpha
        end
        if alpha.value == beta.value
          # printindent("Best possible is ")
          # log_action(next_st, recurse)
          return beta
        end
        # printindent("New best ")
        # log_action(next_st, recurse)
      end
    end
  end
  beta
end

function max_action(st, alpha, beta, depth, hueristic)
  if depth == 0
    return ValuedAction(fake_action, approx_val(hueristic, st))
  end
  for a in shuffled_actions(st)
    next_st = @set apply_action(st, a).player = 2
    if is_terminal(next_st)
      return ValuedAction(a.action, 1)
    else
      # printindent("Max considering  ")
      # log_action(st, a)
      # indent!()
      recurse = min_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, hueristic)
      # dedent!() 
      ub = discount * recurse
      the_action = ValuedAction(a.action, ub.value)
      if ub.value > alpha.value
        alpha = the_action
        if alpha.value > beta.value
          # printindent("Parent would discard ")
          # log_action(next_st, recurse)
          return beta
        end
        if alpha.value == beta.value
          # printindent("Best possible is ")
          # log_action(next_st, recurse)
          return alpha
        end
        # printindent("New best ")
        # log_action(next_st, recurse)
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
  if haskey(cache, nst)
    result = trans(cache[nst])
    # printindent("Cached value ")
    # log_action(st, result)
    result
    result
  elseif depth == 0
    result = trans(ValuedAction(fake_action, approx_val(heuristic, nst)))
    # printindent("Maxdepth ")
    # log_action(st, result)
    result
  else
    best_child = mapreduce(larger_q, smart_actions(nst)) do a
      next_st = @set apply_action(nst, a).player = 2
      if is_terminal(next_st)
        ValuedAction(a.action, 1)
      else
        # printindent("Considering ")
        # log_action(nst, a)
        # indent!()
        child_val = cached_max_action(next_st, depth - 1, cache, heuristic).value
        # dedent!()
        ValuedAction(a.action, discount * child_val)
      end
    end
    # printindent("Best child ")
    # log_action(nst, best_child)
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

function smart_actions(st)
  acts = shuffled_actions(st)
  if abs(acts[1].value) == 1f0
    acts[1:1]
  else
    acts 
  end
end

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
