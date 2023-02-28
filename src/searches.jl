const eps = 1e-3
const no_min_action = ValuedAction(nothing, 1 + eps)
const no_max_action = ValuedAction(nothing, -1 - eps)

function min_action(st, alpha::ValuedAction, beta::ValuedAction, depth)
  if depth == 0
    return ValuedAction(Rand()(st).action, 0)
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
    return ValuedAction(Rand()(st).action, 0)
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

larger_q(a, b) = a.value > b.value ? a : b

function cached_max_action(st::State, depth::Int, cache::Dict)
  trans, nst = normalized(st)
  if is_terminal(nst)
    trans(ValuedAction(nothing, -1))
  elseif haskey(cache, nst)
    trans(cache[nst])
  elseif depth == 0
    chosen = Rand()(nst)
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

struct CachedMinimax
  depth::Int
end

function (mm::CachedMinimax)(st::State)
  cache = Dict{State, ValuedAction}()
  cached_max_action(st, mm.depth, cache)
end

