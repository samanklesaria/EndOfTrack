include("hueristics.jl")

Base.:*(a::Number, b::ValuedAction) = ValuedAction(b.action, a * b.value)

const eps = 1e-3
const fake_action = (0, @SVector zeros(2))
const no_min_action = ValuedAction(fake_action, 1 + eps)
const no_max_action = ValuedAction(fake_action, -1 - eps)

init!(::Any; state=start_state) = nothing

function min_action(st, alpha::ValuedAction, beta::ValuedAction, depth, ab)
  if depth == 0
    return ValuedAction(fake_action, 0f0)
  end
  for a in shuffled_actions(st) 
    next_st = @set apply_action(st, a).player = 1
    if is_terminal(next_st)
      return ValuedAction(a.action, -1)
    else
      # printindent("Min considering  ")
      # log_action(st, a)
      # indent!()
      recurse = max_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, ab)
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

function max_action(st, alpha, beta, depth, ab)
  if depth == 0
    return ValuedAction(fake_action, 0f0)
  end
  for a in shuffled_actions(st)
    next_st = @set apply_action(st, a).player = 2
    if is_terminal(next_st)
      return ValuedAction(a.action, 1)
    else
      # printindent("Max considering  ")
      # log_action(st, a)
      # indent!()
      recurse = min_action(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, ab)
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

# function min_action_nn(st, alpha, beta, depth, ab)
#   acts = shuffled_actions(st)
#   next_sts = next_state.(Ref(st), acts)
#   winning_ix = findfirst(is_terminal.(next_sts))
#   if !isnothing(winning_ix)
#     return ValuedAction(acts[winning_ix], -1)
#   else 
#     if depth == 0
#       val = minimum(approx_vals(next_sts, ab.chan))
#       return ValuedAction(fake_action, val)
#     else
#       for (a, next_st) in zip(acts, next_sts)
#         recurse = min_action_nn(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, ab)
#         ub = discount * recurse
#         the_action = ValuedAction(a.action, ub.value)
#         if ub.value > alpha.value
#           alpha = the_action
#           if alpha.value > beta.value
#             return beta
#           end
#           if alpha.value == beta.value
#             return alpha
#           end
#         end
#       end
#     end
#   end
#   alpha
# end

# function max_action_nn(st, alpha, beta, depth, ab)
#   acts = shuffled_actions(st)
#   next_sts = next_state.(Ref(st), acts)
#   winning_ix = findfirst(is_terminal.(next_sts))
#   if !isnothing(winning_ix)
#     return ValuedAction(acts[winning_ix], 1)
#   else 
#     if depth == 0
#       val = minimum(approx_vals(next_sts, ab.chan))
#       return ValuedAction(fake_action, val)
#     else
#       for (a, next_st) in zip(acts, next_sts)
#         recurse = min_action_nn(next_st, inv_discount * alpha, inv_discount * beta, depth - 1, ab)
#         ub = discount * recurse
#         the_action = ValuedAction(a.action, ub.value)
#         if ub.value > alpha.value
#           alpha = the_action
#           if alpha.value > beta.value
#             return beta
#           end
#           if alpha.value == beta.value
#             return alpha
#           end
#         end
#       end
#     end
#   end
#   alpha
end

struct AlphaBeta{T}
  depth::Int
  chan::T
end

function (ab::AlphaBeta)(st)
  if st.player == 1
    max_action(st, no_max_action, no_min_action, ab.depth, ab)
  else
    min_action(st, no_max_action, no_min_action, ab.depth, ab)
  end
end

larger_q(a, b) = a.value > b.value ? a : b

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

