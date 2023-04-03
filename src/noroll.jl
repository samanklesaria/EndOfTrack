Base.@kwdef mutable struct NoRoll{E} <: MaxFamily
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, MaxNode} = Dict{State, MaxNode}()
  steps::Int = 2000
  estimator::E
end

function expand_leaf!(mcts::NoRoll, nst::State)
  # println()
  parent_key = nothing
  mcts.time += 1
  while true
    if haskey(mcts.cache, nst)
      c = mcts.cache[nst]
      if c.last_access == mcts.time
        mcts.cache[parent_key.state].edges[parent_key.ix] = nothing
        # println("Loop")
        return
      end
      c.last_access = mcts.time
      if !isnothing(parent_key)
        push!(c.parents, parent_key)
      end
      ucbs = [ucb(mcts, c.counts, e) for e in c.edges]
      ix = argmax(ucbs)
      # print("Traversing [$(ucbs[ix])] edge ")
      # log_action(nst, ValuedAction(c.edges[ix].action, 0))
      next_st = @set apply_action(nst, c.edges[ix].action).player = 2
      if is_terminal(next_st)
        # println("Terminal")
        backprop(mcts, nst, 1f0, 1)
        return
      end
      trans, new_nst = normalized(next_st)
      parent_key = BackEdge(ix, nst, trans)
      nst = new_nst
    else
      edges = Edge[Edge(a, approx_q_val(mcts.estimator, nst, a), 1)
        for a in actions(nst)]
      leaf_q = discount * maximum(e.q for e in edges)
      # println("Best guess: $(leaf_q)")
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = MaxNode(mcts.time, 0, edges, parents)
      backprop(mcts, nst, leaf_q, length(edges))
      return
    end
  end
end

