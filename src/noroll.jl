Base.@kwdef mutable struct NoRoll{E} <: MaxFamily
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, MaxNode} = Dict{State, MaxNode}()
  steps::Int = 100
  estimator::E
end

function expand_leaf!(mcts::NoRoll, nst::State)
  parent_key = nothing
  mcts.time += 1
  while true
    if haskey(mcts.cache, nst)
      c = mcts.cache[nst]
      if c.last_access == mcts.time
        mcts.cache[parent_key.state].edges[parent_key.ix] = nothing
        return
      end
      c.last_access = mcts.time
      if !isnothing(parent_key)
        push!(c.parents, parent_key)
      end
      ix = argmax([ucb(mcts, c.counts, e) for e in c.edges])
      # print("Traversing Edge ")
      # log_action(nst, ValuedAction(c.edges[ix].action, 0))
      next_st = @set apply_action(nst, c.edges[ix].action).player = 2
      if is_terminal(next_st) return end
      trans, new_nst = normalized(next_st)
      parent_key = BackEdge(ix, nst, trans)
      nst = new_nst
    else
      # println("Expanding Leaf")
      # indent!()
      edges = Edge[Edge(a, approx_q_val(mcts.estimator, nst, a), 1)
        for a in actions(nst)]
      # dedent!()
      leaf_q = discount * maximum(e.q for e in edges)
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = MaxNode(mcts.time, 0, edges, parents)
      backprop(mcts, nst, leaf_q, length(edges))
      return
    end
  end
end

