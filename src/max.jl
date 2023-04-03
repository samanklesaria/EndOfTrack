abstract type MaxFamily <: MC end

mutable struct MaxNode
  last_access::Int
  counts::Int
  edges::Vector{Union{Edge, Nothing}}
  parents::Set{BackEdge}
end

Base.@kwdef mutable struct MaxMCTS{P,E} <: MaxFamily
  players::P
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, MaxNode} = Dict{State, MaxNode}()
  steps::Int = 100
  rollout_len::Int = 10
  estimator::E
end

qvalue(::MaxFamily, e::Edge) = e.q

function backprop(mcts::MaxFamily, st::State)
  to_process = Queue{State}()
  enqueue!(to_process, st)
  seen = Set{State}()
  while !isempty(to_process)
    st = dequeue!(to_process)
    node = mcts.cache[st]
    # Get all child states. 
    node.counts += n
    for p in node.parents 
      if !haskey(mcts.cache, p.state)
        delete!(node.parents, p.state)
      else 
        edge = mcts.cache[p.state].edges[p.ix]
        if !isnothing(edge)
          newq = discount * p.trans.value_map * q
          edge.n += n
          edge.q = max(edge.q, newq)
          if p.state ∉ seen
            push!(seen, p.state) 
            enqueue!(to_process, p.state=>newq)
          end
        end
      end
    end
  end
end

function expand_leaf!(mcts::MaxMCTS, nst::State)
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
      if is_terminal(next_st)
        return
      end
      trans, new_nst = normalized(next_st)
      parent_key = BackEdge(ix, nst, trans)
      nst = new_nst
    else
      # println("Expanding Leaf")
      # indent!()
      edges = Union{Edge, Nothing}[rollout(nst, a, mcts.players,
        mcts.rollout_len, mcts.estimator) for a in actions(nst)]
      # dedent!()
      leaf_q = discount * maximum(e.q for e in edges)
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = MaxNode(mcts.time, 0, edges, parents)
      backprop(mcts, nst, leaf_q, length(edges))
      return
    end
  end
end

max_mcts(;steps=20, rollout_len=20) = MaxMCTS(players=greedy_players, steps=steps,
  rollout_len=rollout_len, estimator=nothing)
