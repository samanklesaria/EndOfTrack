mutable struct MaxNode
  last_access::Int
  counts::Int
  edges::Vector{Union{Edge, Nothing}}
  parents::Set{BackEdge}
end

Base.@kwdef mutable struct MaxMCTS{P} <: MC
  players::P
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, MaxNode} = Dict{State, MaxNode}()
  steps::Int = 100
  rollout_len::Int = 10
end

qvalue(::MaxMCTS, e::Edge) = e.q

function backprop(mcts::MaxMCTS, st::State, q::Float32, n::Int)
  to_process = Queue{Pair{State, Float32}}()
  push!(to_process, st=>q)
  seen = Set{State}()
  while length(to_process) > 0
    (st, q) = pop!(to_process)
    node = mcts.cache[st]
    node.n += n
    for p in node.parents 
      if !haskey(mcts.cache, p.state)
        delete!(node.parents, p.state)
      else 
        newq = discount * p.trans.value_map * q
        edge = mcts.cache[p.state].edges[p.ix]
        edge.n += n
        edge.q = max(edge.q, newq)
        if p.state ∉ seen
          push!(seen, p.state) 
          push!(to_process, p.state=>newq)
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
      edges = Union{Edge, Nothing}[rollout(nst, a, mcts.players, mcts.rollout_len) for a in actions(nst)]
      # dedent!()
      leaf_q = discount * maximum(e.q for e in edges)
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = MaxNode(mcts.time, 0, edges, parents)
      backprop(mcts, nst, leaf_q, length(edges))
      return
    end
  end
end

max_mcts(steps) = MaxMCTS(players=greedy_players, steps=steps)
