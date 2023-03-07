mutable struct Edge
  action::Action
  q::Float32
  n::Int
end

abstract type MC end

ValuedAction(mc::MC, e::Edge) = ValuedAction(e.action, qvalue(mc, e))

struct BackEdge
  ix::Int8
  state::State
  trans::Transformation
end

Base.hash(a::BackEdge, h::UInt) = hash(a.state, h)
Base.:(==)(a::BackEdge, b::BackEdge) = a.state == b.state

mutable struct Node
  last_access::Int
  counts::Int
  edges::Vector{Edge}
  parents::Set{BackEdge}
end

Base.@kwdef mutable struct ClassicMCTS{P} <: MC
  players::P
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, Node} = Dict{State, Node}()
  steps::Int = 100
  rollout_len::Int = 10
end

qvalue(::ClassicMCTS, e::Edge) = e.q / e.n

ucb(mc::MC, n::Int, e::Edge) = qvalue(mc, e) + sqrt(2) * sqrt(log(n) / e.n)

function gc!(mc::MC)
  to_delete = Vector{State}()
  for (k,v) in mc.cache
    if v.last_access < mc.last_move_time
      push!(to_delete, k)
    end
  end
  for k in to_delete
    delete!(mc.cache, k)
  end
end

function expand_leaf!(mcts::MC, nst::State)
  parent_key = nothing
  mcts.time += 1
  while true
    if haskey(mcts.cache, nst)
      c = mcts.cache[nst]
      if c.last_access == mcts.time
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
      edges = Edge[rollout(nst, a, mcts.players, mcts.rollout_len) for a in actions(nst)]
      # dedent!()
      child_qs = [e.q for e in edges]
      parents = isnothing(parent_key) ? Set{BackEdge}() : Set([parent_key])
      mcts.cache[nst] = Node(mcts.time, 1, edges, parents)
      backprop(mcts, nst, child_qs, length(edges))
      return
    end
  end
end

function backprop(mcts::ClassicMCTS, st::State, child_qs::Vector{Float32}, n::Int)
  q = discount * sum(child_qs)
  to_process = DefaultDict{State, Float32}(0f0)
  to_process[st] = q
  while length(to_process) > 0
    (st, q) = pop!(to_process)
    node = mcts.cache[st]
    for p in node.parents 
      if !haskey(mcts.cache, p.state)
        delete!(node.parents, p.state)
      else 
        newq = discount * p.trans.value_map * q
        edge = mcts.cache[p.state].edges[p.ix]
        edge.n += n
        edge.q += newq
        to_process[p.state] += newq
      end
    end
  end
end

function rollout(st::State, a::Action, players, steps)
  # printindent("Starting Rollout of ")
  # log_action(st, a)
  next_st = @set apply_action(st, a).player = 2
  endst = simulate(next_st, players; steps=steps)
  if isnothing(endst.winner)
    endq = 0f0
  elseif endst.winner == 1
    endq = 1f0
  else
    endq = -1f0
  end
  q = (discount ^ endst.steps) * endq
  # printindent("$(endst.steps) step rollout: ")
  # log_action(st, ValuedAction(a, q))
  Edge(a, q, 1)
end

function (mcts::MC)(st::State)
  mcts.last_move_time = mcts.time
  trans, nst = normalized(st)
  for _ in 1:mcts.steps
    expand_leaf!(mcts, nst)
  end
  gc!(mcts)
  edges = mcts.cache[nst].edges
  # println("Options:")
  # indent!()
  # for e in edges
  #   printindent("")
  #   log_action(st, ValuedAction(e))
  # end
  # dedent!()
  trans(ValuedAction(mcts, edges[argmax([qvalue(mcts, e) for e in edges])]))
end

classic_mcts(steps) = ClassicMCTS(players=greedy_players, steps=steps)
