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

mutable struct AvgNode
  last_access::Int
  counts::Int
  edges::Vector{Union{Edge, Nothing}}
  parent::Union{Nothing, BackEdge}
end

Base.@kwdef mutable struct ClassicMCTS{P} <: MC
  players::P
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, AvgNode} = Dict{State, AvgNode}()
  steps::Int = 100
  rollout_len::Int = 10
end

qvalue(::ClassicMCTS, e::Edge) = e.q / e.n

ucb(mc::MC, n::Int, e::Edge) = qvalue(mc, e) + sqrt(2) * sqrt(log(n) / e.n)
ucb(mc::MC, n::Int, ::Nothing) = -Inf32

function gc!(mc::MC)
  to_delete = Vector{State}()
  for (k,v) in mc.cache
    if v.last_access <= mc.last_move_time
      push!(to_delete, k)
    end
  end
  for k in to_delete
    delete!(mc.cache, k)
  end
end

function expand_leaf!(mcts::ClassicMCTS, nst::State)
  parent_key = nothing
  mcts.time += 1
  depth = 0
  while true
    depth += 1
    @assert depth < 500 
    if haskey(mcts.cache, nst)
      c = mcts.cache[nst]
      if c.last_access == mcts.time
        mcts.cache[parent_key.state].edges[parent_key.ix] = nothing
        return
      end
      c.last_access = mcts.time
      c.parent = parent_key
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
      value = sum(e.q for e in edges)
      mcts.cache[nst] = AvgNode(mcts.time, 0, edges, parent_key)
      backprop(mcts, nst, discount * value, length(edges))
      return
    end
  end
end

# The Q value of an edge is the discount factor, times the expected value of the state we go to
# The value of a node is the average of the q values of its edges
function backprop(mcts::ClassicMCTS, st::State, q::Float32, n::Int)
  depth = 0
  while true
    depth += 1
    @assert depth < 500
    node = mcts.cache[st]
    node.counts += n
    if isnothing(node.parent) || !haskey(mcts.cache, node.parent.state)
      return
    else
      p = node.parent
      q = discount * p.trans.value_map * q
      edge = mcts.cache[p.state].edges[p.ix]
      edge.n += n
      edge.q += q
      st = p.state
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
