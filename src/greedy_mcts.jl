Base.@kwdef mutable struct GreedyMCTS{P} <: MC
  players::P
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, Node} = Dict{State, Node}()
  steps::Int = 100
  rollout_len::Int = 10
end

qvalue(::GreedyMCTS, e::Edge) = e.q

function backprop(mcts::GreedyMCTS, st::State, child_qs::Vector{Float32}, n::Int)
  q = discount * maximum(child_qs)
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
        edge.q = max(edge.q, newq)
        to_process[p.state] = max(to_process[p.state], newq)
      end
    end
  end
end
