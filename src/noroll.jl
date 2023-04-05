mutable struct Edge
  action::Action
  q::Float32
  n::Int
end

ValuedAction(e::Edge) = ValuedAction(e.action, e.q)

struct BackEdge
  ix::Int8
  state::State
  trans::Transformation
end
Base.hash(a::BackEdge, h::UInt) = hash(a.state, h)
Base.:(==)(a::BackEdge, b::BackEdge) = a.state == b.state

function ucb(n::Int, e::Edge)
  @infiltrate n < 0
  bonus = 2 * (log(n) / e.n)
  e.q + sqrt(max(0, bonus))
end
ucb(n::Int, ::Nothing) = -Inf32

mutable struct MaxNode
  last_access::Int
  counts::Int
  edges::Vector{Union{Edge, Nothing}}
  parents::Set{BackEdge}
  q::Float32
end

Base.@kwdef mutable struct NoRoll{E}
  time::Int = 0
  last_move_time::Int = 0
  cache::Dict{State, MaxNode} = Dict{State, MaxNode}()
  steps::Int = 2_000
  estimator::E
end

function gc!(mc)
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

function getit(a)
  @assert a > 0
  @assert a < 1e10
  a
end

softmaximum(qs) = sum(qs .* LogExpFunctions.softmax(qs .* 30))

function backprop(mcts, st::State)
  to_process = Queue{State}()
  enqueue!(to_process, st)
  seen = Set{State}()
  while !isempty(to_process)
    st = dequeue!(to_process)
    node = mcts.cache[st]
    node.counts += 1
    node.q = softmaximum(Float32[e.q for e in node.edges if !isnothing(e)])
    for p in node.parents 
      if !haskey(mcts.cache, p.state)
        delete!(node.parents, p.state)
      else 
        edge = mcts.cache[p.state].edges[p.ix]
        if !isnothing(edge)
          newq = discount * p.trans.value_map * node.q
          edge.n = node.counts
          edge.q = newq
          if p.state ∉ seen
            push!(seen, p.state) 
            enqueue!(to_process, p.state)
          end
        end
      end
    end
  end
end

function init_state!(mcts, new_nst, parents)
  edges = Edge[Edge(a, approx_q_val(mcts.estimator, new_nst, a), 1)
    for a in actions(new_nst)]
  mcts.cache[new_nst] = MaxNode(mcts.time, 1, edges, parents, 0f0)
  backprop(mcts, new_nst)
end

function explore_next_state!(mcts, nst)
    c = mcts.cache[nst]
    c.last_access = mcts.time
    ucbs = [ucb(c.counts, e) for e in c.edges]
    ixs = sortperm(ucbs, rev=true)
    for ix in ixs
        next_st = @set apply_action(nst, c.edges[ix].action).player = 2
        if is_terminal(next_st)
          c.edges[ix].q = 1f0
          c.edges[ix].n += 1
          backprop(mcts, nst)
          # println("Terminal\n")
          return nothing # No further states to explore
        end
        trans, new_nst = normalized(next_st)
        if haskey(mcts.cache, new_nst)
          if mcts.cache[new_nst].last_access != mcts.time
            # print("Traversing cached edge ")
            # log_action(nst, ValuedAction(c.edges[ix].action, 0))
            return (new_nst, BackEdge(ix, nst, trans))
          end
        else
          parents = Set([BackEdge(ix, nst, trans)])
          init_state!(mcts, new_nst, parents)
          # println()
          return nothing
        end
    end
    error("All paths from node lead to loops")
end

function expand_leaf!(mcts::NoRoll, nst::State)
  mcts.time += 1
  if !haskey(mcts.cache, nst)
    init_state!(mcts, nst, Set{BackEdge}())
  else
    result = (nst, nothing)
    while !isnothing(result)
      if !isnothing(result[2])
        push!(mcts.cache[result[1]].parents, result[2])
      end
      result = explore_next_state!(mcts, result[1])
    end
  end
end
           
function (mcts::NoRoll)(st::State)
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
  trans(ValuedAction(edges[argmax([
    isnothing(e) ? -Inf32 : e.q for e in edges])]))
end
