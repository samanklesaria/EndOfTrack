mutable struct Edge{N}
  action::Action
  q::Float32
  n::Int
  dest::Union{Nothing, N}
end

ValuedAction(e::Edge) = ValuedAction(e.action, e.q)

struct BackEdge{N}
  ix::Int8
  node::WeakRef{N}
end

function ucb(n::Int, e::Edge)
  bonus = 2 * (log(n) / e.n)
  e.q + sqrt(bonus)
end

mutable struct Node
  edges::Vector{Edge{Node}}
  parent::Union{Nothing, BackEdge{Node}}
end

Base.@kwdef struct NoRoll{E}
  root::Ref{Node} = Ref{Node}()
  steps::Int = 1_600
  tasks::Int = 32
  estimator::E
  temp::Float32
end

softmaximum(qs, t) = sum(qs .* softmax(qs .* t))

function backprop!(node::Node, temp)
  while !isnothing((back = node.parent))
    parent_node = back.node[]
    edge = parent_node.edges[back.ix]
    edge.n += 1
    edge.q = softmaximum(Float32[e.q for e in node.edges], temp)
    node = parent_node
  end
end

function init_state!(nr, parent, st, task)
  acts = actions(st)
  winning_ix = findfirst(will_win.(Ref(st), acts))
  if isnothing(winning_ix)
    edges = Edge[Edge(a, approx_q_val(nr.estimator, st, a, task), 1, nothing)
      for a in acts]
  else
    edges = [Edge(acts[winning_ix], 1f0, 1, nothing)]
  end
  node = Node(edges, parent)
  if !isnothing(parent)
    parent.node[].edges[parent.ix].dest = node
    backprop!(node, nr.temp)
  end
  node
end

function explore_next_state!(nr, node, st, task)
  while true
    ucbs = Float32[ucb(node.counts, e) for e in node.edges]
    ix = wsample(1:length(ucbs), softmax(ucbs .* nr.temp))
    next_st = @set apply_action(st, node.edges[ix].action).player = 2
    if is_terminal(next_st)
      node.edges[ix].n += 1
      node.edges[ix].n += 1
      backprop!(nr.temp, node)
      return nothing
    else if isnothing(node.edges[ix].dest)
      init_state!(nr, BackEdge(ix, WeakRef(node)), next_st, task)
      return nothing
    else
      node = node.edges[ix].dest
      st = next_st
    end
  end
end

function (nr::NoRoll)(st::State)
  chan = Channel{Nothing}(0)
  node = nr.root[]
  for task in 1:nr.tasks
    @async for _ in chan
      explore_next_state!(nr, node, st, task)
    end
  end
  for _ in 1:nr.steps
    put!(chan, nothing)
  end
  # println("Options:")
  # indent!()
  # for e in edges
  #   printindent("")
  #   log_action(st, ValuedAction(e))
  # end
  # dedent!()
  vals = Float32[e.q for e in node.edges]
  ix = wsample(1:length(vals), softmax(vals .* (nr.temp + 1)))
  nr.root[] = nr.root.edges[ix].dest
  ValuedAction(node.edges[ix].action, vals[ix])
end
