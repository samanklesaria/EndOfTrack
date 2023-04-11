mutable struct EdgeP{N}
  action::Action
  q::Float32
  n::Int
  dest::Union{Nothing, N}
end

struct BackEdge
  ix::Int8
  node::WeakRef
end

mutable struct Node
  counts::Int
  edges::Vector{EdgeP{Node}}
  parent::Union{Nothing, BackEdge}
end

const Edge = EdgeP{Node}

const ReqChan = Channel{Tuple{Array{Float32, 4}, Channel{Vector{Float32}}}}

function ucb(n::Int, e::Edge)
  bonus = 2 * (log(n) / e.n)
  e.q + sqrt(bonus)
end

const TASKS = 20

Base.@kwdef struct NoRoll
  root::Ref{Node} = Ref{Node}()
  steps::Int = 1_600
  task_chans::Vector{Channel{Vector{Float32}}} =
    [Channel{Vector{Float32}}() for _ in 1:1]
  temp::Float32 = 1f-5
  shared::Bool=false
  req::ReqChan
end

function opponent_moved!(nr::NoRoll, action)
  if !nr.shared
    ix = findfirst(e-> e.action == action, nr.root.edges)
    nr.root[] = nr.root.edges[ix].dest
  end
end

softmaximum(qs, t) = sum(qs .* LogExpFunctions.softmax(qs .* t))

function backprop!(node::Node, temp::Float32)
  while !isnothing((back = node.parent))
    parent_node = back.node[]
    edge = parent_node.edges[back.ix]
    edge.n += 1
    parent_node.counts += 1
    edge.q = softmaximum(Float32[e.q for e in node.edges], temp)
    node = parent_node
  end
end


# Wraps communication channels for taking to GPU threads
struct GPUCom
  req_chan::ReqChan
  val_chan::Channel{Vector{Float32}}
end

function init_state!(nr::NoRoll, parent::Union{Nothing, BackEdge},
    st::State, chans::GPUCom)
  acts = actions(st)
  next_sts = next_state.(Ref(st), acts)
  winning_ix = findfirst(is_terminal.(next_sts))
  if isnothing(winning_ix)
    println("Initializing state")
    vs = approx_vals(next_sts, chans)
    edges = Edge[Edge(a, v, 1, nothing) for (a,v) in zip(acts, vs)]
  else
    edges = [Edge(acts[winning_ix], 1f0, 1, nothing)]
  end
  node = Node(1, edges, parent)
  if !isnothing(parent)
    parent.node[].edges[parent.ix].dest = node
    backprop!(node, nr.temp)
  end
  node
end

function explore_next_state!(nr::NoRoll, node::Node, st::State, task_chan::GPUCom)
  while true
    ucbs = Float32[ucb(node.counts, e) for e in node.edges]
    ix = wsample(1:length(ucbs), LogExpFunctions.softmax(ucbs .* nr.temp))
    next_st = @set apply_action(st, node.edges[ix].action).player = 2
    if is_terminal(next_st)
      node.edges[ix].n += 1
      node.edges[ix].n += 1
      backprop!(nr.temp, node)
      return nothing
    elseif isnothing(node.edges[ix].dest)
      init_state!(nr, BackEdge(ix, WeakRef(node)), next_st, task_chan)
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
  for task_chan in nr.task_chans
    t = @async for _ in chan
      explore_next_state!(nr, node, st, GPUCom(nr.req, task_chan))
    end
    bind(chan, t)
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
  ix = wsample(1:length(vals), LogExpFunctions.softmax(vals .* (nr.temp + 0.1)))
  nr.root[] = nr.root.edges[ix].dest
  ValuedAction(node.edges[ix].action, vals[ix])
end
