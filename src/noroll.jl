# Value functions are from the perspective of the current player

# If we have each node take the AVERAGE of its children rather than the max, 
# as in the other papers, what policy are we actually learning about?
# We also don't want to sample children entirely randomly: we want to pick the max.

# Perhaps: when thinking and playing, use a high-ish temperature
# When computing q values... oh wait. 
# The classic alg will ADD the new q value to all the parents, and then increase the number
# So we're not computing the value function for a random policy. We're computing it for the MCTs policy

mutable struct EdgeP{N}
  action::Action
  q::Float32
  n::Int
  dest::Union{Nothing, N}
end

struct BackEdgeP{N}
  ix::Int8
  node::N
end

mutable struct Node
  counts::Int
  edges::Vector{EdgeP{Node}}
  parent::Union{Nothing, BackEdgeP{Node}}
end

const Edge = EdgeP{Node}
const BackEdge = BackEdgeP{Node}

const ReqChan = Channel{Tuple{Array{Float32, 4}, Channel{Vector{Float32}}}}

function ucb(n::Int, e::Edge)
  bonus = 4 * (log(n) / e.n)
  e.q + sqrt(bonus)
end

ValuedAction(n::Node, e::EdgeP) = ValuedAction(e.action, ucb(n.counts, e))

mutable struct NoRollP{R,T}
  root::Node
  steps::Int
  task_chans::Vector{T}
  temp::Float32
  shared::Bool
  req::R
end

function NoRollP{R, T}(req::R; steps=1_600, temp=1f-5, shared=false,
    tasks=1, st=start_state) where {R,T}
  val_chans = [T() for _ in 1:tasks]
  gpucom = gpu_com(req, val_chans[1])
  root = init_state!(nothing, st, gpucom, temp)
  NoRollP{R,T}(root, steps, val_chans, temp, shared, req) 
end

const NoRoll = NoRollP{ReqChan, Channel{Vector{Float32}}}

const TestNoRoll = NoRollP{Nothing,Nothing}

function opponent_moved!(nr::NoRollP, action)
  if !nr.shared
    ix = findfirst(e-> e.action == action, nr.root.edges)
    dest = nr.root.edges[ix].dest
    if !isnothing(dest)
      nr.root = dest 
    end
  end
end

softmaximum(qs, t) = sum(qs .* LogExpFunctions.softmax(qs .* t))

function backprop!(node::Node, temp::Float32)
  back = node.parent
  while !isnothing(back)
    parent_node = back.node
    edge = parent_node.edges[back.ix]
    edge.n += 1
    parent_node.counts += 1
    edge.q = softmaximum(Float32[-e.q for e in node.edges], temp)
    node = parent_node
    back = node.parent
  end
end

# Wraps communication channels for taking to GPU threads
struct GPUCom
  req_chan::ReqChan
  val_chan::Channel{Vector{Float32}}
end

function init_state!(parent::Union{Nothing, BackEdge},
    st::State, gpucom::Union{GPUCom, Nothing}, temp::Float32)
  acts = actions(st)
  next_sts = next_state.(Ref(st), acts)
  winning_ix = findfirst(is_terminal.(next_sts))
  if isnothing(winning_ix)
    vs = approx_vals(next_sts, gpucom)
    edges = Edge[Edge(a, v, 1, nothing) for (a,v) in zip(acts, vs)]
  else
    edges = [Edge(acts[winning_ix], 1f0, 1, nothing)]
  end
  node = Node(length(edges), edges, parent)
  if !isnothing(parent)
    parent.node.edges[parent.ix].dest = node
    backprop!(node, temp)
  end
  node
end

function explore_next_state!(nr::NoRollP, node::Node, st::State,
    gpucom::Union{GPUCom, Nothing})
  while true
    ucbs = Float32[ucb(node.counts, e) for e in node.edges]
    ix = wsample(1:length(ucbs), LogExpFunctions.softmax(ucbs .* nr.temp))
    # log_action(st, ValuedAction(node, node.edges[ix]))
    next_st = @set apply_action(st, node.edges[ix].action).player = next_player(st.player)
    if is_terminal(next_st)
      node.edges[ix].n += 1
      backprop!(node, nr.temp)
      return nothing
    elseif isnothing(node.edges[ix].dest)
      init_state!(BackEdge(ix, node), next_st, gpucom, nr.temp)
      return nothing
    else
      node = node.edges[ix].dest
      st = next_st
    end
  end
end

gpu_com(req::ReqChan, task_chan) = GPUCom(req, task_chan)
gpu_com(::Nothing, _) = nothing

# TODO: Can we replace the channel with a semaphore while 
# pool = Semaphore(nr.steps)
# maintaining the error handling property? Would need to implement that. 

function (nr::NoRollP)(st::State)
  chan = Channel{Nothing}(0)
  @sync begin
    for task_chan in nr.task_chans
      t = @async for _ in chan
        explore_next_state!(nr, nr.root, st, gpu_com(nr.req, task_chan))
        # println("")
      end
      bind(chan, t)
    end
    for _ in 1:nr.steps
      put!(chan, nothing)
    end
    close(chan)
  end
  println("Options:")
  indent!()
  for e in nr.root.edges
    printindent("")
    log_action(st, ValuedAction(e.action, e.q))
  end
  dedent!()
  vals = Float32[e.q for e in nr.root.edges]
  ix = wsample(1:length(vals), LogExpFunctions.softmax(vals .* (nr.temp + 1)))
  chosen = ValuedAction(nr.root.edges[ix].action, vals[ix])
  root = nr.root.edges[ix].dest
  if !isnothing(root)
    root.parent = nothing
    nr.root = root
  end
  chosen
end
