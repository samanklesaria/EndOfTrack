mutable struct EdgeP{N}
  action::Action
  q::Float32
  n::Int
  dest::Union{N, Nothing}
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

const ReqData = Tuple{Array{Float32, 4}, Channel{Vector{Float32}}}
const ReqChan = Channel{ReqData}

function ucb(n::Int, e::Edge)
  bonus = 8 * (log(n) / e.n)
  (e.q / e.n) + sqrt(bonus)
end

ucb_action(n::Node, e::EdgeP) = ValuedAction(e.action, ucb(n.counts, e))
q_action(e::EdgeP) = ValuedAction(e.action, e.q / e.n)
count_action(e::EdgeP) = ValuedAction(e.action, e.n)

mutable struct NoRollP{R,T}
  root::Node
  steps::Int
  task_chans::Vector{T}
  shared::Bool
  req::R
  temp::Float32
end

function NoRollP{R, T}(req::R; steps=1_600, shared=false,
    tasks=16, st=start_state, temp=1f0) where {R,T}
  val_chans = [T() for _ in 1:tasks]
  gpucom = gpu_com(req, val_chans[1])
  root = init_state!(nothing, st, gpucom)
  NoRollP{R,T}(root, steps, val_chans, shared, req, temp) 
end

const NoRoll = NoRollP{ReqChan, Channel{Vector{Float32}}}

const TestNoRoll = NoRollP{Nothing,Nothing}

function opponent_moved!(nr::NoRollP, action::Action)
  if !nr.shared
    ix = findfirst(e-> e.action == action, nr.root.edges)
    if isnothing(ix)
      error("Opponent's moved was impossible")
    end
    dest = nr.root.edges[ix].dest
    if !isnothing(dest)
      nr.root = dest 
    end
  end
end

function backprop!(node::Node, q::Float32)
  back = node.parent
  while !isnothing(back)
    parent_node = back.node
    edge = parent_node.edges[back.ix]
    edge.n += 1
    parent_node.counts += 1
    edge.q += q
    node = parent_node
    back = node.parent
    q = -q
  end
end

# Wraps communication channels for taking to GPU threads
struct GPUCom
  req_chan::ReqChan
  val_chan::Channel{Vector{Float32}}
end

function init_state!(parent::Union{Nothing, BackEdge},
    st::State, gpucom::Union{GPUCom, Nothing})
  acts = actions(st)
  next_sts = next_state.(Ref(st), acts)
  winning_ix = findfirst(is_terminal.(next_sts))
  if isnothing(winning_ix)
    vs = approx_vals(next_sts, gpucom)
    edges = Edge[Edge(a, v, 1, nothing) for (a,v) in zip(acts, vs)]
  else
    edges = [Edge(acts[winning_ix], 1f0, 1, nothing)]
  end
  node = Node(1, edges, parent)
  if !isnothing(parent)
    parent.node.edges[parent.ix].dest = node
    backprop!(node, -mean(e.q for e in edges))
  end
  node
end

function explore_next_state!(node::Node, st::State,
    gpucom::Union{GPUCom, Nothing})
  while true
    ucbs = Float32[ucb(node.counts, e) for e in node.edges]
    ix = argmax(ucbs)
    # if VALIDATE
    #   validate_action(st, node.edges[ix].action)
    # end
    next_st = @set apply_action(st, node.edges[ix].action).player = next_player(st.player)
    if is_terminal(next_st)
      node.edges[ix].n += 1
      node.edges[ix].q += 1
      backprop!(node, -1f0)
      # log_action(st, ucb_action(node, node.edges[ix]))
      return nothing
    elseif isnothing(node.edges[ix].dest)
      init_state!(BackEdge(ix, node), next_st, gpucom)
      # log_action(st, ucb_action(node, node.edges[ix]))
      return nothing
    else
      # log_action(st, ucb_action(node, node.edges[ix]))
      node = node.edges[ix].dest
      st = next_st
    end
  end
end

gpu_com(req::ReqChan, task_chan) = GPUCom(req, task_chan)
gpu_com(::Nothing, _) = nothing

function (nr::NoRollP)(st::State)
  chan = Channel{Nothing}(0)
  @sync begin
    for task_chan in nr.task_chans
      t = @async for _ in chan
        explore_next_state!(nr.root, $st, gpu_com(nr.req, $(task_chan)))
        # println("")
      end
      bind(chan, t)
    end
    for _ in 1:nr.steps
      put!(chan, nothing)
    end
    close(chan)
  end
  # println("Options:")
  # indent!()
  # for e in nr.root.edges
  #   printindent("")
  #   log_action(st, count_action(e))
  # end
  # dedent!()
  vals = Float32[e.n / nr.temp for e in nr.root.edges]
  ix = sample(Weights(softmax(vals)))
  chosen = ValuedAction(nr.root.edges[ix].action, vals[ix])
  root = nr.root.edges[ix].dest
  if !isnothing(root)
    root.parent = nothing
    nr.root = root
  end
  chosen 
end
