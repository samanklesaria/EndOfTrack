# Value functions are from the perspective of the current player

# TODO: better error handling.
# Opponent-moved expects a correct move.

# Perhaps we should be okay with errors happening.
# Instead of erring, we should just terminate the current run and start a new one. 
# Log it to stderr.



mutable struct EdgeP{N}
  action::Action
  dist::Union{ValPrior, Dirac{Float32}}
  dest::Union{N, Nothing}
end

struct BackEdgeP{N}
  ix::Int8
  node::N
end

mutable struct Node
  edges::Vector{EdgeP{Node}}
  parent::Union{Nothing, BackEdgeP{Node}}
end

const Edge = EdgeP{Node}

const BackEdge = BackEdgeP{Node}

const ReqChan = Channel{Tuple{Array{Float32, 4}, Channel{Vector{Float32}}}}

ValuedAction(e::Edge) = ValuedAction(e.action, mean(e.dist))

mutable struct NoRollP{R,T}
  root::Node
  steps::Int
  task_chans::Vector{T}
  shared::Bool
  req::R
end

function NoRollP{R, T}(req::R; steps=1_600, shared=false,
    tasks=16, st=start_state) where {R,T}
  val_chans = [T() for _ in 1:tasks]
  gpucom = gpu_com(req, val_chans[1])
  root = init_state!(nothing, st, gpucom)
  NoRollP{R,T}(root, steps, val_chans, shared, req) 
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


function backprop!(node::Node, n::Int, q)
  q1 = sum(q)
  q2 = sum(qi^2 for qi in q)
  back = node.parent
  while !isnothing(back)
    parent_node = back.node
    edge = parent_node.edges[back.ix]
    edge.dist = posterior(edge.dist, n, q1, q2)
    node = parent_node
    back = node.parent
    q1 = -q1
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
    edges = Edge[Edge(a, ValPrior(v), nothing) for (a,v) in zip(acts, vs)]
  else
    edges = Edge[Edge(acts[winning_ix], Dirac(1f0), nothing)]
  end
  node = Node(edges, parent)
  if !isnothing(parent)
    parent.node.edges[parent.ix].dest = node
    backprop!(node, length(edges), Float32[-mean(e.dist) for e in edges])
  end
  node
end

function explore_next_state!(node::Node, st::State,
    gpucom::Union{GPUCom, Nothing})
  while true
    samples = Float32[rand(e.dist) for e in node.edges]
    ix = argmax(samples)
    # log_action(st, ValuedAction(node.edges[ix]))
    # if VALIDATE
    #   validate_action(st, node.edges[ix].action)
    # end
    next_st = @set apply_action(st, node.edges[ix].action).player = next_player(st.player)
    if is_terminal(next_st)
      backprop!(node, 1, (-1f0,))
      return nothing
    elseif isnothing(node.edges[ix].dest)
      init_state!(BackEdge(ix, node), next_st, gpucom)
      return nothing
    else
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
  #   log_action(st, q_action(e))
  # end
  # dedent!()
  vals = Float32[mean(e.dist) for e in nr.root.edges]
  ix = argmax(vals)
  chosen = ValuedAction(nr.root.edges[ix].action, vals[ix])
  root = nr.root.edges[ix].dest
  if !isnothing(root)
    root.parent = nothing
    nr.root = root
  end
  chosen
end
