# Value functions are from the perspective of the current player

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
end

function NoRollP{R, T}(req::R; steps=1_600, shared=false,
    tasks=1, st=start_state) where {R,T}
  val_chans = [T() for _ in 1:tasks]
  gpucom = gpu_com(req, val_chans[1])
  root = init_state!(nothing, st, gpucom)
  NoRollP{R,T}(root, steps, val_chans, shared, req) 
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
    # log_action(st, ucb_action(node, node.edges[ix]))
    next_st = @set apply_action(st, node.edges[ix].action).player = next_player(st.player)
    if is_terminal(next_st)
      node.edges[ix].n += 1
      node.edges[ix].q += 1
      backprop!(node, -1f0)
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

# TODO: Can we replace the channel with a semaphore while 
# pool = Semaphore(nr.steps)
# maintaining the error handling property? Would need to implement that. 

# If we just take argmax, our games will be deterministic. 
# Every game will be exactly the same, so we won't learn anything
# interesting. We could get around this by using exploring starts. 

# In addition to the learning buffer, there's a position buffer.
# Start by populating the position buffer by some AlphaBeta runs.
# Draw an initial position from the position buffer. 
# When we finish a game, put its equivalence class in the learning buffer. 
# Also, sample a random position from the early part of the game and
# put it in the position buffer

# Alternately, we could do Thompson sampling. Ah, but how do
# we adapt that to the tree formulation? 
# Bai ('13) suggests that the policy you take from a node
# converges during MCTS, so doing Bayesaian updates as if
# in the iid case still works in the limit 

function (nr::NoRollP)(st::State)
  chan = Channel{Nothing}(0)
  @sync begin
    for task_chan in nr.task_chans
      t = @async for _ in chan
        explore_next_state!(nr.root, st, gpu_com(nr.req, task_chan))
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
  vals = Float32[(e.q / e.n) for e in nr.root.edges]
  ix = argmax(vals)
  chosen = ValuedAction(nr.root.edges[ix].action, vals[ix])
  root = nr.root.edges[ix].dest
  if !isnothing(root)
    root.parent = nothing
    nr.root = root
  end
  chosen
end
