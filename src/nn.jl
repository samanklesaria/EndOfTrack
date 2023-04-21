using TensorBoardLogger, Logging

const TRAIN_BATCH_SIZE = 64
const EVAL_BATCH_SIZE = 128

# TODO: if we encoded the positions, our replay buffer would
# take half the space.

# With validation turned on, playoffs between AlphaBeta
# and NoRoll get invalid states. Why?

# How is it possible to have the parent node use a Dirac
# distribution? Why do we occasionally get this error? 

# 1. Don't look at all the data. Try to overfit. Maybe make it bigger?
# 2. Don't look at any state that is equal to the start state.
# 3. Don't augment. 
# 

function sorted_gpus()
  "Gets least busy gpus"
  usage = parse.(Int, split(readchomp(`./getter.sh`), "\n"))
  sortperm(usage)
end

function get_batch(h5::HDF5.File, ix)
  values = h5["values"][ix]
  players = h5["players"][ix]
  pieces1 = h5["pieces1"][ix,:,:]
  pieces2 = h5["pieces2"][ix,:,:]
  balls1 = h5["balls1"][ix,:]
  balls2 = h5["balls2"][ix,:]
  states = [State(players[i], [PlayerState(balls1[i,:], pieces1[i,:,:]),
    PlayerState(balls2[i,:], pieces2[i,:,:])]) for i in 1:length(ix)] 
  mask = states .!= Ref(start_state)
  (states[mask], values[mask])
  # mstates = states[mask]
  # mvals = values[mask]
  # states2 = map_state.(Ref(flip_pos_vert), mstates)
  # states3 = flip_players.(mstates)
  # ([mstates; states2; states3], [mvals; mvals; -mvals])
end

function as_pic(st::State)
  pic = zeros(Float32, limits[1], limits[2], 6, 1)
  for i in 1:2
    for j in 1:5
      x, y = st.positions[i].pieces[:, j]
      pic[x, y, i, 1] = 1
    end
    x, y = st.positions[i].ball
    pic[x, y, 2 + i, 1] = 1
    boundary = (i==1) ? 1 : limits[2]
    pic[:, boundary, 4 + i, 1] .= 1
  end
  pic
end

function static_train()
  device!(sorted_gpus()[1])
  net, cpu_st, cpu_ps = make_net()
  st, ps = (gpu(Lux.trainmode(cpu_st)), gpu(cpu_ps))
  opt = OptimiserChain(ClipGrad(1.0), Optimisers.AdaBelief(8f-4))
  st_opt = Optimisers.setup(opt, ps)
  h5 = h5open("small_gamedb.h5", "r")
  N = Int(length(h5["values"]))
  lg=TBLogger("srun", min_level=Logging.Info)
  counter = 0
  with_logger(lg) do
    for epoch in 1:1000
      for ix in 1:64:(N-63)
        counter += 1
        (games, values) = get_batch(h5, ix:ix+63)
        pics = gpu(cat4(as_pic.(games)))
        loss, grad = withgradient(ps) do ps 
          q_pred, st = Lux.apply(net, pics, ps, st)
          mean(abs2.(q_pred .- gpu(values)))
        end
        st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
        if counter % 50 == 1
          @info "trainer" loss
        end
        if counter % 200 == 199
          cpu_params = (cpu(ps), cpu(st))
          @save "static-checkpoint.bson" cpu_params
        end
      end
    end
  end
  cpu_params = (cpu(ps), cpu(st))
  @save "static-checkpoint.bson" cpu_params
end

cat3(x,y) = cat(x,y; dims=3)

Unpad() = WrappedFunction(x-> pad_zeros(x, 1; dims=(1,2)))

function make_net()
  net = Chain([
    SkipConnection(Chain([Conv((3,3), 6=>8, swish), Unpad()]), cat3),
    SkipConnection(Chain([Conv((3,3), 6+8=>8, swish), Unpad()]), cat3),
    Conv((3,3), 6+2*8=>32, swish),
    FlattenLayer(),
    Dense(32*15, 1, tanh),
    WrappedFunction(x->clamp.(x, -0.95, 0.95))])
  rng = Random.default_rng()
  cpu_params = Lux.setup(rng, net)
  if isfile("ab-checkpoint.bson")
    @load "ab-checkpoint.bson" cpu_params
    println("Loaded weights")
  end
  cpu_ps, cpu_st = cpu_params
  net, cpu_st, cpu_ps
end

function approx_vals(st::Vector{State}, gpucom::GPUCom)
  trans, nst = unzip(normalize_player.(st))
  batch = cat4(as_pic.(nst))
  put!(gpucom.req_chan, (batch, gpucom.val_chan))
  trans .* take!(gpucom.val_chan)
end

approx_vals(st::Vector{State}, ::Nothing) = zeros(Float32, length(st))

cat4(stack) = reduce((x,y)->cat(x,y; dims=4), stack)

struct GameResult
  value::Float32
  states::Vector{State}
end

function with_values(game::GameResult)
  trans, nsts = unzip(normalize_player.(game.states))
  values = trans .* game.value
  nsts, values
end

function game_q(result)
  if isnothing(result.winner)
    q = 0f0
  elseif result.winner == 1
    q = 1f0
  else
    q = -1f0
  end
end
