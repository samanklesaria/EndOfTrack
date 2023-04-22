using TensorBoardLogger, Logging

const TRAIN_BATCH_SIZE = 64
const EVAL_BATCH_SIZE = 128

# TODO: if we encoded the positions, our replay buffer would
# take half the space.

# With validation turned on, playoffs between AlphaBeta
# and NoRoll get invalid states. Why?

# How is it possible to have the parent node use a Dirac
# distribution? Why do we occasionally get this error? 

# Maybe look at gradients?
# Maybe look at outputs? Also variance of outputs?
# Maybe look at weights?

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
  for s in states
    @assert s.player == 1
  end
  (states[mask], (values[mask] .+ 1) ./ 2)
end

function as_pic(st::State)
  @assert st.player == UInt8(1)
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
  net = make_net()
  Flux.trainmode!(net)
  h5 = h5open("gamedb.h5", "r")
  opt = Flux.setup(Flux.Adam(5f-4), net)
  N = Int(length(h5["values"]))
  lg=TBLogger("srun", min_level=Logging.Info)
  counter = 0
  with_logger(lg) do
    for epoch in 1:100_000
      for ix in 1:64:(N-63)
        counter += 1
        (games, values) = get_batch(h5, ix:ix+63)
        pics = gpu(cat4(as_pic.(games)))
        predictions = Float32[]
        loss, grads = Flux.withgradient(net) do m
          q_pred = m(pics)
          predictions = vec(cpu(q_pred))
          Flux.logitbinarycrossentropy(q_pred, gpu(values[na, :]))
        end
        Flux.update!(opt, net, grads[1])
        if counter % 50 == 1
          @info "trainer" loss
          log_histogram(lg, "predictions", predictions)
          log_histogram(lg, "values", vec(values))
        end
      end
    end
  end
end

cat3(x,y) = cat(x,y; dims=3)
pad(x) = pad_zeros(x, 1; dims=(1,2))


function make_net()
  Chain([
    Conv((3,3), 6=>16, swish),
    BatchNorm(16),
    Conv((3,3), 16=>32, swish), 
    BatchNorm(32),
    Conv((3,3), 32=>64, swish), 
    Flux.flatten,
    BatchNorm(128),
    Dense(128, 1)
  ]) |> gpu
end
# function make_stacknet()
#   Chain([
#     SkipConnection(Chain([Conv((3,3), 6=>32, swish), pad]), cat3),
#     SkipConnection(Chain([Conv((3,3), 6+32=>32, swish), pad]), cat3),
#     Conv((3,3), 6+32+32=>128, swish),
#     Flux.flatten,
#     Dense(1920, 1)]) |> gpu
# end

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
