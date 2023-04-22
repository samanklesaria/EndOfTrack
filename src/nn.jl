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

function get_batch(seen, h5::HDF5.File, ix)
  values = h5["values"][ix]
  players = h5["players"][ix]
  pieces1 = h5["pieces1"][ix,:,:]
  pieces2 = h5["pieces2"][ix,:,:]
  balls1 = h5["balls1"][ix,:]
  balls2 = h5["balls2"][ix,:]
  states = [State(players[i], [PlayerState(balls1[i,:], pieces1[i,:,:]),
    PlayerState(balls2[i,:], pieces2[i,:,:])]) for i in 1:length(ix)] 
  mask = states .!= Ref(start_state)
  for (s,v) in zip(states[mask], values[mask])
    if haskey(seen, s)
      @assert seen[s] == v
    else
      seen[s] = v
    end
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
  seen = Dict{State, Float32}()
  device!(sorted_gpus()[1])
  net, cpu_st, cpu_ps = make_stacknet()
  st, ps = (gpu(Lux.trainmode(cpu_st)), gpu(cpu_ps))
  opt = OptimiserChain(Optimisers.Adam(1f-3))
  st_opt = Optimisers.setup(opt, ps)
  h5 = h5open("smalls.h5", "r")
  N = Int(length(h5["values"]))
  lg=TBLogger("srun", min_level=Logging.Info)
  counter = 0
  with_logger(lg) do
    for epoch in 1:100_000
      for ix in [1] # :64:640 # :64:(N-63)
        counter += 1
        (games, values) = get_batch(seen, h5, ix:ix+63)
        pics = gpu(cat4(as_pic.(games)))
        predictions = Float32[]
        loss, grad = withgradient(ps) do ps 
          q_pred, st = net(pics, ps, st)
          predictions = vec(cpu(q_pred))
          mean((q_pred .- gpu(values)).^2)
        end
        st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
        if counter % 50 == 1
          @info "trainer" loss
          log_histogram(lg, "predictions", predictions)
          log_histogram(lg, "values", vec(values))
        end
        # if counter % 200 == 199
        #   cpu_params = (cpu(ps), cpu(st))
        #   @save "static-checkpoint.bson" cpu_params
        # end
      end
    end
  end
  cpu_params = (cpu(ps), cpu(st))
  @save "static-checkpoint.bson" cpu_params
end

cat3(x,y) = cat(x,y; dims=3)

Unpad() = WrappedFunction(x-> pad_zeros(x, 1; dims=(1,2)))

struct ResNet{C,A} <: Lux.AbstractExplicitContainerLayer{(:c1, :c2, :c3)}
  c1::C
  c2::C
  c3::A
end

function (r::ResNet)(x, ps, st)
  y, st1 = r.c1(x, ps.c1, st.c1)
  z, st2 = r.c2(cat3(y, x), ps.c2, st.c2)
  w, st3 = r.c3(cat3(y + z, x), ps.c3, st.c3)
  w, (c1=st1, c2=st2, c3=st3)
end

function make_resnet()
  net = Chain([
    ResNet(
      Chain([Conv((3,3), 6=>8, swish), Unpad()]),
      Chain([Conv((3,3), 6+8=>8, swish), Unpad()]),
      Conv((3,3), 6+8=>8, swish)),
    FlattenLayer(),
    Dense(8*15, 1, tanh),
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

function make_stacknet()
  net = Chain([
    SkipConnection(Chain([Conv((3,3), 6=>32, swish), Unpad()]), cat3),
    SkipConnection(Chain([Conv((3,3), 6+32=>32, swish), Unpad()]), cat3),
    Conv((3,3), 6+32+32=>128, swish),
    FlattenLayer(),
    Dense(1920, 1)])
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
