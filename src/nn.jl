using TensorBoardLogger, Logging

const TRAIN_BATCH_SIZE = 64
const EVAL_BATCH_SIZE = 64

# TODO: if we encoded the positions, our replay buffer would
# take half the space.

# With validation turned on, playoffs between AlphaBeta
# and NoRoll get invalid states. Why?

# How is it possible to have the parent node use a Dirac
# distribution?
# Could this be some strange concurrency issue?

# Note: Doesn't use all CPUs allocated, meaning we
# should add a second evaluator thread

const ReplayBuffer = CircularBuffer{Tuple{State, Float32}}

function as_pic(st::State)
  pic = zeros(Float32, 7,8,6,1)
  for i in 1:2
    for j in 1:5
      x, y = st.positions[i].pieces[:, j]
      pic[x, y, i, 1] = 1
    end
    x, y = st.positions[i].ball
    pic[x, y, 2 + i, 1] = 1
    boundary = (i==1) ? 1 : 8
    pic[:, boundary, 4 + i, 1] .= 1
  end
  pic
end

function make_net()
  net = Chain([
    Conv((3,3), 6=>8, relu),
    BatchNorm(8),
    Conv((3,3), 8=>16, relu),
    BatchNorm(16),
    Conv((3,3), 16=>16, relu),
    BatchNorm(16),
    FlattenLayer(),
    Dense(32, 1, tanh),
    WrappedFunction(x->clamp.(x, -0.95, 0.95))])
  rng = Random.default_rng()
  ps, st = Lux.setup(rng, net)
  if isfile("checkpoint.bson")
    @load "checkpoint.bson" ps
    println("Loaded weights")
  end
  net, st, ps
end

# Used to indicate that updated NN params are available
mutable struct NewParams{P}
  @atomic n::Union{Nothing, P}
end

function approx_vals(st::Vector{State}, gpucom::GPUCom)
  trans, nst = unzip(normalize_player.(st))
  batch = cat4(as_pic.(nst))
  put!(gpucom.req_chan, (batch, gpucom.val_chan))
  trans .* take!(gpucom.val_chan)
end

approx_vals(st::Vector{State}, ::Nothing) = zeros(Float32, length(st))

cat4(stack) = reduce((x,y)->cat(x,y; dims=4), stack)

function evaluator(net, req::ReqChan, newparams::NewParams)
  cpu_ps, cpu_st = newparams.n
  @atomic newparams.n = nothing
  ps, st = gpu(cpu_ps), gpu(Lux.testmode(cpu_st))
  while true
    if !isnothing(newparams.n)
      cpu_ps, cpu_st = newparams.n
      ps, st = gpu(cpu_ps), gpu(Lux.testmode(cpu_st))
    end
    if req.n_avail_items > 0
      pics, outs = unzip([take!(req) for _ in 1:req.n_avail_items])
      sizes = size.(pics, 4)
      cumsizes = [0; cumsum(sizes)]
      batch = gpu(cat4(pics))
      values, _ = Lux.apply(net, batch, ps, st)
      for (i, out) in enumerate(outs)
        vals = cpu(values[(cumsizes[i] + 1):cumsizes[i+1]])
        @assert length(vals) == sizes[i]
        put!(out, vals)
      end
    else
      println("Empty evaluation queue")
      sleep(0.1)
    end
  end
end

struct GameResult
  value::Float32
  states::Vector{State}
end

function with_values(game::GameResult)
  trans, nsts = unzip(normalize_player.(game.states))
  values = trans .* game.value
  nsts, values
end

function noroll_player(buffer_chan::Channel{ReplayBuffer}, req::ReqChan)
  while true
    noroll = NoRoll(req; shared=true)
    players = (noroll, noroll)
    println("Starting game on thread $(Threads.threadid())")
    result = simulate(start_state, players; track=true)
    gameres = GameResult(game_q(result), result.states)
    println("Finished game on $(Threads.threadid())")
    nsts, values = with_values(gameres)
    nsts2 = map_state.(Ref(flip_pos_vert), nsts)
    nsts3 = flip_players.(nsts)
    buffer = take!(buffer_chan)
    append!(buffer, collect(zip(nsts, values)))
    append!(buffer, collect(zip(nsts2, values)))
    append!(buffer, collect(zip(nsts3, -values)))
    put!(buffer_chan, buffer)
  end
end

function noroll_trainer(net, cpu_ps, cpu_st, newparams::Vector{NewParams},
    buffer_chan::Channel{ReplayBuffer}, req::ReqChan)
  seed = rand(TaskLocalRNG(), UInt8)
  println("Started trainer loop")
  st, ps = (gpu(Lux.trainmode(cpu_st)), gpu(cpu_ps))
  opt = OptimiserChain(WeightDecay(1f-4),
    ClipGrad(1.0), Optimisers.AdaBelief(1f-4))
  st_opt = Optimisers.setup(opt, ps)
  lg=TBLogger("runs", min_level=Logging.Info)
  with_logger(lg) do
    for ix in Iterators.countfrom()
      buffer = take!(buffer_chan)
      l = length(buffer)
      if l >= TRAIN_BATCH_SIZE
        ixs = sample(1:l, TRAIN_BATCH_SIZE)
        batch = buffer[ixs]
        put!(buffer_chan, buffer)
        nsts, values = unzip(batch)
        pic_batch = gpu(cat4(as_pic.(nsts)))
        loss, grad = withgradient(ps) do ps
          q_pred, st = Lux.apply(net, pic_batch, ps, st)
          mean(abs2.(q_pred .- gpu(values)))
        end
        st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
        if ix % 10 == 9
          @info "trainer" loss
        end
        if ix % 1000 == 999
          @save "noroll-checkpoint.bson" ps
          println("Saved")
          cpu_params = (cpu(ps), cpu(st))
          for newparam in newparams
            @atomic newparam.n = cpu_params
          end
          valq = validate_noroll(req, seed)
          println("Validating")
          @info "validate" valq
        end
      else
        println("Not enough to train")
        put!(buffer_chan, buffer)
        sleep(10)
      end
    end 
  end
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

function validate_noroll(req::ReqChan, seed::UInt8)
    players = (NoRoll(req; shared=false), AlphaBeta(5, Xoshiro(seed)))
    game_q(simulate(start_state, players))
end

function noroll_train_loop()
  N = Threads.nthreads() - 1
  net, st, ps = make_net()
  buffer_chan = Channel{ReplayBuffer}(1)
  req = ReqChan(EVAL_BATCH_SIZE)
  put!(buffer_chan, ReplayBuffer(1_000_000))
  newparams = NewParams[NewParams((ps, st)) for _ in 1:2]
  @threads :static for i in 1:N
      if i == 1
        device!(1)
        noroll_trainer(net, ps, st, newparams, buffer_chan, req)
      elseif i == 2
        device!(2)
        evaluator(net, req, newparams[i])
      else
        noroll_player(buffer_chan, req)
      end
  end
end
