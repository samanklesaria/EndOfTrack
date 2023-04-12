const TRAIN_BATCH_SIZE = 64
const EVAL_BATCH_SIZE = 32

const ReplayBuffer = CircularBuffer{Tuple{Array{Float32, 4}, Float32}}

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
    Dense(32, 1, tanh)])
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
      println("Evaluating $(length(pics)) on $(Threads.threadid())")
      batch = gpu(cat4(pics))
      values, _ = Lux.apply(net, batch, ps, st)
      for (i, out) in enumerate(outs)
        vals = cpu(values[(cumsizes[i] + 1):cumsizes[i+1]])
        @assert length(vals) == sizes[i]
        put!(out, vals)
      end
    else
      sleep(0.001)
    end
  end
end

struct GameResult
  value::Float32
  states::Vector{State}
end

function as_pics(game::GameResult)
  nsts, values = with_values(game)
  stack = as_pic.(nsts)
  stack, values
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
    pics, values = as_pics(gameres)
    pics2 = as_pic.(map_state.(Ref(flip_pos_vert), gameres.states))
    pics3 = as_pic.(flip_players.(gameres.states))
    buffer = take!(buffer_chan)
    append!(buffer, zip(pics, values))
    append!(buffer, zip(pics2, values))
    append!(buffer, zip(pics3, -values))
    put!(buffer_chan, buffer)
  end
end

function noroll_trainer(net, cpu_ps, cpu_st, newparams::Vector{NewParams},
    buffer_chan::Channel{ReplayBuffer}, req::ReqChan)
  seed = rand(TaskLocalRNG(), UInt8)
  println("Started trainer loop")
  vd=Visdom("noroll")
  st, ps = (gpu(Lux.trainmode(cpu_st)), gpu(cpu_ps))
  opt = OptimiserChain(WeightDecay(1f-4),
    ClipGrad(1.0), Optimisers.AdaBelief(1f-4))
  st_opt = Optimisers.setup(opt, ps)
  for ix in Iterators.countfrom()
    buffer = take!(buffer_chan)
    l = length(buffer)
    if l >= TRAIN_BATCH_SIZE
      println("Sampling batch")
      ixs = sample(1:l, TRAIN_BATCH_SIZE)
      batch = buffer[ixs]
      put!(buffer_chan, buffer)
      println("Training")
      pics, values = unzip(batch)
      pic_batch = gpu(cat4(pics))
      loss, grad = withgradient(ps) do ps
        q_pred, st = Lux.apply(net, pic_batch, ps, st)
        mean(abs2.(q_pred .- gpu(values)))
      end
      st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
      report(vd, "loss", ix, loss)
      if ix % 1000 == 999
        @save "noroll-checkpoint.bson" ps
        println("Saved")
        for newparam in newparams
          @atomic newparam.n = (cpu(ps), cpu(st))
        end
        val_q = validate_noroll(req, seed)
        report(vd, "validation", ix, val_q; log=false, scatter=true)
        report(vd, "weights", fleaves(ps))
      end
      println("Loss ", loss)
    else
      println("Not enough to train")
      put!(buffer_chan, buffer)
      sleep(10)
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
    players = (init!(NoRoll(req; shared=false)), AlphaBeta(5, Xoshiro(seed)))
    game_q(simulate(start_state, players))
end

# Remove explicit RNGs (as tasks have separate rngs anyways)
# Don't use pics in the replay buffer, as they take too much space.

# Start with all your threads doing alphaBeta for 30 steps. Save them to a buffer.
# Then, the runners should sample something from the buffer, and then
# save something to the same place in the buffer when they're done. 

# To start, let's just see if we can run a single noroll loop

function noroll_train_loop()
  # Threads.nthreads() - 1
  net, st, ps = make_net()
  buffer_chan = Channel{ReplayBuffer}(1)
  req = ReqChan(EVAL_BATCH_SIZE)
  put!(buffer_chan, ReplayBuffer(100_000))
  # @threads :static for i in 1:N
  newparams = NewParams[NewParams((ps, st)) for _ in 1:1]
  @sync begin
      t = @async begin
        device!(1)
        noroll_trainer(net, ps, st, newparams, buffer_chan, req)
      end
      bind(req, t); bind(buffer_chan, t)
      errormonitor(t)
      for i in 1:1
        t = @async begin
          device!(1 + $i)
          evaluator(net, req, newparams[$i])
        end
        bind(req, t); bind(buffer_chan, t)
        errormonitor(t)
      end
      for _ in 1:1
        t = @async noroll_player(buffer_chan, req)
        bind(req, t); bind(buffer_chan, t)
        errormonitor(t)
      end
  end
end

