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

function approx_vals(st::Vector{State}, coms::GPUCom)
  trans, nst = unzip(normalize_player.(st))
  batch = cat4(as_pic.(nst))
  println("<$(size(batch, 4))")
  put!(coms.req_chan, (batch, coms.val_chan))
  vals = trans .* take!(coms.val_chan)
  println(">")
  vals
end

cat4(stack) = reduce((x,y)->cat(x,y; dims=4), stack)

function evaluator(net, coms::ReqChan, newparams::NewParams)
  cpu_ps, cpu_st = newparams.n
  @atomic newparams.n = nothing
  ps, st = gpu(cpu_ps), gpu(Lux.testmode(cpu_st))
  println("Initialized evaluator")
  while true
    if !isnothing(newparams.n)
      cpu_ps, cpu_st = newparams.n
      ps, st = gpu(cpu_ps), gpu(testmode(cpu_st))
    end
    if coms.n_avail_items > 0
      pics, outs = unzip([take!(coms) for _ in 1:coms.n_avail_items])
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
      println("Bach eval on $(Threads.threadid()) complete")
    else
      println("Not enough to process ($(coms.n_avail_items))")
      sleep(1)
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
  cat4(stack), values
end

function with_values(game::GameResult)
  multipliers = [1; cumprod(fill(discount, length(game.states) - 1))]
  reverse!(multipliers)
  trans, nsts = unzip(normalize_player.(game.states))
  values = multipliers .* trans .* game.value
  nsts, values
end

function make_noroll(coms::ReqChan; shared=true)
  nr = NoRoll(shared=shared, req=coms)
  println("Making noroll")
  gpucom = GPUCom(coms, nr.task_chans[1])
  node = init_state!(nr, nothing, start_state, gpucom)
  nr.root[] = node
  println("Done making noroll")
  nr
end

function noroll_player(buffer_chan::Channel{ReplayBuffer}, coms::ReqChan)
  while true
    noroll = make_noroll(coms)
    players = (noroll, noroll)
    println("Starting game on thread $(Threads.threadid())")
    result = simulate(start_state, players; track=true)
    gameres = GameResult(game_q(result), result.states)
    println("Finished game on $(Threads.threadid())")
    pics, values = as_pics(gameres)
    pics2 = cat4(as_pic.(map_state.(Ref(flip_pos_vert), gameres.states)))
    buffer = take!(buffer_chan)
    append!(buffer, zip(pics, values))
    append!(buffer, zip(pics2, values))
    put!(buffer_chan, buffer)
  end
end

function noroll_trainer(net, cpu_ps, cpu_st, newparams::Vector{NewParams},
    buffer_chan::Channel{ReplayBuffer}, req_chan::ReqChan)
  seed = rand(TaskLocalRNG(), UInt8)
  validate_com = GPUCom(req_chan, Channel{Vector{Float32}}(0))
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
        val_q = validate_noroll(validate_com, seed)
        report(vd, "validation", ix, val_q; log=false, scatter=true)
        report(vd, "weights", fleaves(ps))
      end
      println("Loss ", loss)
    else
      println("Not enough to train")
      put!(buffer_chan, buffer)
      sleep(5)
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

function validate_noroll(com::GPUCom, seed::UInt8)
    players = (make_noroll(com; shared=false), AlphaBeta(5, Xoshiro(seed)))
    game_q(simulate(start_state, players))
end

# TODO: raise the temperature

function noroll_train_loop()
  # Threads.nthreads() - 1
  net, st, ps = make_net()
  buffer_chan = Channel{ReplayBuffer}(1)
  coms = ReqChan(EVAL_BATCH_SIZE)
  put!(buffer_chan, ReplayBuffer(300_000))
  # @threads :static for i in 1:N
  newparams = NewParams[NewParams((ps, st)) for _ in 1:1]
  @sync begin
      t = @async begin
        device!(1)
        noroll_trainer(net, ps, st, newparams, buffer_chan, coms)
      end
      bind(coms, t); bind(buffer_chan, t)
      errormonitor(t)
      for i in 1:1
        t = @async begin
          device!(1 + $i)
          evaluator(net, coms, newparams[$i])
        end
        bind(coms, t); bind(buffer_chan, t)
        errormonitor(t)
      end
      for _ in 1:1
        t = @async noroll_player(buffer_chan, coms)
        bind(coms, t); bind(buffer_chan, t)
        errormonitor(t)
      end
  end
end

