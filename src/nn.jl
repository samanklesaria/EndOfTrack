using TensorBoardLogger, Logging

const TRAIN_BATCH_SIZE = 64
const EVAL_BATCH_SIZE = 128

# TODO: if we encoded the positions, our replay buffer would
# take half the space.

# With validation turned on, playoffs between AlphaBeta
# and NoRoll get invalid states. Why?

# How is it possible to have the parent node use a Dirac
# distribution? 


const ReplayBuffer = CircularBuffer{Tuple{State, Float32}}
const ReplayVector = Tuple{Vector{State}, Vector{Float32}}

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
  cpu_params = Lux.setup(rng, net)
  if isfile("checkpoint.bson")
    @load "checkpoint.bson" cpu_params
    println("Loaded weights")
  end
  cpu_ps, cpu_st = cpu_params
  net, cpu_st, cpu_ps
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
    fetch(req)
    pics, outs = unzip([take!(req) for _ in 1:req.n_avail_items])
    sizes = size.(pics, 4)
    cumsizes = [0; cumsum(sizes)]
    batch = gpu(cat4(pics))
    values, _ = Lux.apply(net, batch, ps, st)
    for (i, out) in enumerate(outs)
      vals = cpu(values[(cumsizes[i] + 1):cumsizes[i+1]])
      put!(out, vals)
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
  open("errors.log", "w") do io
    with_logger(SimpleLogger(io)) do
      while true
        try
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
        catch exc
          @error exception=exc
        end
      end
    end
  end
end


function ab_player(buffer_chan::Channel{ReplayVector}, req::ReqChan, seed)
  open("errors.log", "w") do io
    with_logger(SimpleLogger(io)) do
      while true
        try
          noroll = NoRoll(req; shared=false)
          players = (noroll,  AlphaBeta(5, Xoshiro(seed)))
          println("Starting game on thread $(Threads.threadid())")
          result = simulate(start_state, players; track=true)
          gameres = GameResult(game_q(result), result.states)
          println("Finished game on $(Threads.threadid())")
          nsts, values = with_values(gameres)
          nsts2 = map_state.(Ref(flip_pos_vert), nsts)
          nsts3 = flip_players.(nsts)
          batch = ([nsts; nsts2; nsts3], [values; values; -values])
          put!(buffer_chan, batch)
        catch exc
          if isa(exc, InterruptException)
            return 
          end
          @error exception=exc
        end
      end
    end
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
          println("Saved")
          cpu_params = (cpu(ps), cpu(st))
          @save "noroll-checkpoint.bson" cpu_params
          for newparam in newparams
            @atomic newparam.n = cpu_params
          end
          println("Validating")          
          try
            valq = validate_noroll(req, seed)
            @info "validate" valq
          catch exc
            open("val_errors.log", "a") do io
              with_logger(SimpleLogger(io)) do                
                @error exception=exc
              end
            end
          end
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

function ab_trainer(buffer_chan, net, cpu_st, cpu_ps, newparams, req, seed)
  println("Started trainer loop")
  st, ps = (gpu(Lux.trainmode(cpu_st)), gpu(cpu_ps))
  opt = OptimiserChain(WeightDecay(1f-4),
    ClipGrad(1.0), Optimisers.AdaBelief(1f-4))
  st_opt = Optimisers.setup(opt, ps)
  lg=TBLogger("abrun", min_level=Logging.Info)
  with_logger(lg) do
    for ix in Iterators.countfrom()
      nsts, values = take!(buffer_chan)
      pic_batch = gpu(cat4(as_pic.(nsts)))
      loss, grad = withgradient(ps) do ps
        q_pred, st = Lux.apply(net, pic_batch, ps, st)
        mean(abs2.(q_pred .- gpu(values)))
      end
      st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
      cpu_params = (cpu(ps), cpu(st))
      for newparam in newparams
        @atomic newparam.n = cpu_params
      end
      @info "trainer" loss
      if ix % 5 == 4
        println("Saved")
        @save "ab-checkpoint.bson" cpu_params
        println("Validating")          
        try
          valq = validate_noroll(req, seed)
          @info "validate" valq
        catch exc
          if isa(exc, InterruptException)
            return 
          end
          open("val_errors.log", "a") do io
            with_logger(SimpleLogger(io)) do                
              @error exception=exc
            end
          end
        end
      end
    end 
  end
end

function async_train_loop()
  seed = rand(UInt8)
  net, st, ps = make_net()
  buffer_chan = Channel{ReplayVector}()
  req = ReqChan(EVAL_BATCH_SIZE)
  newparams = NewParams[NewParams((ps, st)) for _ in 1:2]
  @sync begin
    t = @async begin
      device!(4)
      ab_trainer(buffer_chan, net, st, ps, newparams, req, seed)
    end
    bind(buffer_chan, t)
    errormonitor(t)
    for i in 1:1
      t = @async begin
        device!(i)
        evaluator(net, req, newparams[i])
      end
      bind(req, t)
    end
    for _ in 1:1
      t = @async ab_player(buffer_chan, req, seed)
      bind(req, t)
      bind(buffer_chan, t)
    end
  end
end

function parallel_train_loop()
  net, st, ps = make_net()
  buffer_chan = Channel{ReplayBuffer}(1)
  req = ReqChan(EVAL_BATCH_SIZE)
  put!(buffer_chan, ReplayBuffer(1_000_000))
  newparams = NewParams[NewParams((ps, st)) for _ in 1:2]
  @sync begin
    for _ in 1:6
      Threads.@spawn noroll_player(buffer_chan, req)
    end
    Threads.@spawn begin
      @async begin
        device!(2)
        evaluator(net, req, newparams[1])
      end
      @async begin
        device!(6)
        evaluator(net, req, newparams[2])
      end
    end
    Threads.@spawn begin
      device!(1)
      noroll_trainer(net, ps, st, newparams, buffer_chan, req)
    end
  end
end
