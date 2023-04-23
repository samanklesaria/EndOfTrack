using TensorBoardLogger, Logging

const TRAIN_BATCH_SIZE = 64
const MIN_TRAIN_SIZE = 128
const EVAL_BATCH_SIZE = 128
const REPLAY_SIZE = 700_000

const ReplayBuffer = CircularBuffer{Tuple{State, Float32}}

function sorted_gpus()
  "Gets least busy gpus"
  usage = parse.(Int, split(readchomp(`./getter.sh`), "\n"))
  sortperm(usage)
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

# Used to indicate that updated NN params are available
mutable struct NewParams{P}
  @atomic n::Union{Nothing, P}
end

val_to_prob(a::Float32) = (a + 1) / 2

logit_to_val(a::Float32) = clamp((sigmoid_fast(a) * 2) - 1, -0.96, 0.96)

function trainer(net, req, buffer_chan::Channel{ReplayBuffer}, np::Vector{NewParams})
  seed = rand(TaskLocalRNG(), UInt8)
  Flux.trainmode!(net)
  opt = Flux.setup(Flux.Adam(1f-4), net)
  lg=TBLogger("srun", min_level=Logging.Info)
  predictions = Float32[]
  println("Started training loop")
  with_logger(lg) do
    for ix in Iterators.countfrom()
      buffer = take!(buffer_chan)
      l = length(buffer)
      if l >= MIN_TRAIN_SIZE
        ixs = sample(1:l, TRAIN_BATCH_SIZE)
        batch = buffer[ixs]
        put!(buffer_chan, buffer)
        nsts, values = unzip(batch)
        prob_values = val_to_prob.(values)
        pic_batch = gpu(cat4(as_pic.(nsts)))  
        loss, grads = Flux.withgradient(net) do m
          q_pred = m(pic_batch)
          predictions = vec(cpu(q_pred))
          Flux.logitbinarycrossentropy(q_pred, gpu(prob_values[na, :]))
        end
        Flux.update!(opt, net, grads[1])
        if ix % 50 == 1
          @info "trainer" loss
          log_histogram(lg, "predictions", predictions)
          log_histogram(lg, "values", prob_values)
        end
        if ix % 1000 == 1
          cpu_net = cpu(net)
          @save "checkpoint.bson" cpu_net
          println("Saved")
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
        if ix % 2000 == 1999
          take!(buffer_chan)
          cpu_net = cpu(net)
          for n in np
            @atomic n.n = cpu_net
          end
          put!(buffer_chan, ReplayBuffer(REPLAY_SIZE))
        end
      else
        put!(buffer_chan, buffer)
        println("Not enough to train")
        sleep(30)
      end
    end
  end
end

cat4(stack) = reduce((x,y)->cat(x,y; dims=4), stack)     
cat3(x,y) = cat(x,y; dims=3)
pad(x) = pad_zeros(x, 1; dims=(1,2))

function make_net()
  cpu_net = Chain([
    Conv((3,3), 6=>16, swish),
    BatchNorm(16),
    Conv((3,3), 16=>32, swish), 
    BatchNorm(32),
    Conv((3,3), 32=>64, swish), 
    Flux.flatten,
    BatchNorm(128),
    Dense(128, 1)
  ])
  if isfile("checkpoint.bson")
    @load "checkpoint.bson" cpu_net
    println("Loaded weights")
  end
  cpu_net
end

function approx_vals(st::Vector{State}, gpucom::GPUCom)
  trans, nst = unzip(normalize_player.(st))
  batch = cat4(as_pic.(nst))
  put!(gpucom.req_chan, (batch, gpucom.val_chan))
  trans .* take!(gpucom.val_chan)
end

approx_vals(st::Vector{State}, ::Nothing) = zeros(Float32, length(st))

function evaluator(req::ReqChan, newparams::NewParams)
  net = gpu(newparams.n)
  @atomic newparams.n = nothing
  Flux.testmode!(net)
  while true
    if !isnothing(newparams.n)
      net = gpu(newparams.n)
      Flux.testmode!(net)
    end
    fetch(req)
    pics, outs = unzip([take!(req) for _ in 1:req.n_avail_items if isready(req)])
    if length(pics) == 0
      continue
    end
    sizes = size.(pics, 4)
    cumsizes = [0; cumsum(sizes)]
    batch = gpu(cat4(pics))
    values = logit_to_val.(net(batch))
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


function player(buffer_chan::Channel{ReplayBuffer}, req::ReqChan)
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
          buffer = take!(buffer_chan)
          append!(buffer, collect(zip(nsts, values)))
          put!(buffer_chan, buffer)
        catch exc
          @error exception=exc
        end
      end
    end
  end
end

function train_loop()
  net = make_net()
  buffer_chan = Channel{ReplayBuffer}(1)
  req = ReqChan(EVAL_BATCH_SIZE)
  put!(buffer_chan, ReplayBuffer(REPLAY_SIZE))
  N_WORK=20
  N_EVAL=4
  newparams = NewParams[NewParams(net) for _ in 1:N_EVAL]
  gpus = Iterators.Stateful(sorted_gpus())
  for i in 2:(N_WORK+1)
    t1 = @tspawnat i player(buffer_chan, req)
    bind(req, t1)
    errormonitor(t1)
  end
  for i in 1:N_EVAL
    t = @tspawnat (N_WORK + 1 + i) begin
      device!($(popfirst!(gpus)))
      evaluator(req, newparams[$i])
    end
    bind(req, t)
    errormonitor(t)
  end
  tt = @tspawnat (N_WORK + N_EVAL + 2) begin
    device!($(popfirst!(gpus)))
    trainer(gpu(net), req, buffer_chan, newparams)
  end
  errormonitor(tt)
  bind(req, tt)
end

# TODO: indicate to workers somehow that a new thing is available.
# Or maybe not?
