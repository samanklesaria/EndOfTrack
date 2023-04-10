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
    SkipConnection(
      Conv((3,3), 16=>16, relu), +),
    BatchNorm(16),
    FlattenLayer(),
    Dense(32, 1, tanh)])
  rng = Random.default_rng()
  ps, st = Lux.setup(rng, net)
  if isfile("checkpoint.bson")
    @load "checkpoint.bson" ps
    println("Loaded weights")
  end
  (;net, st), ps
end

mutable struct NeuralEval{N,P,S}
  out_chan::Channel{Pair{Array{Float32, 4}}, Int}
  in_chan::Vector{Channel{Float32}}
  net::N
  params::P
  state::S
end

function approx_q_val(heuristic, st::State, a::Action, ix)
  new_st = apply_action(st, a)
  new_st = @set new_st.player = next_player(new_st.player)
  approx_val(heuristic, new_st, ix)
end

function approx_val(b::NeuralEval, st::State, ix)
  trans, nst = normalize_player(st)
  batch = as_pic(nst)
  put!(b.out_chan, batch=>ix)
  trans * take!(b.in_chan[ix])
end

cat4(stack) = reduce((x,y)->cat(x,y; dims=4), stack)

function evaluator(n::NeuralEval, batch_size)
  pics, ixs = unzip([take!(n.out_chan) for _ in 1:batch_size])
  batch = cat4(pics)
  values, _ = Lux.apply(n.net, batch, n.params, n.state)
  for (v, i) in zip(values, ixs)
    put!(n.in_chan[i], v)
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
  trans, nsts = unzip(normalized.(game.states))
  values = multipliers .* [t.value_map for t in trans] .* game.value
  nsts, values
end

# Channel{CircularBuffer{State}}

function NoRoll(neural::NeuralEval, task)
  nr = NoRoll(estimator=neural)
  node = init_state!(nr, nothing, start_state, task)
  nr.root[] = node
  nr
end

function noroll_player(neural, buffer_chan)
  while true
    players = (NoRoll(), NoRoll())
    result = simulate(start_state, players; track=true)
    gameres = GameResult(game_q(result), result.states)
    pics, values = as_pics(gameres)
    buffer = take!(buffer_chan)
    append!(buffer, zip(pics, values))
    put!(buffer_chan, buffer)
  end
end

function noroll_trainer(param_chan, buffer_chan, cfg, ps)
  seed = rand(TaskLocalRNG(), UInt8)
  println("Started trainer loop")
  st_opt = Optimisers.setup(Optimisers.AdaBelief(1f-4), ps)
  vd=Visdom("noroll")
  for ix in Iterators.countfrom()
    buffer = take!(buffer_chan)
    l = length(buffer)
    put!(buffer_chan, buffer)
    ixs = sample(1:l, 64)
    buffer = take!(buffer_chan)
    batch = buffer[ixs]
    put!(buffer_chan, buffer)
    pics, values = unzip(batch)
    pic_batch = cat4(pics)
    loss, grad = withgradient(ps) do ps
      q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
      mean(abs2.(q_pred .- values))
    end
    st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
    take!(param_chan)
    put!(param_chan, ps)
    report(vd, "loss", ix, loss)
    if ix % 5 == 4
      @save "noroll-checkpoint.bson" ps
      println("Saved")
    end
    if ix % 10 == 9
      val_q = validate_noroll(cfg, ps, seed)
      report(vd, "validation", ix, val_q; log=false, scatter=true)
      report(vd, "weights", fleaves(ps))
    end
    println("Loss ", loss)
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

function validate_noroll(cfg, params, seed)
    neural = NeuralValue(cfg.net, params, cfg.st)
    rng = Xoshiro(seed)
    players = (NoRoll(steps=1000, estimator=neural),
      AlphaBeta(4, nothing, rng))
    result = simulate(start_state, players) 
    game_q(result)
end

function ab_player(game_chan)
  rng = Xoshiro(rand(TaskLocalRNG(), UInt8))
  ab = AlphaBeta(5, nothing, rng)
  players = (ab, ab)
  while true
    println("Started game on $(Threads.threadid())")
    result = simulate(start_state, players; track=true) 
    gameres = GameResult(game_q(result), result.states)
    println("Ended game on $(Threads.threadid())")
    put!(game_chan, gameres) 
  end
end

function neural_ab_player(cfg, game_chan, param_chan)
  rng = Xoshiro(rand(TaskLocalRNG(), UInt8))
  while true
    params = take!(param_chan)
    put!(param_chan, params)
    neural = NeuralValue(cfg.net, params, cfg.st)
    ab = AlphaBeta(4, neural, rng)
    players = (ab, ab)
    println("Started game on $(Threads.threadid())")
    result = simulate(start_state, players; track=true) 
    gameres = GameResult(game_q(result), result.states)
    println("Ended game on $(Threads.threadid())")
    put!(game_chan, gameres) 
  end
end

function noroll_player(cfg, game_chan, param_chan)
  while true
    params = take!(param_chan)
    put!(param_chan, params)
    neural = NeuralValue(cfg.net, params, cfg.st)
    noroll = NoRoll(steps=1000, estimator=neural)
    players = (noroll,noroll)
    println("Started game on $(Threads.threadid())")
    result = simulate(start_state, players; track=true) 
    gameres = GameResult(game_q(result), result.states)
    println("Ended game on $(Threads.threadid())")
    put!(game_chan, gameres) 
  end
end

function ab_trainer(game_chan)
  seed = rand(TaskLocalRNG(), UInt8)
  println("Started trainer loop")
  cfg, ps = make_small_net()
  st_opt = Optimisers.setup(Optimisers.AdaBelief(1f-4), ps)
  vd=Visdom("abpredict")
  println("About to get games")
  for (ix, game) in enumerate(game_chan)
    println("Got a game")
    pics, values = as_pics(game)
    loss, grad = withgradient(ps) do ps
      q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
      mean(abs2.(q_pred .- values))
    end
    st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
    report(vd, "loss", ix, loss)
    if ix % 5 == 4
      report(vd, "weights", fleaves(ps))
      @save "ab-checkpoint.bson" ps
      println("Saved")
    end
    if ix % 10 == 9
      val_q = validate_ab(cfg, ps, seed)
      report(vd, "validation", ix, val_q; log=false, scatter=true)
      report(vd, "weights", fleaves(ps))
    end
    println("Loss ", loss)
  end 
end

struct GameDataset
  file::HDF5.File
end

Base.length(::GameDataset) = 543741

GameDataset() = GameDataset(h5open("gamedb.h5", "r"))

function Base.getindex(g::GameDataset, ix::Int)
  value = g.file["values"][ix]
  player = g.file["players"][ix]
  pieces1 = SMatrix{2,5}(g.file["pieces1"][ix,:,:])
  pieces2 = SMatrix{2,5}(g.file["pieces2"][ix,:,:])
  balls1 = Pos(g.file["balls1"][ix,:])
  balls2 = Pos(g.file["balls2"][ix,:])
  State(player, SVector{2}([PlayerState(balls1, pieces1),
    PlayerState(balls2, pieces2)]))=>value
end

function Base.getindex(g::GameDataset, ix::UnitRange)
  values = g.file["values"][ix]
  players = g.file["players"][ix]
  pieces1 = g.file["pieces1"][ix,:,:]
  pieces2 = g.file["pieces2"][ix,:,:]
  balls1 = g.file["balls1"][ix,:]
  balls2 = g.file["balls2"][ix,:]
  ([State(players[i], [PlayerState(balls1[i,:], pieces1[i,:,:]),
    PlayerState(balls2[i,:], pieces2[i,:,:])]) for i in 1:length(ix)], 
    values)
end

function to_hdf5()
  N = 543741
  open("games", "r") do reader
    h5 = h5open("gamedb.h5", "w")
    values = create_dataset(h5, "values", Float32, (N,), chunk=(64,))
    players = create_dataset(h5, "players", Int8, (N,), chunk=(64,))
    pieces1 = create_dataset(h5, "pieces1", Int8, (N,2,5), chunk=(64,2,5))
    pieces2 = create_dataset(h5, "pieces2", Int8, (N,2,5), chunk=(64,2,5))
    balls1 = create_dataset(h5, "balls1", Int8, (N,2), chunk=(64, 2))
    balls2 = create_dataset(h5, "balls2", Int8, (N,2), chunk=(64, 2))
    ixs = randperm(N)
    counter = 0
    for i in 1:787654
        data = deserialize(reader)
        if abs(data[2]) > 0.5
          counter += 1
          ix = ixs[counter]
          values[ix] = data[2]
          players[ix] = data[1].player
          balls1[ix, :] = data[1].positions[1].ball
          balls2[ix, :] = data[1].positions[2].ball
          pieces1[ix, :, :] = data[1].positions[1].pieces
          pieces2[ix, :, :] = data[1].positions[2].pieces
        end
    end
    close(h5)
  end
end


function count_gathered_data()
  counter = 0
  open("games", "r") do io
    while !eof(io)
      try
        d = deserialize(io)
        if abs(d[2]) > 0.5
          counter += 1
        end
      catch
      end
    end
  end
  counter
end

# TODO: do data augmentation


function static_train()
  game_loader = DataLoader(GameDataset(); batchsize=64)
  cfg, cpu_ps = make_small_net()
  st = gpu(trainmode(cfg.st))
  ps = gpu(cpu_ps)
  st_opt = Optimisers.setup(Optimisers.AdaBelief(1f-5), ps)
  vd=Visdom("staticab")
  counter = 0
  for epoch in 1:3000
    for (games, values) in game_loader
        counter += 1
        pics = gpu(cat4(as_pic.(games)))
        loss, grad = withgradient(ps) do ps 
          q_pred, st = Lux.apply(cfg.net, pics, ps, st)
          mean(abs2.(q_pred .- gpu(values)))
        end
        st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
        if counter % 50 == 49
          report(vd, "loss", counter, loss)
        end
        if counter % 200 == 199
          @save "static-checkpoint.bson" ps
        end
    end
  end
  @save "static-checkpoint.bson" ps
end

function noroll_train_loop()
  N = Threads.nthreads()
  cfg, ps = make_small_net()
  game_chan = Channel{GameResult}(N-1)
  param_chan = Channel{typeof(ps)}(1)
  put!(param_chan, ps)
  @threads :static for i in 1:N
      if i == 1
        noroll_trainer(cfg, ps, game_chan, param_chan)
      else
        noroll_player(cfg, game_chan, param_chan)
      end
  end
end

