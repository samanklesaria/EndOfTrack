function as_pic(st::State)
  pic = zeros(Float32, 7,8,4,1)
  for i in 1:2
    for j in 1:5
      row, col = st.positions[i].pieces[:, j]
      pic[row, col, 2 * (i-1) + 1, 1] = 1
    end
    row, col = st.positions[i].ball
    pic[row, col, 2 * (i-1) + 2, 1] = 1
  end
  pic
end

function make_big_net()
  net = Chain([
    CrossCor((3,3), 4=>8, relu),
    CrossCor((3,3), 8=>8, relu),
    CrossCor((3,3), 8=>8, relu),
    FlattenLayer(),
    Dense(16, 1, tanh)])
  rng = Random.default_rng()
  ps, st = Lux.setup(rng, net)
  if isfile("checkpoint.bson")
    @load "checkpoint.bson" ps
    println("Loaded weights")
  end
  (;net, st), ps
end


function make_small_net()
  net = Chain([
    CrossCor((3,3), 4=>8, relu),
    CrossCor((3,3), 8=>8, relu),
    FlattenLayer(),
    Dense(96, 1, tanh)])
  rng = Random.default_rng()
  ps, st = Lux.setup(rng, net)
  if isfile("small_checkpoint.bson")
    @load "small_checkpoint.bson" ps
    println("Loaded weights")
  end
  (;net, st), ps
end

struct Neural{NN, P, S}
  net::NN
  ps::P
  st::S
  temp::Float32
end

function sample_choices(st, raw_choices, reward)
  k = min(length(raw_choices), 5)
  choices = Vector{ValuedAction}(undef, k)
  sofar = 0
  for a in raw_choices
    new_st = apply_action(st, a)
    if is_terminal(new_st)
      return [ValuedAction(a, reward)]
    end
    if sofar < k
      sofar += 1
      choices[sofar] = ValuedAction(a, 0)
    else
      choices[rand(1:k)] = ValuedAction(a, 0)
    end
  end
  @assert sofar == k
  choices
end

# Need to normalize before calling this function 

function (b::Neural)(st::State)
  raw_choices = actions(st)
  player = st.player == 1 ? 1 : -1
  choices = sample_choices(st, raw_choices, player)
  if length(choices) == 1
    return choices[1]
  end
  batch = cat4(as_pic.(apply_action.(Ref(st), choices)))
  values, _ = Lux.apply(b.net, batch, b.ps, b.st)
  normalized_values = player .* values[1,:] .* b.temp
  softmax!(normalized_values)
  ix = sample(1:length(choices), Weights(normalized_values))
  ValuedAction(choices[ix].action, normalized_values[ix])
end

function approx_val(b::Neural, st::State)
  batch = as_pic(st)
  values, _ = Lux.apply(b.net, batch, b.ps, b.st)
  values[1]
end

struct GameResult
  value::Float32
  states::Vector{State}
end

cat4(stack) = reduce((x,y)->cat(x,y; dims=4), stack)

function as_pics(game::GameResult)
  multipliers = [1; cumprod(fill(discount, length(game.states) - 1))]
  reverse!(multipliers)
  stack = as_pic.(game.states)
  cat4(stack), multipliers
end

function test_net()
  cfg, ps = make_small_net()
  result = simulate(start_state, greedy_players; track=true) 
  gameres = GameResult(result.winner == 1 ? 1f0 : -1f0, result.states)
  pics, values = gpu.(as_pics(gameres))
  q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
  mean(abs2.(q_pred[1,:] .- values))
end

function trainer(cfg, ps, game_chan, param_chan)
  println("Started trainer loop")
  st_opt = Optimisers.setup(Optimisers.AdaBelief(1f-3), ps)
  vd=Visdom("treesearch")
  running_loss = 0f0
  println("About to get games")
  for (ix, game) in enumerate(game_chan)
    println("Got a game")
    pics, values = as_pics(game)
    loss, grad = withgradient(ps) do ps
      q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
      mean(abs2.(q_pred .- values))
    end
    running_loss = 0.1f0 * running_loss + 0.9f0 * loss
    st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
    take!(param_chan)
    put!(param_chan, ps)
    report(vd, "loss", ix, running_loss)
    if ix % 5 == 4
      @save "neural-checkpoint.bson" ps
      println("Saved")
    end
    if ix % 10 == 9
      val_q = validate(cfg, ps)
      report(vd, "validation", ix, val_q)
      report(vd, "weights", fleaves(ps))
    end
    println("Loss ", loss)
  end 
end

function neural_ab_trainer(cfg, ps, game_chan, param_chan)
  println("Started trainer loop")
  st_opt = Optimisers.setup(Optimisers.AdaBelief(1f-3), ps)
  vd=Visdom("neural_ab")
  running_loss = 0f0
  println("About to get games")
  for (ix, game) in enumerate(game_chan)
    println("Got a game")
    pics, values = as_pics(game)
    loss, grad = withgradient(ps) do ps
      q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
      mean(abs2.(q_pred .- values))
    end
    running_loss = 0.1f0 * running_loss + 0.9f0 * loss
    st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
    take!(param_chan)
    put!(param_chan, ps)
    report(vd, "loss", ix, running_loss)
    if ix % 5 == 4
      @save "neural-checkpoint.bson" ps
      println("Saved")
    end
    if ix % 10 == 9
      val_q = ab_validate(cfg, ps)
      report(vd, "validation", ix, val_q)
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

function validate(cfg, params)
    neural_player = Neural(cfg.net, params, cfg.st, 50f0)
    rollout_players = (neural_player, neural_player)
    mc = MaxMCTS(players=rollout_players, steps=40, rollout_len=10)
    players = (AlphaBeta(5), mc)
    result = simulate(start_state, players) 
    game_q(result)
end

function ab_validate(cfg, params)
    neural = Neural(cfg.net, params, cfg.st, 0f0)
    players = (AlphaBeta(5, neural), AlphaBeta(5))
    result = simulate(start_state, players) 
    game_q(result)
end

function mcts_player(cfg, game_chan, param_chan)
  while true
    params = fetch(param_chan)
    println("Started game on $(Threads.threadid())")
    neural_player = Neural(cfg.net, params, cfg.st, 50f0)
    rollout_players = (neural_player, neural_player)
    mc = MaxMCTS(players=rollout_players, steps=20,
      rollout_len=10, estimator=neural_player)
    players = (mc, mc)
    result = simulate(start_state, players; track=true) 
    gameres = GameResult(game_q(result), result.states)
    println("Ended game on $(Threads.threadid())")
    put!(game_chan, gameres) 
  end
end

function ab_player(game_chan)
  ab = AlphaBeta(5)
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
  while true
    params = fetch(param_chan)
    neural = Neural(cfg.net, params, cfg.st, 0f0)
    ab = AlphaBeta(5, neural)
    players = (ab, ab)
    println("Started game on $(Threads.threadid())")
    result = simulate(start_state, players; track=true) 
    gameres = GameResult(game_q(result), result.states)
    println("Ended game on $(Threads.threadid())")
    put!(game_chan, gameres) 
  end
end

function ab_trainer(game_chan)
  println("Started trainer loop")
  cfg, ps = make_small_net()
  st_opt = Optimisers.setup(Optimisers.AdaBelief(1f-4), ps)
  vd=Visdom("abpredict")
  running_loss = 0f0
  println("About to get games")
  for (ix, game) in enumerate(game_chan)
    println("Got a game")
    pics, values = as_pics(game) # gpu.(as_pics(game))
    loss, grad = withgradient(ps) do ps
      q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
      mean(abs2.(q_pred .- values))
    end
    running_loss = 0.1f0 * running_loss + 0.9f0 * loss
    st_opt, ps = Optimisers.update!(st_opt, ps, grad[1])
    report(vd, "loss", ix, running_loss)
    if ix % 5 == 1
      report(vd, "weights", fleaves(ps))
      @save "ab-checkpoint.bson" ps
      println("Saved")
    end
    println("Loss ", loss)
  end 
end

function ab_train_loop()
  N = Threads.nthreads() - 1
  game_chan = Channel{GameResult}(N-1)
  @threads :static for i in 1:N
      if i == 1
        ab_trainer(game_chan)
      else
        ab_player(game_chan)
      end
  end
end

function neural_ab_train_loop()
  N = Threads.nthreads() - 1
  cfg, ps = make_small_net()
  game_chan = Channel{GameResult}(N-1)
  param_chan = Channel{typeof(ps)}(1)
  put!(param_chan, ps)
  @threads :static for i in 1:N
      if i == 1
        neural_ab_trainer(cfg, ps, game_chan, param_chan)
      else
        neural_ab_player(cfg, game_chan, param_chan)
      end
  end
end

function mcts_train_loop()
  N = Threads.nthreads() - 1
  cfg, ps = make_small_net()
  game_chan = Channel{GameResult}(N-1)
  param_chan = Channel{typeof(ps)}(1)
  put!(param_chan, ps)
  @threads :static for i in 1:N
      if i == 1
        trainer(cfg, ps, game_chan, param_chan)
      else
        mcts_player(cfg, game_chan, param_chan)
      end
  end
end

