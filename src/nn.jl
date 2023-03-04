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

function make_net()
  net = Chain([
    CrossCor((3,3), 4=>8, relu),
    CrossCor((3,3), 8=>16, relu),
    CrossCor((3,3), 16=>32, relu),
    FlattenLayer(),
    Dense(64, 1, tanh)])
  rng = Random.default_rng()
  ps, st = Lux.setup(rng, net)
  (;net, st), ps
end

struct GameResult
  value::Float32
  states::Vector{State}
end

function as_pics(game::GameResult)
  multipliers = [1; cumprod(fill(discount, length(game.states) - 1))]
  reverse!(multipliers)
  stack = as_pic.(game.states)
  reduce((x,y)->cat(x,y; dims=4), stack), multipliers
end

function test_net()
  cfg, cpu_ps = make_net()
  ps = gpu(cpu_ps)
  result = simulate(start_state, greedy_players; track=true) 
  gameres = GameResult(result.winner == 1 ? 1f0 : -1f0, result.states)
  pics, values = gpu.(as_pics(gameres))
  loss, grad = withgradient(ps) do ps
    q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
    mean(abs2.(q_pred .- values))
  end
  loss
end

function trainer(cfg, cpu_ps, game_chan, param_chan)
  ps = gpu(cpu_ps) 
  st_opt = Optimisers.setup(Optimisers.ADAM(1f-3), ps)
  vd=Visdom("mcts_greedy")
  running_loss = 0f0
  for (ix, game) in enumerate(game_chan)
    println("Got game")
    pics, values = gpu.(as_pics(game))
    loss, grad = withgradient(ps) do ps
      q_pred, _ = Lux.apply(cfg.net, pics, ps, cfg.st)
      mean(abs2.(q_pred .- values))
    end
    running_loss = 0.1f0 * running_loss + 0.9f0 * loss
    st_opt, ps = Optimisers.update(st_opt, ps, grad[1])
    # take!(param_chan)
    # put!(param_chan, ps)
    report(vd, "loss", ix, running_loss)
    if ix % 20 == 19
      cpu_ps = cpu(ps)
      @save "greedy-checkpoint.bson" cpu_ps
    end
    println("Done updating")
  end 
end

function player(cfg, game_chan, param_chan)
  while true
    # params = fetch(param_chan)
    # neural_player = Neural(cfg.net, params, cfg.st, 50.0)
    # rollout_players = (neural_player, neural_player)
    rollout_players = greedy_players
    mc = MC(players=rollout_players, steps=20)
    players = (mc, mc)
    result = simulate(start_state, players; track=true) 
    if isnothing(result.winner)
      q = 0f0
    elseif result.winner == 1
      q = 1f0
    else
      q = -1f0
    end
    gameres = GameResult(q, result.states)
    put!(game_chan, gameres) 
  end
end

function train_loop()
  cfg, ps = make_net()
  game_chan = Channel{GameResult}(Threads.nthreads())
  param_chan = Channel{typeof(ps)}(1)
  # put!(param_chan, ps)
  train_thread = @spawn trainer(cfg, ps, game_chan, param_chan)
  bind(game_chan, train_thread)
  bind(param_chan, train_thread)
  try
    @sync for p in 1:Threads.nthreads()
      player_thread = @spawn player(cfg, game_chan, param_chan)
      bind(param_chan, player_thread)
    end
  catch my_exception
    if isa(my_exception, InterruptException)
      close(param_chan)
    else rethrow()
    end
  end
end

