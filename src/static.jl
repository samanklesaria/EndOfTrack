using HDF5

const GameChan = Channel{Vector{Tuple{State, Float32}}}
# const GameChan = Channel{Tuple{Vector{State}, Vector{Float32}}}

function ab_selfplay(game_chan::GameChan)
  ab = AlphaBeta(5, Xoshiro(rand(UInt8)))
  players = (ab, ab)
  while true
    println("Started game on $(Threads.threadid())")
    result = simulate(start_state, players; track=true) 
    start = max(length(result.states) - 9, 1)
    gameres = GameResult(game_q(result), result.states[start:end])
    println("Ended game on $(Threads.threadid())")
    nsts, values = with_values(gameres)
    # put!(game_chan, (nsts, values))
    put!(game_chan, collect(zip(nsts, values)))
  end
end

function writer(game_chan::GameChan)
  h5open("gamedb.h5", "w") do h5
    N = 5_000
    ixs = randperm(N)
    values = create_dataset(h5, "values", Float32, (N,), chunk=(64,))
    players = create_dataset(h5, "players", Int8, (N,), chunk=(64,))
    pieces1 = create_dataset(h5, "pieces1", Int8, (N,2,5), chunk=(64,2,5))
    pieces2 = create_dataset(h5, "pieces2", Int8, (N,2,5), chunk=(64,2,5))
    balls1 = create_dataset(h5, "balls1", Int8, (N,2), chunk=(64, 2))
    balls2 = create_dataset(h5, "balls2", Int8, (N,2), chunk=(64, 2))
    for (ix, data) in zip(ixs, Iterators.flatten(game_chan))
      values[ix] = data[2]
      players[ix] = data[1].player
      balls1[ix, :] = data[1].positions[1].ball
      balls2[ix, :] = data[1].positions[2].ball
      pieces1[ix, :, :] = data[1].positions[1].pieces
      pieces2[ix, :, :] = data[1].positions[2].pieces
    end
  end
end

function writer2(game_chan::GameChan)
  for (st, vals) in game_chan
    println("Got $(vals[1]) $(vals[2])")
    @assert abs(mean(vals)) <= 1/ length(vals)
  end
    # Half the values should be 1s, the other half -1s. 
    # Or all zeros.
end


function runner2()
  game_chan = GameChan()
  @sync begin
    bind(game_chan, @async writer2(game_chan))
    bind(game_chan, @async ab_selfplay(game_chan))
  end
end

function runner()
  game_chan = GameChan(50)
  @sync begin
    bind(game_chan, Threads.@spawn writer(game_chan))
    for _ in 1:50
      bind(game_chan, Threads.@spawn ab_selfplay(game_chan))
    end
  end
end
