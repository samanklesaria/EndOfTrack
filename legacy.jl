function to_hdf5()
  open("games", "r") do reader
    h5 = h5open("gamedb.h5", "w")
    values = create_dataset(h5, "values", Float32, (787654,), chunk=(64,))
    players = create_dataset(h5, "players", Int8, (787654,), chunk=(64,))
    pieces1 = create_dataset(h5, "pieces1", Int8, (787654,2,5), chunk=(64,2,5))
    pieces2 = create_dataset(h5, "pieces2", Int8, (787654,2,5), chunk=(64,2,5))
    balls1 = create_dataset(h5, "balls1", Int8, (787654,2), chunk=(64, 2))
    balls2 = create_dataset(h5, "balls2", Int8, (787654,2), chunk=(64, 2))
    for ix in 1:787654
        data = deserialize(reader)
        values[ix] = data[2]
        players[ix] = data[1].player
        balls1[ix, :] = data[1].positions[1].ball
        balls2[ix, :] = data[1].positions[2].ball
        pieces1[ix, :, :] = data[1].positions[1].pieces
        pieces2[ix, :, :] = data[1].positions[2].pieces
    end
    close(h5)
  end
end

# TODO: re-run to_hdf5

function count_gathered_data()
  counter = 0
  open("games", "r") do io
    while !eof(io)
      try
        deserialize(io)
        counter += 1
      catch
      end
    end
  end
  counter
end


function read_gathered_data()
  data = Pair{State, Float32}[]
  open("games", "r") do io
    while !eof(io)
      try
        push!(data, deserialize(io))
      catch
      end
    end
  end
  data
end

function gather_ab_data()
  N = Threads.nthreads() - 1
  game_chan = Channel{GameResult}(N-1)
  # @sync for i in 1:2
  @threads :static for i in 1:N
      if i == 1
        ab_writer(game_chan)
      else
        ab_player(game_chan)
      end
  end
end

function ab_writer(game_chan)
  open("games", "w") do io
    for game in game_chan
      nsts, values = with_values(game)
      for (n, v) in zip(nsts, values)
        serialize(io, n => v)
      end
    end
  end
end
