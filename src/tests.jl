# TODO: NoRoll is slower than AlphaBeta.
# Profile to find out why. Is it all the dynamic dispatches?
# Is it repeated Gamma sampling?

next_player(::Nothing) = nothing

function winner_test(st, steps, winner)
  for (st, winner) in [(st, winner)]
    # (st, winner), (flip_players(st), next_player(winner)),
    # (flip_board(st), winner)]
    test_players = [
      # AlphaBeta(4), 
      TestNoRoll(nothing; st=st, shared=true)
    ]
    for p in test_players
      println("\n$(typeof(p))")
      init!(p; state=st)
      r1 = simulate(st, (p, p), steps=steps + 1, log=true)
      @assert r1.winner == winner
      if !isnothing(winner)
        @assert r1.steps == steps
      end
    end
  end
end
  
function test()
  Random.seed!(1234)

  st1 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  println("\n1 step win test")
  winner_test(st1, 1, 1)
     
  st2 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([5 6; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))
  ]))
  println("\n2 step win test")
  winner_test(st2, 3, 1)
  
  st3 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([7,6]),
      SMatrix{2,5}([3 7; 7 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}([3 4; 4 4; 5 4; 6 4; 6 6]'))
  ]))
  println("\nAnother 2 step win test")
  winner_test(st3, 3, 1)
  
  st33 = State(2, SVector{2}([
    PlayerState(
      SVector{2}([7,6]),
      SMatrix{2,5}([5 8; 7 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}([4 6; 4 4; 5 4; 6 4; 6 6]'))
  ]))
  println("\n1 step block test")
  winner_test(st33, 2, nothing)
  
  st4 = State(2, SVector{2}([
    PlayerState(
      SVector{2}([7,6]),
      SMatrix{2,5}([3 7; 7 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}([3 4; 4 4; 5 4; 6 4; 6 6]'))
  ]))
  println("\n2 step block test")
  winner_test(st4, 0, nothing)
end
