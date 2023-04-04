# TODO: why is NoRoll soooo slow?

function winner_test(st, steps, winner)
  test_players = [
    # AlphaBeta(4), 
    NoRoll(estimator=nothing),
    # CachedMinimax(4)
  ]
  for p in test_players
    println("\n$(typeof(p))")
    r1 = simulate(st, (p, p), steps=steps + 1, log=true)
    @infiltrate r1.winner != winner
    if !isnothing(winner)
      @infiltrate r1.steps != steps
    end
    
    # println("\nIn reverse")
    # nst = State(next_player(st.player), reverse(fmap(flip_pos_hor, st).positions))
    # r1 = simulate(nst, (p, p), steps=steps + 1, log=true)
    # if !isnothing(winner)
    #   winner2 = next_player(winner)
    #   @infiltrate r1.winner != winner2
    # end
    # @infiltrate r1.steps != steps
  end
end

# Maybe we want forward edges instead of back edges. 
# Think through this first!
  
function test()
  Random.seed!(1234)

  # st1 = State(1, SVector{2}([
  #   PlayerState(
  #     SVector{2}([4,6]),
  #     SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  # println("\n1 step win test")
  # winner_test(st1, 1, 1)
     
  # st2 = State(1, SVector{2}([
  #   PlayerState(
  #     SVector{2}([4,6]),
  #     SMatrix{2,5}([5 6; 4 6; 2 1; 3 1; 4 1]')),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))
  # ]))
  # println("\n2 step win test")
  # winner_test(st2, 3, 1)
  
  # st3 = State(1, SVector{2}([
  #   PlayerState(
  #     SVector{2}([7,6]),
  #     SMatrix{2,5}([3 7; 7 6; 2 1; 3 1; 4 1]')),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{2,5}([3 4; 4 4; 5 4; 6 4; 6 6]'))
  # ]))
  # println("\nAnother 2 step win test")
  # winner_test(st3, 3, 1)
  
  
  # st3 = State(2, SVector{2}([
  #   PlayerState(
  #     SVector{2}([7,6]),
  #     SMatrix{2,5}([5 8; 7 6; 2 1; 3 1; 4 1]')),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{2,5}([4 6; 4 4; 5 4; 6 4; 6 6]'))
  # ]))
  # println("\n1 step block test")
  # winner_test(st3, 0, nothing)
  
  st4 = State(2, SVector{2}([
    PlayerState(
      SVector{2}([7,6]),
      SMatrix{2,5}([3 7; 7 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}([3 4; 4 4; 5 4; 6 4; 6 6]'))
  ]))
  println("\n2 step block test")
  winner_test(st4, 4, nothing)
        
  # println("\nTerminal state test")
  # unnorm_state = State(2, SVector{2}([
  #   PlayerState(
  #     SVector{2}([6,8]),
  #     SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  # @assert is_terminal(normalized(unnorm_state)[2]) 
end

# badboy = MCTS.State(1, MCTS.PlayerState[MCTS.PlayerState(Int8[4, 1], Int8[4 3 4 5 6; 6 1 1 1 1]), MCTS.PlayerState(Int8[2, 8], Int8[1 3 5 3 2; 1 5 5 7 8])])
# [0.0] 1 moves from Int8[4, 1] to Int8[2, 3]

# [0.0] 1 moves from Int8[4, 5] to Int8[7, 3]
# problem_state = MCTS.State(1, MCTS.PlayerState[MCTS.PlayerState(Int8[4, 5], Int8[5 4 4 6 4; 2 5 1 1 4]), MCTS.PlayerState(Int8[5, 4], Int8[5 5 1 3 6; 4 6 7 5 8])])
