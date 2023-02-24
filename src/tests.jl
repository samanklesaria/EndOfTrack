
function test()

  # Can win in 1 steps
  st1 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
        
  println("AlphaBeta can see 1 step ahead")
  r1 = simulate(st1, [AlphaBeta(3), rand_policy], steps=4, log=true)
  @assert r1.winner == 1
  @assert r1.steps == 1
  
  println("MCTS can see 1 step ahead")
  r1 = simulate(st1, [MC(10), rand_policy], steps=4; log=true)
  @assert r1.winner == 1
  @assert r1.steps == 1
  
  println("CachedMinimax can see 1 steps ahead")
  result = simulate(st1, [CachedMinimax(3), rand_policy], steps=4; log=true)
  @assert result.winner == 1
  @assert r1.steps == 1
    

  # Can in 2 steps
  st2 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([5 6; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
        
  println("AlphaBeta can see 2 steps ahead")
  r2 = simulate(st2, [AlphaBeta(3), rand_policy]; log=true)
  @assert r2.winner == 1
  @assert r2.steps == 3
  
  println("MCTS can see 2 steps ahead")
  r1 = simulate(st1, [MC(50), rand_policy]; log=true)
  @assert r1.winner == 1
  @assert r1.steps == 1
  
  println("CachedMinimax can see 2 steps ahead")
  result = simulate(st2, [CachedMinimax(3), rand_policy], steps=4, log=true)
  @assert result.winner == 1
  @assert r2.steps == 3
      
  println("Terminal state test")
  unnorm_state = State(2, SVector{2}([
    PlayerState(
      SVector{2}([6,8]),
      SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  @assert is_terminal(normalized(unnorm_state)[2]) 
end

