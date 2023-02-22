
function test()

  # Can win in 1 steps
  # (but as we don't prefer short paths, we may find a longer one)
  println("Test 1")
  st1 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  r1 = simulate(st1, [AlphaBeta(3), rand_policy])
  @assert r1.winner == 1

  # Can in 2 steps
  println("\nTest 2")
  st2 = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([5 6; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  r2 = simulate(st2, [AlphaBeta(3), rand_policy])
  @assert r2.winner == 1
    
  println("Terminal state test")
  unnorm_state = State(2, SVector{2}([
    PlayerState(
      SVector{2}([6,8]),
      SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  @assert is_terminal(normalized(unnorm_state)[2])
  
  println("\nTest 3")
  result = simulate(st1, [CachedMinimax(3), rand_policy], steps=50)
  @assert result.winner == 1
end

