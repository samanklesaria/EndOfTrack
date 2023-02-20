
function test()

  # Can win in 1 steps
  # (but as we don't prefer short paths, we may find a longer one)
  println("Test 1")
  st = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([6 8; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  result = simulate(st, [AlphaBeta(2), rand_policy])
  @assert result[1] == 1

  # Can in 2 steps
  println("\nTest 2")
  st = State(1, SVector{2}([
    PlayerState(
      SVector{2}([4,6]),
      SMatrix{2,5}([5 6; 4 6; 2 1; 3 1; 4 1]')),
    PlayerState(
        SVector{2}([4,4]),
        SMatrix{2,5}(Int8[collect(2:6) fill(4, 5)]'))]))
  result = simulate(st, [AlphaBeta(3), rand_policy])
  @assert result[1] == 1
  
  # result = simulate(st, [CachedMinimax(3), rand_policy])
  # @assert result[1] == 1
end

