
function test()

  m = SMatrix{2,2}([1 2; 3 4])
  @infiltrate !all(nestedview(m)[1] .== [1 2])

  # Can win in 1 steps
  # (but as we don't prefer short paths, we may find a longer one)
  # println("Test 1")
  # st = State(1, SVector{2}([
  #   PlayerState(
  #     SVector{2}([4,6]),
  #     SMatrix{5,2}([6 8; 4 6; 2 1; 3 1; 4 1])),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{5,2}(Int8[collect(2:6) fill(4, 5)]))]))
  # result = simulate(st, [alpha_beta, rand_policy])
  # @infiltrate result[1] != 1

  # Can in 2 steps
  # println("\nTest 2")
  # st = State(1, SVector{2}([
  #   PlayerState(
  #     SVector{2}([4,6]),
  #     SMatrix{5,2}([5 6; 4 6; 2 1; 3 1; 4 1])),
  #   PlayerState(
  #       SVector{2}([4,4]),
  #       SMatrix{5,2}(Int8[collect(2:6) fill(4, 5)]))]))
  # result = simulate(st, [alpha_beta, rand_policy], steps=4)
  # @infiltrate result[1] != 1
end

