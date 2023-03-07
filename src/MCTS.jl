module MCTS
using BSON: @save, @load
using Lux, NNlib, Zygote, Optimisers
using StaticArrays, Accessors, Random, ArraysOfArrays, Unrolled
using Functors, StatsBase, DataStructures
using StatsBase: mean
using Infiltrator 
using VisdomLog
using ThreadTools

include("rules.jl")
include("util.jl")
include("searches.jl")
include("groupops.jl")
include("nn.jl")
include("classic.jl")
include("greedy_mcts.jl")
include("tests.jl")

function playoff(players)
  N = 50
  results = tmap(_->simulate(start_state, players()), 20, 1:N)
  println("$(sum(isnothing(r.winner) for r in results) / N) were nothing")
  win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
  println("Average winner was $win_avg") 
  # histogram([r.steps for r in results if !isnothing(r.winner)])
end

# The rollout should use an upper bound of the true value.

# Potential things:
# The GC could prune earler, by removing everything not touched by the chosen action
# We could only update backwards on the actual path taken rather than all parents. 
# Or, a middle ground: only keep around the past k parents. 
# Could also not do state sharing.
# Could expand a subset of actions rather than all actions when seeing a node. Use a pseudocount
# Could do double Q learning rather than Q learning

# Hueristics:
# Position of the ball? Highest player position?

# Optimizations:
# Encode and use bitvec instead of Dict for ball passing
# Disable bounds checks
# Do stuff in parallel

# Fun:
# Add GUI for human player 

end # module MCTS
