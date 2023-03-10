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
include("max.jl")
include("tests.jl")
include("gui.jl")

function playoff(players)
  N = 50
  results = tmap(_->simulate(start_state, players()), 20, 1:N)
  println("$(sum(isnothing(r.winner) for r in results) / N) were nothing")
  win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
  println("Average winner was $win_avg") 
  # histogram([r.steps for r in results if !isnothing(r.winner)])
end

mutable struct BenchPlayer{P}
  player::P
  steps::Int
  time::Float64
end

function (bp::BenchPlayer)(st)
  bp.time += @elapsed begin
    result = bp.player(st)
  end
  bp.steps += 1
  result
end

# Priorities:
# Start by evaluating the two, as you describe in bench.
# Then, once you have a good baseline, redo NN training.
# While that's happening, try some hueristics
# After that, look at the shuffle bug.

function bench()
  abs = [AlphaBeta(i) for i in 1:6]
  mc = [classic_mcts(steps=s, rollout_len=l) for s in 10:10:100 for l in 10:10:50]
  players = [Rand() abs mc]
  # For each AlphaBeta player, keep testing it against its predecessors
  # until its predecessor always looses. Then stop. Rand is the smallest AlphaBeta
  
  # Put the MC players in a matrix. For each player, test with decreasing numbers of steps, then stop.
  # Then test with decreasing numbers of rollouts, then stop. This defines your comparison region. 
  # Test everything in the comparison region
  
  
  # A 'rollout vs steps' heatmap of computation time for MCTS
  # Also a AlphaBeta depth vs computation time plot
  # A CachedMinimax depth vs computation time plot 
  # winrate vs steps against previous steps, one line for each rollout amt
  # winrate vs rollout against previous rollout, one line for each steps
  
  # Do the same again for max_mcts.
  # And perhaps for hueristic use too. 
  
  # Then, evaluate the best in each category and put them head to head
  # We'll make a pairwise competition matrix.
  
  # How do you evaluate MCTS against AlphaBeta?
  
  # How would we compare all the different MCTS matchups?
  # Maybe we ONLY compare each to its previous two (less steps and less rollouts)
  
  
  # We would plot this by having rollout vs steps heatmap of wins
  
  # Could do the same thing again for the neural policy. 
  # Also could try simple hueristics (ball position, farthest piece position, etc)
  
  # To compare MC and AlphaBeta: WHAT?
  
  # There's also the matter of the neural policy, and hueristics.
  
  # (BenchPlayer(a), BenchPlayer(b))
end

function runner(s)
  seed = Random.rand(UInt8)
  println("Seed $seed")
  Random.seed!(seed)
  simulate(start_state, (AlphaBeta(3), max_mcts(steps=50)), steps=s).winner
end

# Figure out why permutation normalization fails

# TODO:
# Add normalization option to AlphaBeta
# Do timing run for different AlphaBeta levels, with/without normalization
# Get stock MCTS to run. Figure out why it sucks
# Review RL book


# ---
# Add an MCTS option that doesn't expand all the children


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
