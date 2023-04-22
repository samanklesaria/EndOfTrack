module EndOfTrack
using BSON: @save, @load
using Flux, NNlib, CUDA, Distributions
using StaticArrays, Accessors, Random
using Functors, StatsBase, DataStructures
using StatsBase: mean
using Infiltrator
using HDF5
# using ThreadTools
using Unzip

include("rules.jl")
include("util.jl")
include("searches.jl")
include("groupops.jl")
include("dists.jl")
include("noroll.jl")
include("nn.jl")
# include("tests.jl")
include("static.jl")
# include("gui.jl")

# function playoff(players)
#   N = 20
#   results = tmap(_->simulate(start_state, players()), 1:N)
#   println("$(sum(isnothing(r.winner) for r in results) / N) were nothing")
#   win_avg = mean([r.winner for r in results if !isnothing(r.winner)])
#   println("Average winner was $win_avg") 
#   # histogram([r.steps for r in results if !isnothing(r.winner)])
# end

# mutable struct TimeTracker
#   @atomic steps::Int
#   @atomic time::Float64
# end

# TimeTracker() = TimeTracker(0, 0.0)

# struct BenchPlayer{P}
#   player::P
#   tracker::TimeTracker
# end

# function (bp::BenchPlayer)(st)
#   dt = @elapsed begin
#     result = bp.player(st)
#   end
#   @atomic bp.tracker.time += dt
#   @atomic bp.tracker.steps += 1
#   result
# end

# Plots:
# Incremental win rate vs depth
# Computation time per move vs depth
# function bench_AB()
#   ts = [TimeTracker() for i in 1:6]
#   N = 20
#   win_avgs = [
#     begin
#       results = tmap(_->simulate(start_state,
#         (BenchPlayer(AlphaBeta(i), ts[i]), BenchPlayer(AlphaBeta(i+1), ts[i+1]))), 1:N)
#       mean([r.winner for r in results if !isnothing(r.winner)])
#     end for i in 1:(length(ts) - 1)
#   ]
#   times = [p.time / p.steps for p in ts]
#   times, win_avgs 
# end

# Data for computation time per move vs depth
# function bench_minimax()
#   mm = [BenchPlayer(CachedMinimax(i), TimeTracker()) for i in 1:5]
#   for m in mm
#     simulate(start_state, (m, Rand()))
#   end
#   [p.tracker.time / p.tracker.steps for p in mm]
# end

# function mini_mcts_bench()
#   N = 20
#   ss = 10:10:60
#   mc = [TimeTracker() for s in ss]
#   winners = [ begin
#     println("Testing $(ss[s])")
#     results = tmap(_-> simulate(start_state,
#       (BenchPlayer(max_mcts(steps=ss[s], rollout_len=10), mc[s]),
#       (BenchPlayer(max_mcts(steps=ss[s+1], rollout_len=10), mc[s+1])))), Threads.nthreads(), 1:N)
#     mean([r.winner for r in results if !isnothing(r.winner)])
#     end for s in 1:(length(ss) - 1)
#   ]
#   times = map(mc) do p
#     p.time / p.steps
#   end
#   winners, times
# end

# # Plots:
# # Win rate vs rollout_len for fixed steps 
# # Win rate vs steps for fixed rollout len
# # Computation time for grid of steps, rollout_lens
# function bench_mcts(f)
#   N = 40
#   ls = 10:10:20
#   ss = 10:10:60
#   mc = [TimeTracker() for s in ss for l in ls]
#   mcmat = reshape(mc, length(ls), length(ss))
#   s_comparison  = [
#     [
#       begin
#         results = tmap(_->simulate(start_state,
#           (BenchPlayer(f(steps=ss[s], rollout_len=ls[l]), mcmat[l, s]),
#            BenchPlayer(f(steps=ss[s+1], rollout_len=ls[l]), mcmat[l, s+1]))), Threads.nthreads(), 1:N)
#         mean([r.winner for r in results if !isnothing(r.winner)])
#       end for s in 1:length(ss[1:end-1])
#     ]
#   for l in 1:length(ls)]
#   l_comparison  = [
#     [
#       begin
#         results = tmap(_->simulate(start_state,
#           (BenchPlayer(f(steps=ss[s], rollout_len=ls[l]), mcmat[l, s]),
#            BenchPlayer(f(steps=ss[s], rollout_len=ls[l+1]), mcmat[l+1, s]))), Threads.nthreads(), 1:N)
#         mean([r.winner for r in results if !isnothing(r.winner)])
#       end for l in 1:length(ls[1:end-1])
#     ]
#   for s in 1:length(ss)]
#   times = map(mcmat) do p
#     p.time / p.steps
#   end
#   times, s_comparison, l_comparison 
# end

# function winners_circle()
#   N = 20
#   policies = [()->Rand(), ()-> max_mcts(steps=50, rollout_len=20), ()->classic_mcts(steps=50, rollout_len=20), ()->AlphaBeta(5)]
#   results = []
#   for p1f in policies
#     for p2f in policies
#       if p1f !== p2f
#         p1 = p1f()
#         p2 = p2f()
#         push!(results, (p1,p2)=> tmap(_->simulate(start_state, (p1, p2)), 1:N))
#       end
#     end
#   end
# end

# function runner()
#   N = 20
#   ab = TimeTracker()
#   results = tmap(_->simulate(start_state,
#     (BenchPlayer(CachedMinimax(3), ab), Rand())), 1:N)
#   mean([r.winner for r in results if !isnothing(r.winner)])
# end

# function mcts_match()
#   N = 20
#   results = tmap(_->simulate(start_state,
#     (max_mcts(steps=30, rollout_len=20),
#     max_mcts(steps=50, rollout_len=20))), 1:N)
#   mean([r.winner for r in results if !isnothing(r.winner)])
# end

# function mcts_sanity()
#     cfg, ps = make_small_net()
#     val = NeuralValue{Val{false}}(cfg.net, ps, cfg.st)
#     neural_player = NeuralPlayer(val, 50f0)
#     rollout_players = (neural_player, neural_player)
#     mc = MaxMCTS(players=rollout_players, steps=20,
#       rollout_len=10, estimator=neural_player)
#     players = (mc, mc)
#     simulate(start_state, players; track=true) 
# end

# function ab_sanity()
#     cfg, ps = make_small_net()
#     neural_val = NeuralValue{Val{false}}(cfg.net, ps, cfg.st)
#     ab = AlphaBeta(4, neural_val)
#     players = (ab, Rand())
#     simulate(start_state, players; steps=2) 
# end

# function both_mcts_match()
#   N = 20
#   results = tmap(_->simulate(start_state,
#     (classic_mcts(steps=50, rollout_len=20),
#     max_mcts(steps=50, rollout_len=20))), 1:N)
#   mean([r.winner for r in results if !isnothing(r.winner)])
# end

end # module
