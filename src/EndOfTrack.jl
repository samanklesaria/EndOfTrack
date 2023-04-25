module EndOfTrack
using BSON: @save, @load
using Flux, NNlib, CUDA, Distributions
using StaticArrays, Accessors, Random
using Functors, StatsBase, DataStructures
using StatsBase: mean
using Infiltrator
using ThreadPools
using Unzip

include("rules.jl")
include("util.jl")
include("searches.jl")
include("groupops.jl")
include("dists.jl")
include("noroll.jl")
include("nn.jl")
include("tests.jl")
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

mutable struct TimeTracker
  @atomic steps::Int
  @atomic time::Float64
end

TimeTracker() = TimeTracker(0, 0.0)

struct BenchPlayer{P}
  player::P
  tracker::TimeTracker
end

function (bp::BenchPlayer)(st)
  dt = @elapsed begin
    result = bp.player(st)
  end
  @atomic bp.tracker.time += dt
  @atomic bp.tracker.steps += 1
  result
end

function opponent_moved!(bp::BenchPlayer, action::Action)
  opponent_moved!(bp.player, action)
end

function bench_AB()
  ts = [TimeTracker() for i in 1:6]
  N = 20
  win_avgs = [
    begin
      results = tmap(seed->simulate(start_state,
        (BenchPlayer(AlphaBeta(i, Xoshiro(seed)), ts[i]),
         BenchPlayer(AlphaBeta(i+1, Xoshiro(seed)), ts[i+1]))), 1:N)
      mean([r.winner for r in results if !isnothing(r.winner)])
    end for i in 1:(length(ts) - 1)
  ]
  times = [p.time / p.steps for p in ts]
  times, win_avgs 
end

function do_validate(req, seed)
  try
    return validate_noroll(req, seed)
  catch exc
    return missing
  end
end

# When pretraining, we get 0.143
function bench_noroll()
  N = 20
  N_EVAL = 4
  gpus = Iterators.Stateful(sorted_gpus())
  np = NewParams(make_net())
  req = ReqChan(EVAL_BATCH_SIZE)
  for i in 1:N_EVAL
    t = @tspawnat (1 + i) begin
      device!($(popfirst!(gpus)))
      evaluator(req, np)
    end
    bind(req, t)
    errormonitor(t)
  end
  results = skipmissing(tmap(seed->do_validate(req, UInt8(seed)), 1:N))
  mean(results)
end

function test_validate(seed)
  players = (TestNoRoll(nothing; shared=false), AlphaBeta(5, Xoshiro(seed)))
  game_q(simulate(start_state, players))
end

# This gives -0.45
function bench_testnoroll()
  N = 20
  mean(skipmissing(tmap(seed->test_validate(UInt8(seed)), 1:N)))
end



end # module
