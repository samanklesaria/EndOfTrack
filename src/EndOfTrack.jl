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
include("noroll.jl")
include("nn.jl")
include("tests.jl")
include("static.jl")
# include("gui.jl")
# include("cudasim.jl")

# For Thompson Sampling:
# include("dists.jl")
# include("thompson.jl")

# For State Normalization
# include("normalization.jl")
# include("dag.jl")

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
        (BenchPlayer(AlphaBeta(i), ts[i]),
         BenchPlayer(AlphaBeta(i+1), ts[i+1]))), 1:N)
      mean([r.winner for r in results if !isnothing(r.winner)])
    end for i in 1:(length(ts) - 1)
  ]
  times = [p.time / p.steps for p in ts]
  times, win_avgs 
end

function bench_noroll()
  N = 40
  N_EVAL = 4
  gpus = Iterators.Stateful(sorted_gpus())
  np = [NewParams(make_resnet(;where="checkpoint2.bson")) for _ in 1:N_EVAL]
  req = ReqChan(EVAL_BATCH_SIZE)
  for i in 1:N_EVAL
    t = @tspawnat (1 + i) begin
      device!($(popfirst!(gpus)))
      evaluator(req, $(np[i]))
    end
    bind(req, t)
    errormonitor(t)
  end
  results = tmap(seed->validate_noroll(req), 1:N)
  mean(results)
end

function test_validate()
  players = (TestNoRoll(nothing; shared=false), AlphaBeta(5))
  game_q(simulate(start_state, players))
end

function initial_q_vals(;st=start_state)
  gpus = Iterators.Stateful(sorted_gpus())
  net = make_resnet(;where="checkpoint2.bson")
  acts = actions(st)
  next_sts = next_state.(Ref(st), acts)
  _, nst = unzip(normalize_player.(next_sts))
  batch = gpu(cat4(as_pic.(nst)))
  values = logit_to_val.(net(batch))
  result = (acts, cpu(values))
  @save "qvals.bson" result
end

function bench_testnoroll()
  N = 50
  mean(tmap(_->test_validate(), 1:N))
end

end # module
