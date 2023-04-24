# A Normal Gamma with prior mu=0, alpha=0.5, beta=0.5, lambda=1
struct ValPrior <: ContinuousUnivariateDistribution
  m1::Float32 # sum of observations
  m2::Float32 # sum of squares
  n::Int # number of observations
end

ValPrior(val::Float32) = ValPrior(val, val^2, 1)

function posterior(dist::ValPrior, n, q, q2)
  ValPrior(discount * dist.m1 + q, discount * dist.m2 + q2, dist.n+n)
end

function posterior(dist::Dirac, n, q, q2)
  error("This should never be called")
end

function Base.rand(rng::AbstractRNG, d::ValPrior)
  true_n = (1 - discount ^ d.n) / (1 - discount)
  lam = true_n + 1
  a = lam / 2
  mu = d.m1 / true_n
  aux = mu^2 * true_n
  ns = d.m2 - aux
  b = 0.5 + 0.5 * (ns + (aux / lam))
  tau = rand(rng, Gamma(a, 1/b))
  rand(rng, truncated(Normal(mu,  sqrt(1 / tau)), -1, 1))
end

StatsBase.mean(d::ValPrior) = d.m1 / (d.n + 1)

function StatsBase.std(d::ValPrior)
  lam = d.n + 1
  a = lam / 2
  mu = d.m1 / d.n
  aux = mu^2 * d.n
  ns = d.m2 - aux
  b = 0.5 + 0.5 * (ns + (aux / lam))
  std(truncated(Normal(0, sqrt(b / a)), -1, 1))
end

upperbound(d::ValPrior) = mean(d) + 2*std(d)
upperbound(d::Dirac) = mean(d)
