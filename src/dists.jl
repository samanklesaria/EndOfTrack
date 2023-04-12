# A Normal Gamma with prior mu=0, alpha=1, beta=0.5, lambda=1
struct ValPrior <: ContinuousUnivariateDistribution
  m1::Float32 # sum of observations
  m2::Float32 # sum of squares
  n::Int # number of observations
end

ValPrior(val::Float32) = ValPrior(val, val^2, 1)

function posterior(dist::ValPrior, n, q, q2)
  ValPrior(dist.m1 + q, dist.m2 + q2, dist.n+n)
end

function Base.rand(rng::AbstractRNG, d::ValPrior)
  a = 1 + (d.n / 2)
  lam = 1 + d.n
  mu = mean(d)
  aux = mu^2 * d.n
  ns = d.m2 - aux
  b = 0.5 + 0.5 * (ns + (aux / lam))
  tau = rand(rng, Gamma(a, b))
  rand(rng, Normal(mu,  1 / tau))
end

StatsBase.mean(d::ValPrior) = d.m1 / d.n

