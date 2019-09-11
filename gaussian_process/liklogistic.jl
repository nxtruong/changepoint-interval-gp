# Logistic-like likelihood

using StatsFuns

"""
    Logistic-like Likelihood for Intervals

Given an interval `[l,h]` the likelihood is calculated by the logistic function:
    `p(y = [l,h] | f) = logistic(a*(x-l)) - logistic(a*(x-h))`
where `a` is a parameter, which is often large.
Special cases can easily be derived for when `l = -∞` and when `h = ∞` but at
least one of them must be finite. When l is infinite, logistic(l) = 1;
when h is infinite, logistic(h) = 0.

`a` is chosen so that:
- At `f=l+x` and `f=h-x`, `p` is large enough, close to 1.
- And of course `x` should be small enough, much less than `(h-l)/2` or the radius
  of the interval.

We can show that `a = log(p/(1-p))/x` where `x <= (h-l)/2` (should be much smaller
than that) and `p` is for instance 0.95 or 0.99.
"""
struct IntervalLogisticLikelihood
    a
end # struct

"""
    likelihood(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f::T)

Calculate the likelihood of `f` in the interval `y`.
"""
@inline function likelihood(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f) where{T}
    Fl = isinf(y.l) ? 1.0 : logistic(lik.a*(f-y.l))
    Fh = isinf(y.h) ? 0.0 : logistic(lik.a*(f-y.h))
    Fl - Fh
end # function


"""
    loglikelihood(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f::T)

Returns `LL` - the log likelihood of likInterval at `f`, `y`.
"""
@inline function loglikelihood(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f) where{T}
    if isinf(y.l)
        return -log1pexp(lik.a*(f-y.h))
    elseif isinf(y.h)
        return -log1pexp(lik.a*(y.l-f))
    else
        sa = lik.a*(f-y.l)
        r1 = sa <= -35 ? sa : -log1p(exp(-sa))

        sb = lik.a*(f-y.h)
        r2 = sb >= 35 ? -sb : -log1p(exp(sb))
        return log1mexp(sb-sa) + r1 + r2
    end
end


"""
    loglikelihood_grad(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f::T; calcGGLL=true)

Returns tuple `(GLL, GGLL)` where
`GLL` is the derivative of the log likelihood at `f`, `y` w.r.t. `f`;
`GGLL` is its second derivative (Hessian) w.r.t. `f`;

Calculation of GGLL can be turned off by the named arguments.
"""
function loglikelihood_grad(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f; calcGGLL=true) where{T}
    dlpl = isinf(y.l) ? 0.0 : logistic(lik.a*(y.l-f))
    dlph = isinf(y.h) ? 0.0 : logistic(lik.a*(f-y.h))

    GLL = lik.a * (dlpl - dlph)

    if !calcGGLL
        return GLL
    end

    GGLL = -lik.a^2 * (dlpl + dlph - dlpl^2 - dlph^2)

    return (GLL, GGLL)
end # function


"""
    loglikelihood_grad3(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f::T)

Calculates the derivative of GGLL (see `loglikelihood_grad`) with respect to `f`.

Returns G3latent.
- `G3latent` is `∇³log p(y|f)` (it can be called `d3lp` similarly to `dlp`, `d2lp`)
"""
function loglikelihood_grad3(lik::IntervalLogisticLikelihood, y::IntervalType{T}, f::T) where{T}
    a3 = lik.a*lik.a*lik.a
    if isinf(y.l)
        L_h = logistic(lik.a*(f-y.h))  # L(h)
        G3latent = a3 * L_h * (L_h-1) * (1-2L_h)
    elseif isinf(y.h)
        L_l = logistic(lik.a*(f-y.l))  # L(l)
        G3latent = a3 * L_l * (L_l-1) * (1-2L_l)
    else
        L_h = logistic(lik.a*(f-y.h))  # L(h)
        L_l = logistic(lik.a*(f-y.l))  # L(l)
        G3latent = a3 * (L_l * (L_l-1) * (1-2L_l) + L_h * (L_h-1) * (1-2L_h))
    end

    return G3latent
end # function

"""
    initialvec(lik::IntervalLogisticLikelihood, Y::AbstractVector{IntervalType{T}})

Returns an initial vector for a list of interval observations, to be used for laplace_mode.
"""
@inline function initialvec(lik::IntervalLogisticLikelihood, Y::AbstractVector{IntervalType{T}}, K) where{T}
    # Find good f
    p = 0.98
    f = @inbounds [isfinite(y.l) ? min(y.l + log(p/(1-p))/lik.a, (y.l+y.h)/2) : (isfinite(y.h) ? y.h - log(p/(1-p))/lik.a : 0.0) for y in Y]

    # Calculate alpha = K^(-1)*f
    return calc_α_from_f(f, lik, Y, K)
end # function
