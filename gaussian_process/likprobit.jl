# Probit-like likelihood

"""
    Probit-like Likelihood for Intervals

Given an interval `[l,h]` the likelihood is essentially the probability of the
latent variable `f` in that interval.
    `p(y = [l,h] | f) = normcdf(h-α-f) - normcdf(l+α-f)`
where `normcdf` is of a Normal distribution with zero mean and stdvar σ.
Special cases can easily be derived for when `l = -∞` and when `h = ∞` but at
least one of them must be finite.

σ and α are chosen so that:
- At `f=l` and `f=h`, `p` is small, close to 0.
- At `f=l+x` and `f=h-x`, `p` is large enough, close to 1.
- And of course `x` should be small enough, much less than `(h-l)/2` or the radius
  of the interval.

Generally we choose `α = k σ` where `k ≥ 2` and then `x = α + 2σ = (k+2)σ`.
So we should have σ much less than `(h-l)/2/(k+2)`.  If we choose `k=2` then σ
should be much less than `(h-l)/8`.
"""
struct IntervalProbitLikelihood
    σ
    α
    d   # The Normal distribution N(0,σ)
    σ2  # square of σ
    iσ2 # 1/σ2
    IntervalProbitLikelihood(σ) = new(σ, 2σ, Normal(0,σ), σ*σ, 1/(σ*σ))
end # struct

"""
    likelihood(lik::IntervalProbitLikelihood, y::IntervalType{T}, f::T)

Calculate the likelihood of `f` in the interval `y`.
"""
@inline function likelihood(lik::IntervalProbitLikelihood, y::IntervalType{T}, f) where{T}
    cdfl = isinf(y.l) ? 0.0 : cdf(lik.d, y.l + lik.α - f)
    cdfh = isinf(y.h) ? 1.0 : cdf(lik.d, y.h - lik.α - f)
    cdfh - cdfl
end # function


"""
    loglikelihood(lik::IntervalProbitLikelihood, y::IntervalType{T}, f::T)

Returns `LL` - the log likelihood of likInterval at `f`, `y`.
"""
@inline function loglikelihood(lik::IntervalProbitLikelihood, y::IntervalType{T}, f) where{T}
    if isinf(y.l)
        # likelihood = Phi(h) so we can use logcdf
        return logcdf(lik.d, y.h - lik.α -f)
    elseif isinf(y.h)
        # likelihood = 1 - Phi(l-f) = Phi(f-l)
        return logcdf(lik.d, f - y.l - lik.α)
    else
        # This is tricky, especially when l and h are close, resulting in likelihood = 0
        F = cdf(lik.d, y.h - lik.α - f) - cdf(lik.d, y.l + lik.α - f)
        if iszero(F)
            # likelihood is the area below the normal pdf from l+α to h-α
            # We approximate it by (pdf(h-α) + pdf(l+α))*(h-l-2α)/2
            # And take its log
            return log(pdf(lik.d, y.h-lik.α-f) + pdf(lik.d, y.l+lik.α-f)) + log(y.h-y.l-2lik.α) - log(2)
        else
            return log(F)
        end
    end
end


"""
    loglikelihood_grad(lik::IntervalProbitLikelihood, y::IntervalType{T}, f::T, logF; calcGGLL=true)

Returns tuple `(GLL, GGLL, PDF_l, PDF_h)` where
`GLL` is the derivative of the log likelihood at `f`, `y` w.r.t. `f`;
`GGLL` is its second derivative (Hessian) w.r.t. `f`;
PDF_l` is `N(l+α; f, σ)`; `PDF_h` is similar but for `h-α`.

Calculation of GGLL can be turned off by the named arguments, in which case
`PDF_l` and `PDF_h` are not returned, only `GLL`.

logF is loglikelihood(y, f).
"""
function loglikelihood_grad(lik::IntervalProbitLikelihood, y::IntervalType{T}, f, logF; calcGGLL=true) where{T}
    PDF_l = isinf(y.l) ? 0.0 : pdf(lik.d, y.l + lik.α - f)
    PDF_h = isinf(y.h) ? 0.0 : pdf(lik.d, y.h - lik.α - f)

    if logF < -700
        # Very small F, can result in Inf, so we will calculate */F differently
        divbyF = x -> iszero(x) ? x : (exp(log(abs(x))-logF) * sign(x))
    else
        explogF = exp(-logF)
        divbyF = x -> x*explogF
    end

    if isinf(y.l)
        # loglikelihood is logcdf(h-α-f), and we can calculate
        # GLL = -1/σ * pdf(h-α-f) / cdf(h-α-f)
        # Then expand pdf(h-α-f) and use cdf(h-α-f) = exp(logcdf(h-α-f)) = exp(logF)
        z = (y.h-lik.α-f)/lik.σ
        GLL = -exp(-z*z/2-logF)/lik.σ/sqrt(2π)
    elseif isinf(y.h)
        # Similarly, loglikelihood is logcdf(f-l-α), then use similar equation as above
        z = (f-y.l-lik.α)/lik.σ
        GLL = exp(-z*z/2-logF)/lik.σ/sqrt(2π)
    else
        # GLL = (PDF_l - PDF_h) / F
        GLL = divbyF(PDF_l - PDF_h)
    end

    if !calcGGLL
        return GLL
    end

    GGLL_l = isinf(y.l) ? 0.0 : (y.l+lik.α-f)*PDF_l
    GGLL_h = isinf(y.h) ? 0.0 : (y.h-lik.α-f)*PDF_h

    GGLL = divbyF((GGLL_l - GGLL_h)/(lik.σ*lik.σ)) - GLL*GLL

    return (GLL, GGLL, PDF_l, PDF_h)
end # function

@inline function loglikelihood_grad(lik::IntervalProbitLikelihood, y::IntervalType{T}, f; calcGGLL=true) where{T}
    return loglikelihood_grad(lik, y, f, loglikelihood(lik, y, f); calcGGLL=calcGGLL)
end


"""
    loglikelihood_grad3(lik::IntervalProbitLikelihood, y::IntervalType{T}, f::T, logF)

Calculates the derivative of GGLL (see `loglikelihood_grad`) with respect to `f`.

Returns G3latent.
- `G3latent` is `∇³log p(y|f)` (it can be called `d3lp` similarly to `dlp`, `d2lp`)
"""
function loglikelihood_grad3(lik::IntervalProbitLikelihood, y::IntervalType{T}, f::T, logF=nothing) where{T}
    if logF == nothing
        logF = loglikelihood(lik, y, f)
    end
    if logF < -700
        # Very small F, can result in Inf, so we will calculate */F differently
        divbyF = x -> iszero(x) ? x : (exp(log(abs(x))-logF) * sign(x))
    else
        explogF = exp(-logF)
        divbyF = x -> x*explogF
    end

    GLL, GGLL, PDF_l, PDF_h = loglikelihood_grad(lik, y, f, logF)
    G = GGLL + GLL*GLL

    if isinf(y.l)
        PDF_l2 = 0.0
        PDF_l3 = 0.0
    else
        z = y.l+lik.α-f
        PDF_l2 = z*z*PDF_l
        PDF_l3 = z*PDF_l2
    end

    if isinf(y.h)
        PDF_h2 = 0.0
        PDF_h3 = 0.0
    else
        z = y.h-lik.α-f
        PDF_h2 = z*z*PDF_h
        PDF_h3 = z*PDF_h2
    end

    alpha = PDF_l2 - PDF_h2
    beta = PDF_l3 - PDF_h3

    G3latent = divbyF(alpha/(lik.σ2*lik.σ2)) - GLL*(lik.iσ2 + G + 2GGLL)

    # G2lik = divbyF(alpha*lik.iσ2/lik.σ) - GLL*lik.σ*(lik.iσ2 + G)

    # G3lik = ((beta*iσ2 - 2*alpha*GLL)*iσ2/F - (G*(1-σ2*G) + 2*GGLL*(1+σ2*G))) / σ
    # G3lik = (divbyF(beta*lik.iσ2*lik.iσ2) - G*(3 + lik.σ2*G)) / lik.σ - 2*GLL*G2lik

    return G3latent
end # function

"""
    initialvec(lik::IntervalProbitLikelihood, Y::AbstractVector{IntervalType{T}})

Returns an initial vector for a list of interval observations, to be used for laplace_mode.
"""
@inline function initialvec(lik::IntervalProbitLikelihood, Y::AbstractVector{IntervalType{T}}, K) where{T}
    # Find good f
    f = @inbounds [isfinite(y.l) ? y.l+lik.α+2lik.σ : (isfinite(y.h) ? y.h-lik.α-2lik.σ : 0.0) for y in Y]

    # Calculate alpha = K^(-1)*f
    return calc_α_from_f(f, lik, Y, K)
end # function
