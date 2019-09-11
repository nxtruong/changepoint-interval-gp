# Main module for GP with interval observations
# (C) 2019 by Truong X. Nghiem (truong.nghiem@gmail.com)

module GPInterval

using Distributions
using LinearAlgebra
import Base: show
using NLsolve
using Optim
import LineSearches
using GaussianProcesses

export IntervalType, IntervalProbitLikelihood, IntervalLogisticLikelihood
export GPInt, train!, infLaplace, predict

"Interval type of the form [l, h]."
IntervalType{T} = NamedTuple{(:l, :h),Tuple{T,T}}

function Base.show(io::IO, x::IntervalType)
    print(io, isinf(x.l) ? "(-∞," : "[$(x.l),", isinf(x.h) ? "+∞)" : "$(x.h)]")
end

"""
    GPInterval: GP with interval observations

    Inputs are real vectors in `R^n`, but outputs are intervals `[l, h], each of
    which contains the noisy output correponding to the input.
"""
mutable struct GPInt
    """Covariance function from GaussianProcesses.jl"""
    cov

    """The likelihood object."""
    lik

    """Inputs of training data in columns of X."""
    X

    """Vector of interval observations."""
    Y

    """Input dimension."""
    dim

    """Laplace mode f̂."""
    f̂

    """α vector = K^(-1)*f̂ where f̂ is the Laplace mode; used for prediction."""
    α

    """L = cholesky(B) where B = I + sqrt(W)*K*sqrt(W); used for prediction."""
    L

    """Wsqrt = sqrt(W) where W = ∇^2 log(p(y|f̂)); used for prediction."""
    Wsqrt

    """Covariance data structure: only depends on X, not on the hyperparameters."""
    covdata

    """Function to calculate the derivative of covariance matrix w.r.t a covariance hyperparameter p."""
    fdcov!

    """Log marginal likelihood."""
    lml

    """
        GPInt(lik, cov, X, Y)

        Initialize a GP object for Interval observations.
        - lik: an interval likelihood instance (e.g., `IntervalProbitLikelihood`)
        - cov: a GaussianProcesses.jl kernel
    """
    function GPInt(lik, cov::GaussianProcesses.Kernel, X, Y::AbstractVector{IntervalType{T}}) where{T}
        N = size(X, 2)
        @assert N == length(Y)

        α = Vector{Float64}(undef, N)
        L = nothing
        Wsqrt = Vector{Float64}(undef, N)

        f̂ = nothing
        covdata = GaussianProcesses.KernelData(cov, X, X)

        gp = new(
            cov, lik, X, Y, size(X,1), f̂, α, L, Wsqrt,
            covdata,
            (C,p) -> GaussianProcesses.grad_slice!(C, cov, X, covdata, p),
            0.0)

        update_params!(gp)
    end
end # struct


"""
    predict(gp::GPInt, x::AbstractMatrix)

Return posterior mean and variance of the Gaussian Process `gp` at specfic points which are
given as columns of matrix `X`.
"""
function predict(gp::GPInt, x::AbstractMatrix)
    size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))

    ## Calculate prediction for each point independently
    μ = Array{eltype(x)}(undef, size(x,2))
    σ2 = similar(μ)
    kx = Matrix{eltype(x)}(undef, size(gp.X,2), 1)
    for k in 1:size(x,2)
        xk = x[:,k:k]
        μ[k] = dot(GaussianProcesses.cov!(kx, gp.cov, gp.X, xk), gp.α)
        v = gp.L.L \ (gp.Wsqrt .* kx)
        σ2[k] = max(GaussianProcesses.cov(gp.cov, xk, xk)[1,1] - dot(v, v), 0.0)
    end
    return μ, σ2
end # function


"""
    update_params!(gp::GPInt, covhyps, K=nothing)

Update the hyperparameters of a GP.  Most hyperparameters are given in log.
- covhyps: log of hyperparameters of kernel
- K can be provided as pre-allocated covariance matrix
"""
function update_params!(gp::GPInt, covhyps=nothing, K=nothing)
    if covhyps != nothing
        set_params!(gp.cov, covhyps)
    end

    # Perform Laplace inference again to update the precomputed values
    if K == nothing
        N = size(gp.X,2)
        K = Matrix{Float64}(undef, N, N)
    end
    GaussianProcesses.cov!(K, gp.cov, gp.X, gp.covdata)
    nlZ, _, gp.f̂, gp.α, gp.L, gp.Wsqrt = infLaplace(gp.lik, GaussianProcesses.num_params(gp.cov), gp.fdcov!, K, gp.f̂, gp.Y)
    gp.lml = -nlZ
    gp
end # function


function Base.show(io::IO, gp::GPInt)
    println(io, "Gaussian Process with Interval observations:")
    println(io, "- Kernel: ", gp.cov)
    println(io, "- Log marginal likelihood: ", gp.lml)
    println(io, "- Laplace mode: ", gp.f̂)
end


"""
    train!(gp::GPInt; kwargs...)

Train a GPInt model.  Optimization method and Optim.jl's parameters can be given.
Default method is ConjugateGradient.
"""
function train!(gp::GPInt; method = ConjugateGradient(linesearch=LineSearches.HagerZhang()), kwargs...)
    # Pre-allocate space for covariance matrix
    N = size(gp.X,2)
    K = Matrix{Float64}(undef, N, N)

    # Create functions for optimization
    function fg!(F,G,x)
        # Set the parameters of the GP
        set_params!(gp.cov, x)

        GaussianProcesses.cov!(K, gp.cov, gp.X, gp.covdata)
        f̂0 = nothing
        # f̂0 = gp.f̂

        if G != nothing
            nlZ, dnlZcov, gp.f̂ = infLaplace(gp.lik, GaussianProcesses.num_params(gp.cov), gp.fdcov!, K, f̂0, gp.Y)
            G .= dnlZcov
        elseif F != nothing
            nlZ, gp.f̂ = infLaplace(gp.lik, GaussianProcesses.num_params(gp.cov), gp.fdcov!, K, f̂0, gp.Y, false)
        end
        if F != nothing
            F = nlZ
            return nlZ
        end
    end

    # Get the current parameters (initial)
    x0 = GaussianProcesses.get_params(gp.cov)

    # Call the optimizer
    results = optimize(Optim.only_fg!(fg!), x0; method=method, kwargs...)
    x = Optim.minimizer(results)

    update_params!(gp, x, K)
    gp
end # function


"""
    laplace_mode(lik, f̂, K, Y::IntervalType{T})

Find the posterior mode of the GP with interval observations, for Laplace method.
Similar to Algorithm 3.1 in GPML book (page 46).
Current implementation: solves for fixed point `F(f̂) = f̂ - K (∇log p(y | f̂)) = 0`
Future implementation will use IRLS algorithm as implemented in
GPML toolbox's `infLaplace.m`.

It's very important to choose good initial values of `f̂`. In particular, the
initial values should be inside the observed intervals to avoid the case when
the likelihood is zero, causing numerical issues (log likelihood becomes -Inf,
Jacobian and Hessian become NaN or Inf).

Inputs:
- `lik`: the interval likelihood instance
- `f̂`: initial values for `f̂`
- `K`: covariance matrix
- `Y`: vector of interval observations

Outputs:
- `f̂`: modal value
- `sumlp`: sum of approximated `log p(y | f)` for all y in Y
- `dlp`: approximated `∇ log p(y | f)`
- `d2lp`: approximated `∇ ∇ log p(y | f)`
"""
function laplace_mode(lik, f0, K, Y::AbstractVector{IntervalType{T}}; maxiter=100, Wmin=0.0, tol=1e-6) where{T}
    if f0 == nothing
        f0 = initialvec(lik, Y)
    end
    N = length(Y)
    Wmin = -Wmin

    # Pre-allocate dlp (derivative of lp), d2lp
    dlp = Vector{Float64}(undef, N)
    d2lp = Vector{Float64}(undef, N)

    # Function that calculates `F(f̂)` and its Jacobian
    function Ffj!(F, J, x)
        calcJ = !(J == nothing)
        if calcJ
            @inbounds @simd for i = 1:N
                dlp[i], d2lp[i] = loglikelihood_grad(lik, Y[i], x[i])
                d2lp[i] = min(Wmin, d2lp[i])
            end
        else
            @inbounds @simd for i = 1:N
                dlp[i] = loglikelihood_grad(lik, Y[i], x[i]; calcGGLL=false)
            end
        end

        if !(F == nothing)
            F .= (x - K*dlp)
            if any(isinf.(F))
                print("non-finite F: $F, at x = $x, dlp = $dlp")
            end
        end

        if calcJ
            J .= (I - K.*d2lp')
            if any(isinf.(J))
                print("non-finite J: $J, at x = $x, d2lp = $d2lp")
            end
        end
    end

    # BackTracking linesearch seems to be more robust
    r = nlsolve(only_fj!(Ffj!), f0; iterations=maxiter, ftol=tol, xtol=1e-9, method=:newton, linesearch=LineSearches.BackTracking())
    if !converged(r)
        # println(r)
        println("Laplace mode not converged!")
    end
    f̂ = r.zero

    sumlp = 0.0
    @inbounds @simd for i = 1:N
        dlp[i], d2lp[i] = loglikelihood_grad(lik, Y[i], f̂[i])
        d2lp[i] = min(Wmin, d2lp[i])
        sumlp += loglikelihood(lik, Y[i], f̂[i])
    end

    return (r.zero, sumlp, dlp, d2lp)
end # function


"""
    infLaplace(lik, ncovhyps, fdcov!, K, f, Y, calcderiv=true)

Similar to infLaplace.m in GPML.  This function calculates the approximate log
marginal likelihood using Laplace method and its derivative w.r.t the
hyperparameters of the covariance function, but not the mean function currently).

We assume the mean function is zero for now; suppor for mean function is a future direction.

Inputs:
- lik: the likelihood instance
- ncovhyps: number of covariance function hyperparameters
- fdcov!: function fdcov!(C, i) calculates the derivative of the covariance matrix K
    w.r.t hyperparameter theta_i and writes the results to C
- K: covariance matrix with the current hyperparameters
- f: current latent variable values for all observations; can be `nothing`
    initially and they will be automatically calculated.
- Y: vector of observations
- calcderiv: true if derivatives (`ndlZcov` and `ndlZlik`) are to be calculated.

Outputs: (nlZ, ndlZcov, f, a, L, Wsqrt)
- nlZ: negative log marginal likelihood
- ndlZcov: gradient of nlZ w.r.t all covariance function hyperparameters; only if calcderiv=true
- f: the Laplace mode
- a: the vector alpha = K^(-1) f
- L: Cholesky decomposition of B = I + W½*K*W½
- Wsqrt: W½ = square root of W where W = -∇∇ log(p(y|f))


Note that a better implementation will find the Laplace mode for `α` instead
of `f̂`; however for ease of implementation we will not implement that.
That will be for future work.  The current implementation uses Algorithm 5.1
in the GPML book.
"""
function infLaplace(lik, ncovhyps, fdcov!, K, f, Y::AbstractVector{IntervalType{T}}, calcderiv=true) where{T}
    N = length(Y)

    # Calculate Laplace mode
    f, sumlp, dlp, W = laplace_mode(lik, f, K, Y)
    W .= -W
    # b = W.*f + dlp   # Commented out because a doesn't need to be recomputed
    W .= sqrt.(W)  # W is now W^½

    L = cholesky( Symmetric(I + W.*(K.*W')) )
    R = W .* (L \ Diagonal(W))

    # Calculate α = K^(-1)*f, which turns out to be dlp = ∇ p(y|f)
    # a = b - R*(K*b)  # a = K^(-1)*f
    a = dlp

    # log marginal likelihood
    logZ = -0.5*dot(a,f) + sumlp - sum(log.(diag(L.U)))
    if isinf(logZ)
        println("logZ inf: a = $a, f = $f, sumlp = $sumlp")
    end
    if !calcderiv
        return (-logZ, f, a, L, W)
    end

    G3latent = Vector{Float64}(undef, N)
    @inbounds @simd for i = 1:N
        G3latent[i] = loglikelihood_grad3(lik, Y[i], f[i])
    end

    C = L.L \ (W.*K)
    s2pre = (diag(K) - vec(sum(C.*C, dims=1)))
    s2 = s2pre.*G3latent

    dlogZcov = Vector{Float64}(undef, ncovhyps)
    b = G3latent    # Reuse vector G3latent for b to reduce allocations: G3latent not used anymore
    @inbounds @simd for i = 1:ncovhyps
        fdcov!(C, i)
        dlogZcov[i] = (dot(a, C*a) - sum(R.*C)) / 2
        b .= C * dlp
        dlogZcov[i] += 0.5*dot(s2, b - K*(R*b))
    end

    #=
    # Calculate derivative of log marginal likelihood to likelihood parameters
    # Explicit part
    dlogZlik = sumdlplik + dot(s2pre, G3lik)/2

    # Implicit part, similar to the derivative w.r.t kernel hyperparameters
    b = K*G2lik
    dfdσ = b - K*(R*b)
    dlogZlik += 0.5*dot(s2, dfdσ)

    # dlogZlik times σ for derivative w.r.t log(σ)
    =#

    return (-logZ, -dlogZcov, f, a, L, W)
end


include("likprobit.jl")
include("liklogistic.jl")

end  # module GPInterval
