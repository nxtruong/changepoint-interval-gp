# Main module for GP with interval observations
# (C) 2019 by Truong X. Nghiem (truong.nghiem@gmail.com)

module GPInterval

using Distributions
using LinearAlgebra
import Base: show
using Optim
import LineSearches
using GaussianProcesses
import NLopt

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
    function GPInt(lik, cov::GaussianProcesses.Kernel, X, Y::AbstractVector{IntervalType{T}}; α=nothing) where{T}
        N = size(X, 2)
        @assert N == length(Y)

        # α = Vector{Float64}(undef, N)
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
    nlZ, _, gp.f̂, gp.α, gp.L, gp.Wsqrt = infLaplace(gp.lik, GaussianProcesses.num_params(gp.cov), gp.fdcov!, K, gp.α, gp.Y)
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
function train!(gp::GPInt; method, lower_bounds=-Inf, upper_bounds=Inf, ftol_rel=1e-3, xtol_rel=1e-3, kwargs...)
    # Pre-allocate space for covariance matrix
    N = size(gp.X,2)
    K = Matrix{Float64}(undef, N, N)

    # Create functions for optimization with NLopt
    function fg!(x,G)
        # Set the parameters of the GP
        set_params!(gp.cov, x)

        GaussianProcesses.cov!(K, gp.cov, gp.X, gp.covdata)
        if any(isnan.(K))
            println(x)
            println(exp.(x))
            println(K)
            error("K is nan")
        end

        f̂0 = nothing
        # f̂0 = gp.f̂

        try
            if length(G) > 0
                nlZ, dnlZcov, gp.f̂ = infLaplace(gp.lik, GaussianProcesses.num_params(gp.cov), gp.fdcov!, K, f̂0, gp.Y)
                G .= dnlZcov
            else
                nlZ, gp.f̂ = infLaplace(gp.lik, GaussianProcesses.num_params(gp.cov), gp.fdcov!, K, f̂0, gp.Y, false)
            end
        catch
            println("ERROR: Exception in train! causing forced stop.")
            throw(NLopt.ForcedStop())
        end

        return nlZ
    end

    # Create functions for optimization with Optim
    function optimfg!(F,G,x)
        # Set the parameters of the GP
        set_params!(gp.cov, x)

        GaussianProcesses.cov!(K, gp.cov, gp.X, gp.covdata)
        if any(isnan.(K))
            println(x)
            println(exp.(x))
            println(K)
            error("K is nan")
        end

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
    DIM = length(x0)

    # Try with Optim first
    x = nothing
    try
        results = optimize(Optim.only_fg!(optimfg!), x0; method=method, kwargs...)
        x = Optim.minimizer(results)
    catch
        println("ERROR with Optim. Fall back to NLopt.")
        opt = NLopt.Opt(:LD_MMA, DIM)

        opt.min_objective = fg!

        if !isa(lower_bounds, Vector)
            if isfinite(lower_bounds)
                opt.lower_bounds = fill(lower_bounds, DIM)
            end
        else
            opt.lower_bounds = lower_bounds
        end

        if !isa(upper_bounds, Vector)
            if isfinite(upper_bounds)
                opt.upper_bounds = fill(upper_bounds, DIM)
            end
        else
            opt.upper_bounds = upper_bounds
        end

        opt.ftol_rel = ftol_rel
        opt.xtol_rel = xtol_rel

        (optf, x, ret) = NLopt.optimize(opt, x0)
    end

    if x != nothing
        update_params!(gp, x, K)
        return gp
    else
        error("ERROR: All optimization solvers failed!!!")
    end
end # function
#=
function train!(gp::GPInt; method = ConjugateGradient(linesearch=LineSearches.HagerZhang()), kwargs...)
    # Pre-allocate space for covariance matrix
    N = size(gp.X,2)
    K = Matrix{Float64}(undef, N, N)

    # Create functions for optimization
    function fg!(F,G,x)
        # Set the parameters of the GP
        set_params!(gp.cov, x)

        GaussianProcesses.cov!(K, gp.cov, gp.X, gp.covdata)
        if any(isnan.(K))
            println(x)
            println(exp.(x))
            println(K)
            error("K is nan")
        end

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
=#

"""
    laplace_mode(lik, f̂, K, Y::IntervalType{T})

Find the posterior mode of the GP with interval observations, for Laplace method.
Similar to Algorithm 3.1 in GPML book (page 46).
Current implementation: directly maximize Ψ(α) where α = K*f̂.
Future implementation may use IRLS algorithm as implemented in
GPML toolbox's `infLaplace.m`.

Inputs:
- `lik`: the interval likelihood instance
- `α0`: initial values for `α`
- `K`: covariance matrix
- `Y`: vector of interval observations

Outputs:
- `f̂`: modal value, which is K*α
- `α`: the solution
- `sumlp`: sum of approximated `log p(y | f)` for all y in Y
- `dlp`: approximated `∇ log p(y | f)`, in the optimal, this should be equal to α
- `d2lp`: approximated `∇ ∇ log p(y | f)`
"""
function laplace_mode(lik, α0, K, Y::AbstractVector{IntervalType{T}}; maxiter=1000, Wmin=0.0, tol=1e-6) where{T}
    @assert !any(isnan.(K)) "K is nan in laplace_mode."
    N = length(Y)
    if α0 == nothing
        α0 = initialvec(lik, Y, K)
    end
    Wmin = -Wmin

    # Pre-allocate dlp (derivative of lp), d2lp
    dlp = Vector{Float64}(undef, N)
    d2lp = Vector{Float64}(undef, N)

    # Functions that calculate `F(α)` and its Jacobian
    function F(x)
        f = K*x     # The modal vector

        # Calculate objective
        sumlp = 0.0
        @inbounds @simd for i = 1:N
            sumlp += loglikelihood(lik, Y[i], f[i])
        end
        obj = 0.5*dot(x,f) - sumlp
        if !isfinite(obj)
            println("non-finite objective: $obj, at x = $x, sumlp = $sumlp")
        end
        return obj
    end

    function G!(J, x)
        f = K*x     # The modal vector

        @inbounds @simd for i = 1:N
            dlp[i] = loglikelihood_grad(lik, Y[i], f[i]; calcGGLL=false)
        end
        J .= K*(x - dlp)
        if any(isinf.(J))
            println("non-finite J: $J, at x = $x, dlp = $dlp")
        end
    end

    r = optimize(F, G!, α0, BFGS(linesearch=LineSearches.BackTracking()), Optim.Options(iterations=maxiter, f_tol=tol, x_tol=1e-9))
    # r = optimize(F, G!, α0, BFGS(linesearch=LineSearches.BackTracking()), Optim.Options(iterations=maxiter, f_tol=tol, x_tol=1e-9))
    #if !Optim.converged(r)
        # println(r)
    #    println("Laplace mode not converged!")
    #end
    α = Optim.minimizer(r)
    f̂ = K*α

    sumlp = 0.0
    @inbounds @simd for i = 1:N
        dlp[i], d2lp[i] = loglikelihood_grad(lik, Y[i], f̂[i])
        d2lp[i] = min(Wmin, d2lp[i])
        sumlp += loglikelihood(lik, Y[i], f̂[i])
    end

    return (f̂, α, sumlp, dlp, d2lp)
end # function


"""
    infLaplace(lik, ncovhyps, fdcov!, K, α, Y, calcderiv=true)

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
- α: current α = K\f vector; can be `nothing` initially and they will be automatically calculated.
- Y: vector of observations
- calcderiv: true if derivatives (`ndlZcov` and `ndlZlik`) are to be calculated.

Outputs: (nlZ, ndlZcov, f, a, L, Wsqrt)
- nlZ: negative log marginal likelihood
- ndlZcov: gradient of nlZ w.r.t all covariance function hyperparameters; only if calcderiv=true
- f: the Laplace mode
- a: the vector alpha = K^(-1) f
- L: Cholesky decomposition of B = I + W½*K*W½
- Wsqrt: W½ = square root of W where W = -∇∇ log(p(y|f))
"""
function infLaplace(lik, ncovhyps, fdcov!, K, α0, Y::AbstractVector{IntervalType{T}}, calcderiv=true) where{T}
    N = length(Y)

    # Calculate Laplace mode
    f, a, sumlp, dlp, W = laplace_mode(lik, α0, K, Y)
    W .= -W
    # b = W.*f + dlp   # Commented out because a doesn't need to be recomputed
    W .= sqrt.(W)  # W is now W^½

    L = cholesky( Symmetric(I + W.*(K.*W')) )
    R = W .* (L \ Diagonal(W))

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

"""
    calc_α_from_f(f, lik, Y, K)

documentation
"""
function calc_α_from_f(f, lik, Y::AbstractVector{IntervalType{T}}, K) where{T}
    @assert !any(isnan.(K)) "K is nan"
    @assert !any(isnan.(f)) "f is nan"

    # Calculate a = K\f
    try
        a = K \ f

        return a
    catch
        @show K

        # If failed, try another way
        N = length(Y)

        # Pre-allocate dlp (derivative of lp), d2lp
        dlp = Vector{Float64}(undef, N)
        d2lp = Vector{Float64}(undef, N)

        @inbounds @simd for i = 1:N
            dlp[i], d2lp[i] = loglikelihood_grad(lik, Y[i], f[i])
        end
        @assert !any(isnan.(dlp)) "dlp is nan"
        @assert !any(isnan.(d2lp)) "d2lp is nan"

        W = d2lp
        W .= -W
        b = W.*f + dlp
        W .= sqrt.(W)  # W is now W^½
        @assert !any(isnan.(W)) "W is nan"
        @assert !any(isnan.(b)) "b is nan"

        IWKW = Symmetric(I + W.*(K.*W'))
        try
            L = cholesky(IWKW)
            R = W .* (L \ Diagonal(W))
            @assert !any(isnan.(R)) "R is nan"

            a = b - R*(K*b)

            return a
        catch
            println("I + W*K*W is not POSDEF! Eigenvalues:")
            println(eigvals(IWKW))
            rethrow()
        end
    end
end # function


include("likprobit.jl")
include("liklogistic.jl")

end  # module GPInterval
