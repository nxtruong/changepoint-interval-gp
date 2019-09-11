# Test code for GPInterval.jl
push!(LOAD_PATH, pwd())

using GPInterval
using ForwardDiff

# The following code uses AD to check the above implementation is correct
y1 = IntervalType{Float64}((0.0, 0.5))
y2 = IntervalType{Float64}((0.1, Inf))
y3 = IntervalType{Float64}((-Inf, 0.3))
Y = [y1, y2, y3]
f0 = [isfinite(y.l) ? y.l : y.h for y in Y]

σ = 0.01
lik = GPInterval.IntervalProbitLikelihood(σ)

f = 0.11
i = 1

ll = GPInterval.loglikelihood(lik, Y[i], f)

GLL, GGLL = GPInterval.loglikelihood_grad(lik, Y[i], f)

checkGLL = ForwardDiff.derivative(x -> GPInterval.loglikelihood(lik, Y[i], x), f)
@show checkGLL - GLL

checkGGLL = ForwardDiff.derivative(
    x -> ForwardDiff.derivative(x -> GPInterval.loglikelihood(lik, Y[i], x), x),
    f)
@show checkGGLL - GGLL

G3f = GPInterval.loglikelihood_grad3(lik, Y[i], f)

checkG3f = ForwardDiff.derivative(
            x -> ForwardDiff.derivative(
                    x -> ForwardDiff.derivative(x -> GPInterval.loglikelihood(lik, Y[i], x), x),
                    x),
            f)
@show checkG3f - G3f
# End of test code



# Test code for laplace_mode and infLaplace
import GaussianProcesses: cov!, KernelData, grad_slice!, cov, SEArd

X = rand(3,3)

covhyps = log.([1.0, 1.5, 2.7])
covsigma = log(0.2)
covf = SEArd(covhyps, covsigma)
covdata = KernelData(covf, X, X)

K = zeros(3,3)
cov!(K, covf, X, covdata)

fdcov!(C,p) = grad_slice!(C, covf, X, covdata, p)
lZ, dlZcov, f̂, α = infLaplace(lik, 4, fdcov!, K, nothing, Y)

dσ = 0.000001

covhyps1 = copy(covhyps)
# covhyps1[3] += dσ
covsigma1 = copy(covsigma)
#covsigma1 += dσ
covf1 = SEArd(covhyps1, covsigma1)
covdata1 = KernelData(covf1, X, X)
K1 = zeros(3,3)
cov!(K1, covf1, X, covdata1)
fdcov!1(C,p) = grad_slice!(C, covf1, X, covdata1, p)

f̂, sumlp, dlp, d2lp, sumdlplik = GPInterval.laplace_mode(f0, K, Y, σ)
f̂1, sumlp1, dlp1, d2lp1, sumdlplik1 = GPInterval.laplace_mode(f0, K1, Y, σ+dσ)
check_df̂_dσ = (f̂1 - f̂)/dσ
nlZ, dnlZcov, dnlZlik, f̂, α = GPInterval.infLaplace(log(σ), 4, fdcov!, K, nothing, Y)
nlZ1, dnlZcov1, dnlZlik1, f̂1, α1 = GPInterval.infLaplace(log(σ)+dσ, 4, fdcov!1, K1, nothing, Y)

check_dnlZ = (nlZ1 - nlZ)/dσ
@show check_dnlZ - dnlZlik
# End of test code



# Test GPInterval prediction (basic computation)
using GaussianProcesses

x = rand(1, 20) * 4π
y = vec(sin.(x) .+ randn(1, size(x,2))*0.001)
kern = SE(0.0, 0.0)
mZero = MeanZero()

gp = GP(x, y, mZero, kern)
optimize!(gp)

scaling_factor = 1    # For numerical robustness of GPInt

yint = [IntervalType{Float64}((rand() < 0.2 ? -Inf : (v-rand()*0.1)*scaling_factor, (v+rand()*0.1)*scaling_factor)) for v in y]

minwidth = minimum([y.h-y.l for y in yint])
σ = minwidth / 10

lik = IntervalProbitLikelihood(σ)

kern = SE(0.0, 0.0)

gpint = GPInt(lik, kern, x, yint)
train!(gpint)

xtest = collect(range(π, 3π, length=20))
xtest = xtest'
ifmean, iσ2 = GPInterval.predict(gpint, xtest)
ifmean ./ scaling_factor
iσ2 ./ (scaling_factor*scaling_factor)


using Plots
gr()
plot(gp)
scatter!(vec(xtest), vec(ifmean))
plot!(vec(xtest), vec(ifmean) .+ 2*sqrt.(iσ2))
plot!(vec(xtest), vec(ifmean) .- 2*sqrt.(iσ2))
# End of test code
