# Sample selection based on Gaussian Process model

push!(LOAD_PATH, pwd())

using GaussianProcesses
using GPInterval
using LoadData
using TableReader
using DataFrames, CSV
import NLopt
import LineSearches
using Optim

"""
Ninit½: half the initial number of samples to be selected randomly.  If it's negative,
the first pair are the extreme intervals with the smallest upper limit and the largest
lower limit; the rest are random.  If it's positive, all initial sampels are random.
"""
function ExperimentDesign(full_X, full_Y::Vector{IntervalType{Float64}},
    test_X, test_Y::Vector{IntervalType{Float64}}, Ntrain, Ninit½=1)

    @assert abs(Ninit½) >= 1 "Invalid initial number of samples."

    D, N = size(full_X)
    @assert N == length(full_Y) "Inconsistent numbers of training samples."

    @assert Ntrain >= 2 && Ntrain <= N && Ntrain > 2*Ninit½ "Invalid Ntrain"

    # Select random initial samples
    # Find indices of samples with lower limit [YL is finite] and upper limit (YH is finite)
    idx_l = findall(y -> isfinite(y.l), full_Y)
    idx_h = findall(y -> isfinite(y.h), full_Y)

    # Select random initial samples
    IdxSelected = Vector{Int}()
    if Ninit½ < 0
        # First pair are extremes
        if !isempty(idx_l)
            idx = findmax([y.l for y in full_Y[idx_l]])[2]
            push!(IdxSelected, idx_l[idx])
            deleteat!(idx_l, idx)
        end
        if !isempty(idx_h)
            idx = findmin([y.h for y in full_Y[idx_h]])[2]
            push!(IdxSelected, idx_h[idx])
            deleteat!(idx_h, idx)
        end
        Ninit½ = -Ninit½ - 1
    end
    for k = 1:Ninit½
        if !isempty(idx_l)
            idx = rand(1:length(idx_l))
            push!(IdxSelected, idx_l[idx])
            deleteat!(idx_l, idx)
        end
        if !isempty(idx_h)
            idx = rand(1:length(idx_h))
            push!(IdxSelected, idx_h[idx])
            deleteat!(idx_h, idx)
        end
    end

    # All the remaining samples
    IdxRemaining = setdiff(1:N, IdxSelected)

    result_predictions = Dict{Int64, Vector{Float64}}()
    result_accuracy = Dict{Int64, Float64}()

    # likσ = 0.1
    # lik = IntervalProbitLikelihood(likσ)
    a = 3 # 4
    lik = IntervalLogisticLikelihood(a)


    # Loop: train, select next sample in the remainings that have the max variance
    for k = length(IdxSelected):Ntrain
        println("====================================================")
        println("Selection for $(k+1) number of samples...")

        seltrain_X = @view full_X[:, IdxSelected]
        seltrain_Y = @view full_Y[IdxSelected]

        # Set up the GP
        # We set up the kernel here so that its hyperparameters are retained, hopefully to get better optimization results
        # kern = SEArd(zeros(D), 0.0)
        kern = LinArd(zeros(D))
        # kern = RQArd(zeros(D), 0.0, 0.0)

        gpmodel = GPInt(lik, kern, seltrain_X, seltrain_Y)
        println("- Training GP model with selected samples ...")
        # train!(gpmodel; method=ConjugateGradient(), iterations=1000, show_trace=true, show_every=10)
        train!(gpmodel; method=ConjugateGradient(), lower_bounds=-20.0, upper_bounds=20.0, ftol_rel=0, xtol_rel=0)

        # Test
        println("- Calculating accuracy of the GP model ...")
        pred_Y, pred_σ2 = GPInterval.predict(gpmodel, test_X)
        check_test = [y.l <= p <= y.h for (p,y) in zip(pred_Y, test_Y)]
        result_accuracy[k] = count(check_test) / length(test_Y) * 100.0
        result_predictions[k] = pred_Y

        println("- Model Accuracy: $(result_accuracy[k])")

        if k < Ntrain
            # Predict at remaining samples
            pred_Y, pred_σ2 = GPInterval.predict(gpmodel, @view full_X[:, IdxRemaining])

            # Find sample with the max variance
            varmax, idxmax = findmax(pred_σ2)
            idxnext = IdxRemaining[idxmax]

            # Store idxmax sample and remove it from IdxRemaining
            push!(IdxSelected, idxnext)
            deleteat!(IdxRemaining, idxmax)

            println("- Done! Selected sample $idxnext.")
        end
    end

    # Returns the samples and the LR models
    return IdxSelected, result_accuracy, result_predictions
end


# Settings for the data sets
cvtype = "chrom"

# Select the fold (1 to 6)
cvfold = 1

# The directory name
resultdirname = "sampleSelectionGP_LIN"

## Load and normalize data
train_inputs, train_outputs, test_inputs, test_outputs = LoadData.load_data(
    "../neuroblastoma-data/data/systematic",
    "../neuroblastoma-data/data/systematic/cv/" * cvtype,
    cvfold,
    rmchrom=true)

# Remove chr* columns from normalization
normcols = filter(x -> !startswith(String(x),"chr"), names(train_inputs))
deleteat!(normcols, findall(x->x==:sequenceID, normcols))
train_inputs, norms = normalize_df!(train_inputs, normcols)

ncols = size(train_inputs, 2)

# Apply same normalization to test_inputs
normalize_df!(test_inputs, norms)

# Create the vectors of training data set
# selfeatures = [2, 4, 6, 11, 36, 55] .+ 1
selfeatures = 2:ncols

full_X = Matrix{Float64}(train_inputs[selfeatures])'

train_YL = Vector{Float64}(train_outputs[Symbol("min.log.lambda")])
train_YH = Vector{Float64}(train_outputs[Symbol("max.log.lambda")])
full_Y = [IntervalType{Float64}((l,h)) for (l,h) in zip(train_YL,train_YH)]

test_X = Matrix{Float64}(test_inputs[:, selfeatures])'
test_YL = Vector{Float64}(test_outputs[Symbol("min.log.lambda")])
test_YH = Vector{Float64}(test_outputs[Symbol("max.log.lambda")])
test_Y = [IntervalType{Float64}((l,h)) for (l,h) in zip(test_YL,test_YH)]

testfold = 1

while testfold <= 5
    global testfold
    # Call experiment design function
    if testfold == 1
        Ninit = -2
    else
        Ninit = 2
    end
    IdxSelected, result_accuracy, result_predictions = ExperimentDesign(full_X, full_Y, test_X, test_Y, 80, Ninit)

    # Save the results
    fullresultdir = "../neuroblastoma-data/data/systematic/cv/$cvtype/testFolds/$cvfold/$resultdirname/$testfold"
    mkpath(fullresultdir)

    CSV.write("$fullresultdir/order.csv", DataFrame(sequenceID=train_inputs[IdxSelected, :sequenceID]))

    mykeys = sort(collect(keys(result_accuracy)))
    CSV.write("$fullresultdir/gpaccuracy.csv", DataFrame(n=mykeys, accuracy=[result_accuracy[k] for k in mykeys]))

    CSV.write("$fullresultdir/gppredictions.csv", insertcols!(DataFrame(result_predictions), 1; sequenceID=test_inputs[:sequenceID]))

    testfold += 1
end
