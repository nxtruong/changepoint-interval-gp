push!(LOAD_PATH, pwd())

using GaussianProcesses
using GPInterval
using LoadData
using TableReader
using DataFrames, CSV

cvtype = "profileSize"

# Select the fold (1 to 6)
cvfold = 3

# Select the random test fold order (1 to 5)
testfolds = 5

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

## Load the random selection order
sample_order = readcsv("../neuroblastoma-data/data/systematic/cv/$cvtype/testFolds/$cvfold/randomTrainOrderings/$testfolds/order.csv")

# Create the vectors of training data set
# selfeatures = [2, 4, 6, 11, 36, 55] .+ 1
selfeatures = 2:ncols

# Use the same order as in sample_order
train_seqs = train_inputs[:sequenceID]
sample_idx = [findfirst(x->x==r[1], train_seqs) for r in eachrow(sample_order)]      # indices in train_inputs of the samples in sample_order

train_X = Matrix{Float64}(train_inputs[sample_idx, selfeatures])'

train_YL = Vector{Float64}(train_outputs[sample_idx, Symbol("min.log.lambda")])
train_YH = Vector{Float64}(train_outputs[sample_idx, Symbol("max.log.lambda")])
train_Y = [IntervalType{Float64}((l,h)) for (l,h) in zip(train_YL,train_YH)]
D, Ntrain = size(train_X)

test_X = Matrix{Float64}(test_inputs[:, selfeatures])'
test_YL = Vector{Float64}(test_outputs[Symbol("min.log.lambda")])
test_YH = Vector{Float64}(test_outputs[Symbol("max.log.lambda")])
test_Y = [IntervalType{Float64}((l,h)) for (l,h) in zip(test_YL,test_YH)]

# Sequence of number of training samples taken
nbr_trainings = [collect(10:10:100); 150]  # collect(150:50:300)]
model_accuracies = Vector{Float64}()

for Nsel in nbr_trainings
    println("====================================================")
    println("For $Nsel number of samples...")

    seltrain_X = @view train_X[:,1:Nsel]
    seltrain_Y = @view train_Y[1:Nsel]

    # Set up the GP
    #kern = SEArd(zeros(D), 0.0)
    kern = LinArd(zeros(D))
    # kern = RQArd(zeros(D), 0.0, 0.0)

    #likσ = 0.1
    #lik = IntervalProbitLikelihood(likσ)
    a = 4 # 4
    lik = IntervalLogisticLikelihood(a)

    gpmodel = GPInt(lik, kern, seltrain_X, seltrain_Y)
    train!(gpmodel; iterations=1000, show_trace=true, show_every=10)

    # Test
    pred_Y, pred_σ2 = GPInterval.predict(gpmodel, test_X)
    check_test = [y.l <= p <= y.h for (p,y) in zip(pred_Y, test_Y)]
    check_test_accuracy = count(check_test) / length(test_Y) * 100.0

    println("Model Accuracy: $check_test_accuracy")

    push!(model_accuracies, check_test_accuracy)
end

CSV.write("../neuroblastoma-data/data/systematic/cv/$cvtype/testFolds/$cvfold/randomTrainOrderings/$testfolds/gpmodel1.csv",
    DataFrame(trainsize=nbr_trainings[1:length(model_accuracies)], accuracy=model_accuracies))
