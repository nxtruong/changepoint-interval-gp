push!(LOAD_PATH, pwd())

using GaussianProcesses
using GPInterval
using LoadData
import BSON

## Load and normalize data
train_inputs, train_outputs, test_inputs, test_outputs = LoadData.load_data(
    "../neuroblastoma-data/data/systematic",
    "../neuroblastoma-data/data/systematic/cv/sequenceID",
    1,
    rmchrom=false)

# Remove chr* columns from normalization
normcols = filter(x -> !startswith(String(x),"chr"), names(train_inputs))
deleteat!(normcols, findall(x->x==:sequenceID, normcols))
train_inputs, norms = normalize_df!(train_inputs, normcols)

# normalize all columns of training inputs, except sequenceID (idx=1)
ncols = size(train_inputs, 2)
#train_inputs, norms = normalize_df!(train_inputs, collect(2:ncols))

# Apply same normalization to test_inputs
normalize_df!(test_inputs, norms)

## Train GPInt modelk
# Create the vectors of training and test data sets
# selfeatures = [6, 10, 36, 37, 38, 43, 75, 76, 78] .+ 1
selfeatures = 2:ncols

train_X = Matrix{Float64}(train_inputs[:, selfeatures])'

train_YL = Vector{Float64}(train_outputs[Symbol("min.log.lambda")])
train_YH = Vector{Float64}(train_outputs[Symbol("max.log.lambda")])
train_Y = [IntervalType{Float64}((l,h)) for (l,h) in zip(train_YL,train_YH)]
D, Ntrain = size(train_X)

test_X = Matrix{Float64}(test_inputs[:, selfeatures])'
test_YL = Vector{Float64}(test_outputs[Symbol("min.log.lambda")])
test_YH = Vector{Float64}(test_outputs[Symbol("max.log.lambda")])
test_Y = [IntervalType{Float64}((l,h)) for (l,h) in zip(test_YL,test_YH)]

# Select a subset of training data
Nsel = 500
seltrain_X = @view train_X[:,1:Nsel]
seltrain_Y = @view train_Y[1:Nsel]

# Set up the GP
kern = SEArd(zeros(D), 0.0)
# kern = LinArd(zeros(D))
# kern = RQArd(zeros(D), 0.0, 0.0)

# likσ = 0.15
# lik = IntervalProbitLikelihood(likσ)
a = 10
lik = IntervalLogisticLikelihood(a)

gpmodel = GPInt(lik, kern, seltrain_X, seltrain_Y)
train!(gpmodel; iterations=1000, show_trace=true, show_every=10)

# Test
pred_Y_train, pred_σ2_train = GPInterval.predict(gpmodel, seltrain_X)
check_train = [y.l <= p <= y.h for (p,y) in zip(pred_Y_train, seltrain_Y)]

pred_Y, pred_σ2 = GPInterval.predict(gpmodel, test_X)
check_test = [y.l <= p <= y.h for (p,y) in zip(pred_Y, test_Y)]
check_test_accuracy = count(check_test) / length(test_Y) * 100.0

covhyps = GaussianProcesses.get_params(gpmodel.cov)[1:end-1]
selfeatures = findall(exp.(covhyps) .<= 40)

BSON.bson("gpse_sequenceID_fold1_0500_logistic_fullfeatures.bson",
    covhyp=GaussianProcesses.get_params(gpmodel.cov),
    # liksigma=likσ,
    liklogistica=a,
    logmarginallik=gpmodel.lml,
    Nsel=Nsel,
    selfeatures=selfeatures,
    accuracy=check_test_accuracy)




#=
set_params!(kern, [-0.43408, -0.97038, -2.80301, -9.74359, -28.4308, -0.696018, -22.8862, -30.4673, 9.28371, 5.33928, -1.13137, -27.6784, -56.4111, -27.6784, -0.000755737, -9.05907e-5, -7.04285e-5, -0.000271731, -0.000313506, -0.000342159, -0.000410633, -0.000409064, -0.000449461, -0.000472888, -0.000509878, -0.000504685, -0.000540695, -0.000529386, -0.000556507, -0.000545655, -0.000553621, -0.000544941, -0.000544523, -0.000535377, -0.345653, -0.244553, -0.211529, -0.832528, -1.05707, -1.21487, -1.6066, -1.68444, -1.98963, -2.19854, -2.54264, -2.60767, -3.01788, -3.07001, -3.48123, -3.55415, -3.84847, -3.96252, -4.20078, -4.34063, -0.148594, -0.0890283, -0.697586, -0.327385, -0.244283, -0.200346, -0.128624, -0.112369, -0.0785302, -0.0607284, -0.0379107, -0.0336808, -0.0168877, -0.0143501, -0.00351739, -0.0014485, 0.00326679, 0.00463685, 0.00691629, 0.00775436, -10.6579, -5.72171, -8.887, -4.86598, -3.9296, -3.37239, -2.28909, -2.09234, -1.49374, -1.15679, -0.6657, -0.601613, -0.172764, -0.120175, 0.21701, 0.269675, 0.449529, 0.497372, 0.599157, 0.644728, 17.4824, 0.0, -364.26, -2.81514, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 34.5513, 0.0, 0.0, 0.0, 0.0, 0.0, 0.458971, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.947673])
=#
