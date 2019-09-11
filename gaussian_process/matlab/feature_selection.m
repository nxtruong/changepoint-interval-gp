% Select features for a data set

clear all;
close all;

infMethod = @infEP;

dataset_type = 'detailed';  % the name of the data set
rmchrom = false;  % whether to remove chr* columns from inputs
Ntrain = -1;  % number of samples used for training; -1 if automatic
lik_a = 10;  % parameter 'a' of the likelihood function

lengthscale_gap = 2.5;  % remove features whose length scales are more than exp(lengthscale_gap) times the minimum length scale

datadir = ['~/working/neuroblastoma-data/data/' dataset_type];

% Load and normalize data
df_inputs = readtable([datadir '/inputs.csv']);

% The output data (Y - targets)
df_outputs = readtable([datadir '/outputs.csv']);

% Remove rows where both Y outputs are Inf
doubleinf_idx = isinf(df_outputs{:, 'min_log_lambda'}) & isinf(df_outputs{:, 'max_log_lambda'});
df_outputs(doubleinf_idx,:) = [];
df_inputs(doubleinf_idx,:) = [];

% Do not select some features, especially 'sequenceID'
selfeatures = find(cellfun(@(x) ~startsWith(x, 'chr') && ~startsWith(x, 'rss_') && ~startsWith(x, 'mse_') && ~strcmp(x, 'sequenceID'), df_inputs.Properties.VariableNames));

% Remove features that have non-finite values
selfeatures(any(~isfinite(df_inputs{:,selfeatures}), 1)) = [];

% Remove chr* columns from normalization
normcols = selfeatures;
normcols = df_inputs.Properties.VariableNames(normcols);
[df_inputs, norms] = normalize_df(df_inputs, normcols);

% Normalization may create non-finite values; remove them
selfeatures(any(~isfinite(df_inputs{:,selfeatures}), 1)) = [];

ncols = size(df_inputs, 2);

if Ntrain < 0
    Ntrain = min(size(df_inputs,1), 5*numel(selfeatures));
end

% Generate training data
sample_idx = 1:Ntrain;
train_X = df_inputs{sample_idx, selfeatures};
[Ntrain, D] = size(train_X);
            
train_YL = df_outputs{sample_idx, 'min_log_lambda'};
train_YH = df_outputs{sample_idx, 'max_log_lambda'};

assert(Ntrain == length(train_YH) && length(train_YL) == Ntrain && Ntrain > 0);

train_Y = RealInterval(train_YL,train_YH);

% Set up GP model
hyp = struct;

likfunc = {@likIntervalLogistic, lik_a};
hyp.lik = [];

covfunc = @covSEard;
hyp.cov = zeros(1, D+1);

meanfunc = @meanZero;
hyp.mean = [];

% Train
disp('Start training ...');
tic;
hyp = minimize(hyp, @gp, 20000, infMethod, meanfunc, covfunc, likfunc, train_X, train_Y);
[nlZ, dnlZ] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, train_X, train_Y);
fprintf('Finished training after %f seconds; nlZ = %f.\n', toc(), nlZ);

% Inspect the length scales
ls_all = hyp.cov(1:end-1);
ls_min = min(ls_all);

% Selected or removed features, whose indices are from the original columns
% of input data table (inputs.csv)
rm_features = selfeatures(ls_all > (ls_min + lengthscale_gap) & ls_all > 0);
sel_features = selfeatures(ls_all <= (ls_min + lengthscale_gap) | ls_all <= 0);

% Save feature selection results to a file
save([datadir '/feature_selection_GP.mat'], 'hyp', 'nlZ', 'dnlZ', 'selfeatures', 'rm_features', 'sel_features', 'rmchrom', 'Ntrain', 'lik_a', 'dataset_type');