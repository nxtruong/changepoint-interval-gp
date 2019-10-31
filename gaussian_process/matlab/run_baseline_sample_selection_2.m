function run_baseline_sample_selection_2(PATH, cvfold, max_samples, resultpostfix)
    % Baseline sample selection based on Gaussian Process model
    % It doesn't use an interval likelihood function but instead train the
    % GPs on data where scalar labels are generated from interval labels by
    % taking points within the intervals.
    %
    % With only two features:
    % log.hall = log(variance estimate)
    % log2.n = log(log(number of data points)
    %
    % However these two column names are not present in all feature files. below on the left we have the
    % column name that may appear in the feature files, and then on the right the canonical feature name.
    % rep.val.vec <- c(
    % log.log.bases="log2.n",
    % n.loglog="log2.n",
    % "diff abs.identity.quantile.50%"="log.hall",
    % log.sd="log.hall")
    %
    % PATH is the path to the testfold, in the form similar to:
    % "neuroblastoma-data/data/systematic/cv/sequenceID/testFolds/1"
    % The results are placed inside PATH/sampleSelectionGP_<kernel-name>
    %
    % cvfold is the fold number, typically the same as the last part of
    % PATH (in the above example, it's 1).
    %
    % max_samples: Max number of samples to be selected
    %
    % resultpostfix is an optional string that adds a postfix to the result
    % directory's name. Default is ''. Example: '_somepostfix'.
    
    assert(ischar(PATH) && ~isempty(PATH) && exist(PATH, 'dir'), 'Invalid path.');

    if ~exist('max_samples', 'var'), max_samples = 10; end
    assert(max_samples >= 8);
    
    if ~exist('resultpostfix', 'var'), resultpostfix = ''; end
    assert(ischar(resultpostfix), 'resultpostfix must be a string.');
    
    % The result directory
    resultdirname = sprintf('sampleSelectionGP_%s%s', 'baseline', resultpostfix);
    resultdir = fullfile(PATH, resultdirname);
    if ~exist(resultdir, 'dir')
        mkdir(resultdir);
    end
    
    % Load and normalize data
    cvdir = go_up(PATH, 2);
    datadir = go_up(cvdir, 2);
    [train_inputs, train_outputs, test_inputs, test_outputs] = load_data(...
        datadir, cvdir, cvfold, true);
    
    % Select only two features above (see comment at top)
    
    % "log_log_bases" or "n_loglog" or "log2_n"
    selfeatures = find(cellfun(@(x) ismember(x, {'log_log_bases', 'n_loglog', 'log2_n'}), train_inputs.Properties.VariableNames));
    assert(numel(selfeatures) == 1, 'Feature "log2.n" not available.');
    
    % "diffAbs_identity_quantile_50_" or "log_sd" or "log_hall"
    selfeatures = [selfeatures, find(cellfun(@(x) ismember(x, {'diffAbs_identity_quantile_50_', 'log_sd', 'log_hall'}), train_inputs.Properties.VariableNames))];
    assert(numel(selfeatures) == 2, 'Feature "log.hall" not available.');
    
    normcols = selfeatures;
    [train_inputs, norms] = normalize_df(train_inputs, normcols);
    
    % ncols = size(train_inputs, 2);
    
    % Apply same normalization to test_inputs
    test_inputs = normalize_df(test_inputs, norms);
    
    % Create the vectors of training data set
    full_X = train_inputs{:, selfeatures};
    
    full_YL = train_outputs{:, 'min_log_lambda'};
    full_YH = train_outputs{:, 'max_log_lambda'};
    
    % Remove all training points with both YL and YH being inf
    useless_pnts = find(isinf(full_YL) & isinf(full_YH));
    if ~isempty(useless_pnts)
        disp('Found training points with inf lower and upper values. They will be removed.');
        full_X(useless_pnts, :) = [];
        full_YL(useless_pnts) = [];
        full_YH(useless_pnts) = [];
    end
    
    % Convert from intervals to scalars
    full_Y = full_YL + 1;
    idx = isfinite(full_YL) & isfinite(full_YH);
    full_Y(idx) = (full_YL(idx) + full_YH(idx)) * 0.5;
    idx = isinf(full_Y);
    full_Y(idx) = full_YH(idx) - 1;

    % [Ntrain, D] = size(full_X);
    
    test_X = test_inputs{:, selfeatures};
    test_YL = test_outputs{:, 'min_log_lambda'};
    test_YH = test_outputs{:, 'max_log_lambda'};
    % Ntest = size(test_X, 1);
    
    testfold = 1;
    maxtestfold = 5;
    
    while testfold <= maxtestfold
        % Call experiment design function
        %     if testfold == 1
        %        Ninit = -2;
        %     else
        %        Ninit = 2;
        %     end
        Ninit = -2;
        
        %usetrainset = true;
        [IdxSelected, result_accuracy, result_predictions, result_covfunc, result_meanfunc, result_likfunc, n_seq] = experiment_design_for_test_set(full_X, full_Y, test_X, test_YL, test_YH, max_samples, Ninit, @infGaussLik, 'erf');  % the likname = 'erf' does not really matter
        
        % Save the results
        testfoldresultdir = fullfile(resultdir, int2str(testfold));
        if ~exist(testfoldresultdir, 'dir')
            mkdir(testfoldresultdir);
        end
        
        writetable(train_inputs(IdxSelected, 'sequenceID'), fullfile(testfoldresultdir, 'order.csv'));
        
        results = table(n_seq, result_accuracy(:), result_covfunc(:), result_meanfunc(:), result_likfunc(:), ...
            'VariableNames', {'n', 'accuracy', 'cov', 'mean', 'lik'});
        writetable(results, fullfile(testfoldresultdir, 'gpaccuracy.csv'));
        
        results = table(test_inputs{:,'sequenceID'}, result_predictions{:}, ...
            'VariableNames', [{'sequenceID'}, arrayfun(@(n) sprintf('n%d', n), n_seq(:)', 'UniformOutput', false)]);
        writetable(results, fullfile(testfoldresultdir, 'gppredictions.csv'));
        
        testfold = testfold + 1;
    end
end

function P = go_up(PATH, levels)
    % Given a path, returns a path P that is a given number of levels up
    % from PATH.  Returns empty string if error.
    assert(levels > 0);
    if isempty(PATH), P = ''; return; end
    if levels == 1
        P = PATH;
        name = '';
        while isempty(name) && ~isempty(P)
            % This loop is needed because if there are one or more file
            % separates at the end of PATH, name will be empty and P is
            % essentially the same.
            [P,name,~] = fileparts(P);
        end
    else
        P = go_up(go_up(PATH, 1), levels-1);
    end
end