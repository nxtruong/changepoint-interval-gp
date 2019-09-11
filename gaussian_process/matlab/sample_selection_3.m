% Sample selection based on Gaussian Process model
% With only three features:
% log.hall = log(variance estimate)
% log2.n = log(log(number of data points)
% log.mad

% However these two column names are not present in all feature files. below on the left we have the
% column name that may appear in the feature files, and then on the right the canonical feature name.
% rep.val.vec <- c(
% log.log.bases="log2.n",
% n.loglog="log2.n",
% "diff abs.identity.quantile.50%"="log.hall",
% log.sd="log.hall")

clear all;
close all;

cvtypes = {'sequenceID'}; % {'profileSize', 'profileID', 'chrom', 'sequenceID'};
for ktype = 1:numel(cvtypes)
    cvtype = cvtypes{ktype};
%     switch cvtype
%         case 'profileSize'
%             cvfolds = 6;
%         case 'sequenceID'
%             cvfolds = [1, 2, 3, 5, 6];
%         otherwise
%             cvfolds = 1:6;
%     end
    cvfolds = 1; %1:6;
    for cvfold = cvfolds
        usetrainset = false;
        
        % The directory name
        resultdirname = 'sampleSelectionGP_logistics_EP_testtailored_3';
        mkdir(sprintf('~/working/neuroblastoma-data/data/systematic/cv/%s/testFolds/%d/%s', cvtype, cvfold, resultdirname));
        
        % Load and normalize data
        [train_inputs, train_outputs, test_inputs, test_outputs] = load_data(...
            '~/working/neuroblastoma-data/data/systematic',...
            ['~/working/neuroblastoma-data/data/systematic/cv/' cvtype],...
            cvfold,...
            true);
        
        % Select only the features above (see comment at top)
        
        % "log_log_bases" or "n_loglog" or "log2_n"
        selfeatures = find(cellfun(@(x) ismember(x, {'log_log_bases', 'n_loglog', 'log2_n'}), train_inputs.Properties.VariableNames));
        assert(numel(selfeatures) == 1, 'Feature "log2.n" not available.');

        % "diffAbs_identity_quantile_50_" or "log_sd" or "log_hall"
        selfeatures = [selfeatures, find(cellfun(@(x) ismember(x, {'diffAbs_identity_quantile_50_', 'log_sd', 'log_hall'}), train_inputs.Properties.VariableNames))];
        assert(numel(selfeatures) == 2, 'Feature "log.hall" not available.');
        
        % "log_mad"
        selfeatures = [selfeatures, find(cellfun(@(x) ismember(x, {'log_mad'}), train_inputs.Properties.VariableNames))];
        assert(numel(selfeatures) == 3, 'Feature "log.mad" not available.');

        normcols = selfeatures;
        [train_inputs, norms] = normalize_df(train_inputs, normcols);
        
        ncols = size(train_inputs, 2);
        
        % Apply same normalization to test_inputs
        test_inputs = normalize_df(test_inputs, norms);
        
        % Create the vectors of training data set        
        full_X = train_inputs{:, selfeatures};
        
        full_YL = train_outputs{:, 'min_log_lambda'};
        full_YH = train_outputs{:, 'max_log_lambda'};
        full_Y = RealInterval(full_YL, full_YH);
        [Ntrain, D] = size(full_X);
        
        test_X = test_inputs{:, selfeatures};
        test_YL = test_outputs{:, 'min_log_lambda'};
        test_YH = test_outputs{:, 'max_log_lambda'};
        Ntest = size(test_X, 1);
        
        testfold = 1;
        
        while testfold <= 5
            % Call experiment design function
            %     if testfold == 1
            %        Ninit = -2;
            %     else
            %        Ninit = 2;
            %     end
            Ninit = -2;
            % [IdxSelected, result_accuracy, result_predictions, result_covfunc, result_meanfunc, result_likfunc, n_seq] = experiment_design(full_X, full_Y, test_X, test_YL, test_YH, 60, Ninit, usetrainset);
            [IdxSelected, result_accuracy, result_predictions, result_covfunc, result_meanfunc, result_likfunc, n_seq] = experiment_design_for_test_set(full_X, full_Y, test_X, test_YL, test_YH, 60, Ninit);
            
            % Save the results
            fullresultdir = sprintf('~/working/neuroblastoma-data/data/systematic/cv/%s/testFolds/%d/%s/%d', cvtype, cvfold, resultdirname, testfold);
            mkdir(fullresultdir);
            
            writetable(train_inputs(IdxSelected, 'sequenceID'), [fullresultdir '/order.csv']);
            
            results = table(n_seq, result_accuracy(:), result_covfunc(:), result_meanfunc(:), result_likfunc(:), ...
                'VariableNames', {'n', 'accuracy', 'cov', 'mean', 'lik'});
            writetable(results, [fullresultdir '/gpaccuracy.csv']);
            
            results = table(test_inputs{:,'sequenceID'}, result_predictions{:}, ...
                'VariableNames', [{'sequenceID'}, arrayfun(@(n) sprintf('n%d', n), n_seq(:)', 'UniformOutput', false)]);
            writetable(results, [fullresultdir '/gppredictions.csv']);
            
            testfold = testfold + 1;
        end
    end
end