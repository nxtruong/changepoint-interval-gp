% Sample selection based on Gaussian Process model
clear all;
close all;

cvtypes = {'profileSize'}; % {'profileSize', 'profileID', 'chrom', 'sequenceID'};
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
    cvfolds = 3; %1:6;
    for cvfold = cvfolds
        usetrainset = false;
        
        % The directory name
        resultdirname = 'sampleSelectionGP_Erf_EP_testtailored';
        mkdir(sprintf('~/working/neuroblastoma-data/data/systematic/cv/%s/testFolds/%d/%s', cvtype, cvfold, resultdirname));
        
        % Load and normalize data
        [train_inputs, train_outputs, test_inputs, test_outputs] = load_data(...
            '~/working/neuroblastoma-data/data/systematic',...
            ['~/working/neuroblastoma-data/data/systematic/cv/' cvtype],...
            cvfold,...
            true);
        
        % Remove chr* columns from normalization
        normcols = cellfun(@(x) ~startsWith(x, 'chr') && ~strcmp(x, 'sequenceID'), train_inputs.Properties.VariableNames);
        normcols = train_inputs.Properties.VariableNames(normcols);
        [train_inputs, norms] = normalize_df(train_inputs, normcols);
        
        ncols = size(train_inputs, 2);
        
        % Apply same normalization to test_inputs
        test_inputs = normalize_df(test_inputs, norms);
        
        % Create the vectors of training data set
        % selfeatures = [2, 4, 6, 11, 36, 55] .+ 1
        selfeatures = find(cellfun(@(x) ~startsWith(x, 'rss_') && ~startsWith(x, 'mse_') && ~strcmp(x, 'sequenceID'), train_inputs.Properties.VariableNames));
        
        % The following removed features were gotten by training SEard
        % model with zero mean on 300 samples of random selection for
        % sequenceID, fold 1, random sequence 1
        rmfeatures = [1, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54];
        %rmfeatures = [];
        
        selfeatures(rmfeatures) = [];
        
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