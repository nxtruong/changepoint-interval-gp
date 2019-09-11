clear all;
close all;

infMethod = @infEP;

for cvtypec = {'sequenceID'} % , 'profileSize', 'profileID', 'chrom'}
    cvtype = cvtypec{1};
    
    for cvfold = 1:6
        % Select the fold (1 to 6)
        
        for testfolds = 1:5
            % Select the random test fold order (1 to 5)
            
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
            
            % Load the random selection order
            sample_order = readtable(...
                sprintf('~/working/neuroblastoma-data/data/systematic/cv/%s/testFolds/%d/randomTrainOrderings/%d/order.csv', cvtype, cvfold, testfolds),...
                'ReadVariableNames', true, 'Delimiter', 'comma');
            
            % Create the vectors of training data set
            % selfeatures = [2, 4, 6, 11, 36, 55] .+ 1
            selfeatures = find(cellfun(@(x) ~startsWith(x, 'rss_') && ~startsWith(x, 'mse_') && ~strcmp(x, 'sequenceID'), train_inputs.Properties.VariableNames));
            
            % The following removed features were gotten by training SEard
            % model with zero mean on 300 samples of random selection for
            % sequenceID, fold 1, random sequence 1
            rmfeatures = [1, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54];
            %rmfeatures = [];
            
            selfeatures(rmfeatures) = [];
            
            % Use the same order as in sample_order
            train_seqs = train_inputs{:, 'sequenceID'};
            sample_idx = cellfun(@(x) find(strcmp(x, train_seqs),1), sample_order{:,1});
            
            train_X = train_inputs{sample_idx, selfeatures};
            
            train_YL = train_outputs{sample_idx, 'min_log_lambda'};
            train_YH = train_outputs{sample_idx, 'max_log_lambda'};
            train_Y = RealInterval(train_YL,train_YH);
            [Ntrain, D] = size(train_X);
            
            test_X = test_inputs{:, selfeatures};
            test_YL = test_outputs{:, 'min_log_lambda'};
            test_YH = test_outputs{:, 'max_log_lambda'};
            Ntest = size(test_X, 1);
            
            % Train and test
            % Sequence of number of training samples taken
            nbr_trainings = 10:10:60;
            model_accuracies = [];
            
            for Nsel = nbr_trainings
                disp('====================================================');
                fprintf('For CV %s, fold %d, test %d, %d samples...\n', cvtype, cvfold, testfolds, Nsel);
                
                seltrain_X = train_X(1:Nsel,:);
                seltrain_Y = train_Y(1:Nsel);
                
                [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, 15, [], infMethod);
                
                fprintf('Model Accuracy: %g\n', accuracy);
                
                % Code to find large hyperparameters to remove
                %ardhyps = hyp.cov(1:end-1);
                %rmfeatures = find(ardhyps > min(ardhyps)+log(30))
                
                model_accuracies = [model_accuracies, accuracy];
            end
            
            results = table(reshape(nbr_trainings(1:length(model_accuracies)), [], 1), model_accuracies(:), 'VariableNames', {'trainsize', 'accuracy'});
            writetable(results, ...
                sprintf('~/working/neuroblastoma-data/data/systematic/cv/%s/testFolds/%d/randomTrainOrderings/%d/gpmodel_EP.csv', cvtype, cvfold, testfolds));
        end
    end
end