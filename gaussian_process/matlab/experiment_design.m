function [IdxSelected, result_accuracy, result_predictions, result_covfunc, result_meanfunc, result_likfunc, n_seq] = experiment_design(full_X, full_Y, test_X, test_YL, test_YH, Ntrain, Ninithalf, usetrainset, infMethod)
    % usetrainset = true then the train set is used for assessing the
    % accuracy of the model in selecting model. The result_accuracy and
    % result_predictions are still on the test set.
    
    assert(abs(Ninithalf) >= 1, 'Invalid initial number of samples.');
    
    if ~exist('usetrainset', 'var') || isempty(usetrainset), usetrainset = false; end
    
    if ~exist('infMethod', 'var'), infMethod = @infEP; end

    [N, ~] = size(full_X);
    
    assert(Ntrain >= 2 && Ntrain <= N && Ntrain > 2*Ninithalf, 'Invalid Ntrain.');
    
    Ntest = size(test_X,1);

    % Select random initial samples
    % Find indices of samples with lower limit [YL is finite] and upper limit (YH is finite)
    full_Ylimit = [full_Y.l];
    idx_l = find(isfinite(full_Ylimit));
    if ~isempty(idx_l)
        [~, idx_lextreme] = max(full_Ylimit(idx_l));
    end
    
    full_Ylimit = [full_Y.h];
    idx_h = find(isfinite(full_Ylimit));
    if ~isempty(idx_h)
        [~, idx_hextreme] = min(full_Ylimit(idx_h));
    end
    
    % Select random initial samples
    IdxSelected = [];
    if Ninithalf < 0
        % First pair are extremes
        if ~isempty(idx_l)
            IdxSelected = [IdxSelected, idx_l(idx_lextreme)];
            idx_l(idx_lextreme) = [];
        end
        if ~isempty(idx_h)
            IdxSelected = [IdxSelected, idx_h(idx_hextreme)];
            idx_h(idx_hextreme) = [];
        end
        Ninithalf = -Ninithalf - 1;
    end
    
    for k = 1:Ninithalf
        if ~isempty(idx_l)
            idx = randi(length(idx_l), 1);
            IdxSelected = [IdxSelected, idx_l(idx)];
            idx_l(idx) = [];
        end
        if ~isempty(idx_h)
            idx = randi(length(idx_h), 1);
            IdxSelected = [IdxSelected, idx_h(idx)];
            idx_h(idx) = [];
        end
    end

    % All the remaining samples
    IdxRemaining = setdiff(1:N, IdxSelected);

    result_predictions = {};
    result_accuracy = [];
    
    result_covfunc = {};
    result_meanfunc = {};
    result_likfunc = {};
    
    N0 = length(IdxSelected);
    n_seq = reshape(N0:Ntrain, [], 1);

    % Loop: train, select next sample in the remainings that have the max variance
    for k = N0:Ntrain
        disp('====================================================');
        fprintf('Selection for %d number of samples...\n', (k+1));

        seltrain_X = full_X(IdxSelected,:);
        seltrain_Y = full_Y(IdxSelected);
        
        if usetrainset
            % [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, seltrain_X, [seltrain_Y.l]', [seltrain_Y.h]', 1, 30, infMethod);
            optimal_a = 0.1;
            [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, seltrain_X, [seltrain_Y.l]', [seltrain_Y.h]', [0.5, 0.2, optimal_a], [], infMethod);
        else
            % [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, 1, 30, infMethod);
            optimal_a = 0.1;
            [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, [0.5, 0.2, optimal_a], [], infMethod);
        end
        
        result_covfunc{end+1} = char(mlreportgen.utils.toString(covfunc));
        result_meanfunc{end+1} = char(mlreportgen.utils.toString(meanfunc));
        result_likfunc{end+1} = char(mlreportgen.utils.toString(likfunc));
        
        % Predict at test points
        [~, ~, Y_pred, ~] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, seltrain_X, seltrain_Y, test_X);
        result_predictions{end+1} = real(Y_pred(:));
        
        check_test = test_YL <= Y_pred & Y_pred <= test_YH;
        accuracy = sum(check_test) / Ntest * 100;
        result_accuracy(end+1) = accuracy;
        
        fprintf('- Model Accuracy: %g\n', accuracy);
            
        if k < Ntrain
            % Predict at remaining samples
            [~, ~, ~, pred_s2] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, seltrain_X, seltrain_Y, full_X(IdxRemaining,:));

            % Find sample with the max variance
            [~, idxmax] = max(pred_s2);
            idxnext = IdxRemaining(idxmax);

            % Store idxmax sample and remove it from IdxRemaining
            IdxSelected(end+1) = idxnext;
            IdxRemaining(idxmax) = [];

            fprintf('- Done! Selected sample %d.\n', idxnext);
        end
    end
end