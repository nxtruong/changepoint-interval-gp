function [IdxSelected, result_accuracy, result_predictions, result_covfunc, result_meanfunc, result_likfunc, n_seq] = ...
        experiment_design_for_test_set(full_X, full_Y, test_X, test_YL, test_YH, Ntrain, Ninithalf, infMethod)
    % A special experiment design method that aims to maximize the
    % prediction accuracy on a specific test set.
    %
    % usetrainset = true then the train set is used for assessing the
    % accuracy of the model in selecting model. The result_accuracy and
    % result_predictions are still on the test set.
    
    assert(abs(Ninithalf) >= 1, 'Invalid initial number of samples.');
    
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

        % Train the best GP model on selected training samples
        % [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, 1, 30, infMethod);
        
        % For Erf
        %optimal_a = 0.1;
        %[meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, [0.05, 0.25, optimal_a], [], infMethod);
        
        % For logistic
        optimal_a = 15;
        [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, [10, 20, optimal_a], [], infMethod);
        
        result_covfunc{end+1} = char(mlreportgen.utils.toString(covfunc));
        result_meanfunc{end+1} = char(mlreportgen.utils.toString(meanfunc));
        result_likfunc{end+1} = char(mlreportgen.utils.toString(likfunc));
        
        % Predict at test points
        [~, ~, Y_pred, S2_pred] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, seltrain_X, seltrain_Y, test_X);
        result_predictions{end+1} = real(Y_pred(:));
        
        check_test = test_YL <= Y_pred & Y_pred <= test_YH;
        accuracy = sum(check_test) / Ntest * 100;
        result_accuracy(end+1) = accuracy;
        
        fprintf('- Model Accuracy: %g\n', accuracy);
            
        if k < Ntrain
            % The selection of the next training sample is tailored to give
            % hopefully the best prediction on the test data set. First,
            % identify a test point with false prediction and with largest
            % variance; if accuracy is 100% then select the one with the
            % largest variance. Then, among the remaining training samples,
            % select a sample that is closest to the test point, w.r.t the
            % current kernel-induced metric (i.e., the value of k() is
            % highest).
            
            if accuracy < 100
                failed_test_pnts = find(~check_test);
                [~,worst_test_pnt] = max(S2_pred(failed_test_pnts));
                worst_test_pnt = failed_test_pnts(worst_test_pnt);
            else
                [~,worst_test_pnt] = max(S2_pred);
            end
            
            % Calculate the kernel value between all remaining training
            % samples and the worst test point
            if iscell(covfunc)
                kern_test_pnt = feval(covfunc{1}, covfunc{2:end}, hyp.cov, test_X(worst_test_pnt,:), full_X(IdxRemaining,:));
            else
                kern_test_pnt = feval(covfunc, hyp.cov, test_X(worst_test_pnt,:), full_X(IdxRemaining,:));
            end
            
            % Find the closest point (largest kernel value)
            [~, idxmax] = max(kern_test_pnt);
            
            % Find sample of that point
            idxnext = IdxRemaining(idxmax);

            % Store idxmax sample and remove it from IdxRemaining
            IdxSelected(end+1) = idxnext;
            IdxRemaining(idxmax) = [];

            fprintf('- Done! Selected sample %d.\n', idxnext);
        end
    end
end