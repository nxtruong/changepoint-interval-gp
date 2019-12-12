function [IdxSelected, result_accuracy, result_predictions, result_covfunc, result_meanfunc, result_likfunc, n_seq] = ...
        experiment_design_for_test_set(full_X, full_Y, test_X, test_YL, test_YH, Ntrain, Ninithalf, infMethod, likname, samplingmetric, euclideandist)
    % A special experiment design method that aims to maximize the
    % prediction accuracy on a specific test set.
    %
    % usetrainset = true then the train set is used for assessing the
    % accuracy of the model in selecting model. The result_accuracy and
    % result_predictions are still on the test set.
    %
    % samplingmetric is a function that takes arguments
    %       samplingmetric(s2train, s2test, distinv)
    % and returns sample selection metric values for all the training points;
    % where s2train is the vector of predicted variances of the training set,
    % s2test is the vector of predicted variances of the test set,
    % distinv is a matrix of pairwise inversed distance between a training
    % point (columns) and a test point (rows).  The distance metric is
    % either determined by the GP kernel or by a standard Euclidean
    % distance (see euclideandist below).
    %
    % euclideandist = false (default) if the kernel function will be used
    % to calculate the distance between a training input and a test input;
    % true if standard Euclidean distance is used.
    
    assert(abs(Ninithalf) >= 1, 'Invalid initial number of samples.');
    
    if ~exist('infMethod', 'var'), infMethod = @infEP; end
    if ~exist('likname', 'var')
        likname = 'erf';
    else
        likname = lower(likname);
    end

    [N, ~] = size(full_X);
    
    assert(Ntrain >= 2 && Ntrain <= N && Ntrain > 2*Ninithalf, 'Invalid Ntrain.');
    
    Ntest = size(test_X,1);
    
    % If no 'samplingmetric' is given, use the default which is
    % distmin-full
    if ~exist('samplingmetric', 'var') || isempty(samplingmetric)
        samplingmetric = @(s2, distinv) max(distinv);
    end
    
    % Default euclideandist is false
    if ~exist('euclideandist', 'var') || isempty(euclideandist)
        euclideandist = false;
    end

    % Select random initial samples
    if isnumeric(full_Y)
        IdxSelected = [];
        
        if Ninithalf < 0
            % First pair are extremes
            [~,idx] = min(full_Y);
            IdxSelected = [IdxSelected, idx];
            
            [~,idx] = max(full_Y);
            IdxSelected = [IdxSelected, idx];
            
            Ninithalf = -Ninithalf - 1;
            
            % All the remaining samples
            IdxRemaining = setdiff(1:N, IdxSelected);
        else
            IdxRemaining = 1:N;
        end
        
        idx = randperm(length(IdxRemaining), Ninithalf*2);
        IdxSelected = [IdxSelected, IdxRemaining(idx)];
        IdxRemaining(idx) = [];
    else
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
    end

    result_predictions = {};
    result_accuracy = [];
    
    result_covfunc = {};
    result_meanfunc = {};
    result_likfunc = {};
    
    N0 = length(IdxSelected);
    n_seq = reshape(N0:Ntrain, [], 1);

    % Loop: train, select next sample in the remainings
    for k = N0:Ntrain
        disp('====================================================');
        fprintf('Selection for %d number of samples...\n', (k+1));

        seltrain_X = full_X(IdxSelected,:);
        seltrain_Y = full_Y(IdxSelected);

        % Train the best GP model on selected training samples
        % [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, 1, 30, infMethod, likname);
        
        switch likname
            case 'erf'
                % For Erf
                [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, [0.01, 0.05, 0.1, 0.2, 0.3], [], infMethod, 'erf');
                
            case 'logistic'
                % For logistic
                [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, [10, 15, 20, 25, 30], [], infMethod, 'logistic');
        end
        
        result_covfunc{end+1} = char(mlreportgen.utils.toString(covfunc));
        result_meanfunc{end+1} = char(mlreportgen.utils.toString(meanfunc));
        result_likfunc{end+1} = char(mlreportgen.utils.toString(likfunc));
        
        % Predict at test points
        hasproblem = false;
        try
            [~, ~, Y_pred, S2_pred] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, seltrain_X, seltrain_Y, test_X);
        catch
            hasproblem = true;
        end
        
        if hasproblem
            disp('- Numerical problems with the GP model.');
            result_predictions{end+1} = nan(size(test_X,1),1);
            result_accuracy(end+1) = nan;
        else
            Y_pred = real(Y_pred(:));
            result_predictions{end+1} = Y_pred;

            if any(isnan(Y_pred))
                accuracy = nan;
            else
                check_test = test_YL <= Y_pred & Y_pred <= test_YH;
                accuracy = sum(check_test) / Ntest * 100;
            end
            result_accuracy(end+1) = accuracy;
            
            fprintf('- Model Accuracy: %g\n', accuracy);
        end
            
        if k < Ntrain
            % The selection of the next training sample is tailored to give
            % hopefully the best prediction on the test data set. First,
            % identify a test point with false prediction and with largest
            % variance; if accuracy is 100% then select the one with the
            % largest variance. Then, among the remaining training samples,
            % select a sample that is closest to the test point, w.r.t the
            % current kernel-induced metric (i.e., the value of k() is
            % highest).
            
            if hasproblem
                % Select a random next point
                idxrand = randi(numel(IdxRemaining), 1);
                idxnext = IdxRemaining(idxrand);
                IdxRemaining(idxrand) = [];
                
            else
                %{
                % Find the worst test point: it's the point with the
                % largest predicted variance among all failed points (or if
                % there are no failed points, among all test points).
                if accuracy < 100
                    failed_test_pnts = find(~check_test);
                    [~,worst_test_pnt] = max(S2_pred(failed_test_pnts));
                    worst_test_pnt = failed_test_pnts(worst_test_pnt);
                else
                    [~,worst_test_pnt] = max(S2_pred);
                end
                %}
                
                % Calculate the distance from each remaining training point
                % to each test point.
                % - if euclideandist = false, the distance metric is based
                % on the kernel function. The distance is reciprocal to the
                % kernel value between two points.  So the min distance is
                % the max kernel value.
                % - otherwise, the distance metric is the standard
                % Euclidean squared distance between two vectors.
                
                if euclideandist
                    distinv = 1./euclidean_distance(test_X, full_X(IdxRemaining,:));
                else
                    % Calculate the max kernel value between each remaining training
                    % sample and the test points
                    if iscell(covfunc)
                        distinv = feval(covfunc{1}, covfunc{2:end}, hyp.cov, test_X, full_X(IdxRemaining,:));
                    else
                        distinv = feval(covfunc, hyp.cov, test_X, full_X(IdxRemaining,:));
                    end
                end
                
                % Predict at remaining training samples, to get variance
                try
                    [~, ~, ~, s2train] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, seltrain_X, seltrain_Y, full_X(IdxRemaining,:));
                catch
                    hasproblem = true;
                end
                
                % If there is any numerical problem, the variances will be
                % 1 for all training samples
                if hasproblem
                    s2train = ones(1, numel(IdxRemaining));
                else
                    s2train = reshape(s2train, 1, []);  % make sure row vector
                end
                
                % Calculate the sample selection metric for each remaining
                % training sample, based on: the predicted variance and the
                % inversed distance between each training point and each
                % test point
                sample_metric = feval(samplingmetric, s2train, S2_pred, distinv);
                
                % Choose the point with the max metric value
                [~, idxmax] = max(sample_metric);
                
                % Find sample of that point
                idxnext = IdxRemaining(idxmax);
                
                % Store idxmax sample and remove it from IdxRemaining
                IdxRemaining(idxmax) = [];
            end
            
            IdxSelected(end+1) = idxnext;
            fprintf('- Done! Selected sample %d.\n', idxnext);
        end
    end
end

function D2 = euclidean_distance(x,z)
    % Code similar to GPML's Mahalanobis distance computation
    n = size(x,1); m = size(z,1);
    mu = (m/(n+m))*mean(z,1) + (n/(n+m))*mean(x,1); z = bsxfun(@minus,z,mu);
    x = bsxfun(@minus,x,mu);
    sax = sum(x.*x,2);
    saz = sum(z.*z,2);
    D2 = max(bsxfun(@plus,sax,bsxfun(@minus,saz',2*x*z')),0);     % computation
end