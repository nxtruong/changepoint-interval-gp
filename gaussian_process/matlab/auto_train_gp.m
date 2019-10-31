function [meanfunc, covfunc, likfunc, hyp, accuracy, optimal_a] = auto_train_gp(seltrain_X, seltrain_Y, test_X, test_YL, test_YH, amin, amax, infMethod, likname)
    %AUTO_TRAIN_GP Try different options to train the best GP model
    % Can either provide a range of a : [amin, amax] to search for a
    % Or provide a specific a or several values of a in amin (amax must be
    % omitted).
    
    if ~exist('infMethod', 'var'), infMethod = @infEP; end
    if ~exist('likname', 'var'), likname = 'erf'; end
    
    Ntest = size(test_X, 1);
    D = size(seltrain_X, 2);
    
    if isnumeric(seltrain_Y)
        [meanfunc, covfunc, likfunc, hyp, accuracy] = train_for_lik(@likGauss, 0);
        optimal_a = [];
        return;
    end
    
    if ~exist('amax', 'var') || isempty(amax)
        assert(all(amin > 0));
        avalues = amin;
        accuracy = -Inf;
    else
        assert(amax > amin);
        
        % Train with Full likelihood to find optimal a as well
        likfunc_k = {@likIntervalLogisticFull, [amin, amax]};
        [meanfunc, covfunc, likfunc, hyp, accuracy] = train_for_lik(likfunc_k, 0);
        
        % Get the optimal a
        optimal_a = (amax-amin)/(1+exp(-hyp.lik)) + amin;
        avalues = optimal_a;
    end
    
    for ka = 1:numel(avalues)
        switch likname
            case 'erf'
                likfunc_k = {@likIntervalErf, avalues(ka)};
            case 'logistic'
                likfunc_k = {@likIntervalLogistic, avalues(ka)};
        end
        try
            [meanfunc2, covfunc2, likfunc2, hyp2, accuracy2] = train_for_lik(likfunc_k, []);
        catch
            continue;
        end
        
        if accuracy2 >= accuracy
            meanfunc = meanfunc2;
            covfunc = covfunc2;
            likfunc = likfunc2;
            hyp = hyp2;
            accuracy = accuracy2;
            optimal_a = avalues(ka);
        end
    end
    
    
    function [meanfunc, covfunc, likfunc, hyp, accuracy] = train_for_lik(likfunc_k, hyplik)
        % Train for a specific likelihood function
        accuracy = 0;
        meanfunc = [];
        covfunc = [];
        likfunc = [];
        hyp = struct;
        
        hyp_k = struct;
        hyp_k.lik = hyplik;
        
        allcovs = {@covSEard};  % @covRQard, {@covMaternard, 5}
        allcovdefaults = {zeros(1, D+1)};  % zeros(1, D+2) (if using covRQard); zeros(1, D+1) if using covMaternard
        ncovs = numel(allcovs);
        
        for kcov = 1:ncovs
            covfunc_k = allcovs{kcov};
            hypcov_k_default = allcovdefaults{kcov};
            
            hyp_k.cov = hypcov_k_default;
            hyp_k.mean = [];
            
            allmeans = {@meanZero, {@meanSum, {@meanLinear, @meanConst}}};  % @meanConst, 
            allmeannhyps = [0, 1+D];  % 1 (if using meanConst)
            nmeans = numel(allmeans);
            
            for kmean = 1:nmeans
                meanfunc_k = allmeans{kmean};
                
                % Add 0's to mean hyps if necessary
                if allmeannhyps(kmean) > numel(hyp_k.mean)
                    hyp_k.mean = [hyp_k.mean; zeros(allmeannhyps(kmean) - numel(hyp_k.mean), 1)];
                end
                
                hasproblem = false;
                
                % Optimize
                try
                    hyp_k = minimize(hyp_k, @gp, 10000, infMethod, meanfunc_k, covfunc_k, likfunc_k, seltrain_X, seltrain_Y);
                catch
                    hasproblem = true;
                end
                if hasproblem, continue; end
                
                try
                    [nlZ, dnlZ] = gp(hyp_k, infMethod, meanfunc_k, covfunc_k, likfunc_k, seltrain_X, seltrain_Y);
                catch
                    hasproblem = true;
                end
                if hasproblem, continue; end
                
                if ~isfinite(nlZ) || ~all(isfinite(dnlZ.mean)) || ~all(isfinite(dnlZ.cov))
                    % Optimized hyperparameters seem to have issues
                    % Rest hyperparameters
                    hyp_k = struct;
                    hyp_k.mean = zeros(allmeannhyps(kmean), 1);  % Max number of mean hyperparameters
                    hyp_k.cov = hypcov_k_default;
                    hyp_k.lik = hyplik;
                    
                    continue;
                end
                
                % Test
                try
                    [~, ~, pred_Y, ~] = gp(hyp_k, infMethod, meanfunc_k, covfunc_k, likfunc_k, seltrain_X, seltrain_Y, test_X);
                catch
                    hasproblem = true;
                end
                if hasproblem, continue; end
                
                pred_Y = real(pred_Y(:));
                if any(isnan(pred_Y))
                    continue;
                else
                    check_test = test_YL <= pred_Y & pred_Y <= test_YH;
                    check_test_accuracy = sum(check_test) / Ntest * 100;
                    
                    if check_test_accuracy > accuracy
                        % Save the best
                        meanfunc = meanfunc_k;
                        covfunc = covfunc_k;
                        likfunc = likfunc_k;
                        hyp = hyp_k;
                        accuracy = check_test_accuracy;
                    end
                end
            end
        end
    end
end

