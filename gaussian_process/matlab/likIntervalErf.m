function [varargout] = likIntervalErf(sigma, hyp, y, mu, s2, infMethod, i)
    %
    %     NORMCDF-based Likelihood for Intervals
    %
    % Given an interval `[l,h]` the likelihood is calculated by the logistic function:
    %     `p(y = [l,h] | f) = normcdf((f-l)/sigma) - normcdf((f-h)/sigma)`
    % where `sigma` is a parameter (often small) and normcdf is the standard Normal cdf.
    % Special cases can easily be derived for when `l = -inf` and when `h = inf` but at
    % least one of them must be finite. When l is infinite, normcdf((f-l)/sigma) = 1;
    % when h is infinite, normcdf((f-h)/sigma) = 0.
    %
    % sigma should be chosen so that (h-l) > 6*sigma.
    %
    % Several modes are provided, for computing likelihoods, derivatives and moments
    % respectively, see likFunctions.m for the details. In general, care is taken
    % to avoid numerical issues when the arguments are extreme. The moments
    % \int f^k likLogistic(y,f) N(f|mu,var) df are calculated via a cumulative
    % Gaussian scale mixture approximation.
    %
    % (C) 2019 by Truong X. Nghiem.
    %
    % See also LIKFUNCTIONS.M, LIKERF.M
    
    if nargin<4, varargout = {'0'}; return; end   % report number of hyperparameters
    
    sigma2 = sigma^2;
    
    % When y is empty, GPML tries to predict the output from latent function
    % But in this case, we can't / don't need it, so we just use a default
    % interval. Do not use the returned output predictions.
    if isempty(y)
        N = length(mu);
        l = zeros(N,1);
        h = inf(N,1);
    else
        l = [y.l]';
        h = [y.h]';
        N = length(l);
        
        if isscalar(mu)
            mu = repmat(mu, N, 1);
        end
    end
    
    if nargin<6                              % prediction mode if inf is not present
        s2zero = ~(nargin>3&&numel(s2)>0&&norm(s2)>eps);     % s2==0 ?
        
        if s2zero                                         % log probability evaluation
            lp = loglikelihood(sigma, mu-l, mu-h);
        else                                                              % prediction
            lp = likIntervalErf(sigma, hyp, y, mu, s2, 'infEP');
        end
        p = exp(lp); ymu = {}; ys2 = {};
        if nargout>1
            ymu = 2*p-1;                                                % first y moment
            if nargout>2
                ys2 = 4*p.*(1-p);                                        % second y moment
            end
        end
        varargout = {lp,ymu,ys2};
    else                                                            % inference mode
        switch infMethod
            case 'infLaplace'
                error('Not yet implemented.');
                if nargin<7                                             % no derivative mode
                    f = mu; yf = y.*f;                            % product latents and labels
                    varargout = cell(nargout,1); [varargout{:}] = logphi(yf);   % query logphi
                    if nargout>1
                        varargout{2} = y.*varargout{2};
                        if nargout>3, varargout{4} = y.*varargout{4}; end
                    end
                else                                                       % derivative mode
                    varargout = {[],[],[]};                         % derivative w.r.t. hypers
                end
                
            case 'infEP'
                if nargin<7                                             % no derivative mode
                    lZ_l = zeros(size(mu));
                    lZ_h = -inf(size(mu));
                    
                    Z_l = ones(size(mu));
                    Z_h = lZ_l;
                    
                    lZ = {}; dlZ = {}; d2lZ = {};
                    
                    finIdx_l = find(isfinite(l));
                    finIdx_h = find(isfinite(h));
                    
                    if nargout == 1
                        if ~isempty(finIdx_l)
                            lZ_l(finIdx_l) = infEPcalc((mu(finIdx_l)-l(finIdx_l))/sigma, s2(finIdx_l)/sigma2);
                            Z_l(finIdx_l) = exp(lZ_l(finIdx_l));
                        end
                        
                        if ~isempty(finIdx_h)
                            lZ_h(finIdx_h) = infEPcalc((mu(finIdx_h)-h(finIdx_h))/sigma, s2(finIdx_h)/sigma2);
                            Z_h(finIdx_h) = exp(lZ_h(finIdx_h));
                        end
                    
                        lZ = log(Z_l - Z_h);
                        
                    elseif nargout > 1
                        dlZ_l = zeros(size(mu));
                        dlZ_h = dlZ_l;
                        
                        if nargout > 2
                            d2lZ_l = dlZ_l;
                            d2lZ_h = dlZ_l;
                        end
                        
                        if ~isempty(finIdx_l)
                            if nargout > 2
                                [lZ_l(finIdx_l), dlZ_l(finIdx_l), d2lZ_l(finIdx_l)] = infEPcalc((mu(finIdx_l)-l(finIdx_l))/sigma, s2(finIdx_l)/sigma2);
                                d2lZ_l(finIdx_l) = d2lZ_l(finIdx_l) / sigma2;
                            else
                                [lZ_l(finIdx_l), dlZ_l(finIdx_l)] = infEPcalc((mu(finIdx_l)-l(finIdx_l))/sigma, s2(finIdx_l)/sigma2);
                            end
                            Z_l(finIdx_l) = exp(lZ_l(finIdx_l));
                            dlZ_l(finIdx_l) = dlZ_l(finIdx_l) / sigma;
                        end
                        
                        if ~isempty(finIdx_h)
                            if nargout > 2
                                [lZ_h(finIdx_h), dlZ_h(finIdx_h), d2lZ_h(finIdx_h)] = infEPcalc((mu(finIdx_h)-h(finIdx_h))/sigma, s2(finIdx_h)/sigma2);
                                d2lZ_h(finIdx_h) = d2lZ_h(finIdx_h) / sigma2;
                            else
                                [lZ_h(finIdx_h), dlZ_h(finIdx_h)] = infEPcalc((mu(finIdx_h)-h(finIdx_h))/sigma, s2(finIdx_h)/sigma2);
                            end
                            Z_h(finIdx_h) = exp(lZ_h(finIdx_h));
                            dlZ_h(finIdx_h) = dlZ_h(finIdx_h) / sigma;
                        end
                    
                        Zinv = 1./(Z_l-Z_h);
                        lZ = -log(Zinv);
                        dlZ = Zinv .* (dlZ_l.*Z_l - dlZ_h.*Z_h);
                        if nargout > 2
                            d2lZ = Zinv .* (Z_l.*(d2lZ_l + dlZ_l.^2) - Z_h.*(d2lZ_h + dlZ_h.^2)) - dlZ.^2;
                        end
                    end
                    
                    varargout = {lZ,dlZ,d2lZ};
                else                                                       % derivative mode
                    varargout = {[]};                                     % deriv. wrt hyp.lik
                end
                
            case 'infVB'
                error('infVB not supported');
        end
    end
end

function lp = loglikelihood(sigma, xl, xh)
    % Calculate log of (normcdf((f-l)/sigma) - normcdf((f-h)/sigma))
    % taking into account corner cases.
    lp = zeros(size(xl));
    
    % When l = -inf, likelihood = 1-Phi(xh) = Phi(-xh) so we can use logcdf
    infl = isinf(xl);
    lp(infl) = logphi(-xh(infl) / sigma);
    
    % When h = +inf, likelihood = Phi(xl) so we can use logcdf
    infh = isinf(xh);
    lp(infh) = logphi(xl(infh) / sigma);
    
    % When both are finite, this is tricky, especially when l and h are close, resulting in likelihood = 0
    finlh = ~(infl | infh);
    if any(finlh)
        xl = xl(finlh);
        xh = xh(finlh);
        F = normcdf(xl / sigma) - normcdf(xh / sigma);
        % When F is zero, likelihood is the area below the normal pdf from l to h
        % We approximate it by (pdf(xl) + pdf(xh))*(h-l)/2, and take its log
        Fzero = F <= 0;
        F(Fzero) = log(normpdf(xl(Fzero) / sigma) + normpdf(xh(Fzero) / sigma)) + log(xl(Fzero)-xh(Fzero)) - log(2);
        F(~Fzero) = log(F(~Fzero));
        
        lp(finlh) = F;
    end
end

function varargout = infEPcalc(mu, s2)
    % Calculate the [lZ, dlZ, d2lZ] required for infEP for likelihood
    % likErf. These will be used to calculate the values for likelihood
    % likIntervalErf.
    z = mu./sqrt(1+s2); dlZ = {}; d2lZ = {};
    if nargout<=1
        lZ = logphi(z);                         % log part function
    else
        [lZ,n_p] = logphi(z);
    end
    if nargout>1
        dlZ = n_p./sqrt(1+s2);                      % 1st derivative wrt mean
        if nargout>2
            d2lZ = -n_p.*(z+n_p)./(1+s2);         % 2nd derivative
        end
    end
    varargout = {lZ,dlZ,d2lZ};
end