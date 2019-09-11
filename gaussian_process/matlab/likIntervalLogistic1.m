function [varargout] = likIntervalLogistic1(a, hyp, y, mu, s2, infMethod, i)
% 
%     Logistic-like Likelihood for Intervals
%
% This function is similar to likIntervalLogistic.m, but it reuses the
% function likLogistic when infMethod = 'infEP'. However, tests showed that
% its calculations, especially for dlZ and d2lZ are not consistent compared
% with numerical derivatives. While likIntervalLogistic is more consistent
% though less accurate in lZ at the tails (outside the interval). Therefore
% likIntervalLogistic() should be used instead.
% 
% Given an interval `[l,h]` the likelihood is calculated by the logistic function:
%     `p(y = [l,h] | f) = logistic(a*(f-l)) - logistic(a*(f-h))`
% where `a` is a parameter, which is often large.
% Special cases can easily be derived for when `l = -inf` and when `h = inf` but at
% least one of them must be finite. When l is infinite, logistic(l) = 1;
% when h is infinite, logistic(h) = 0.
% 
% `a` is chosen so that:
% - At `f=l+x` and `f=h-x`, `p` is large enough, close to 1.
% - And of course `x` should be small enough, much less than `(h-l)/2` or the radius
%   of the interval.
% 
% We can show that `a = log(p/(1-p))/x` where `x <= (h-l)/2` (should be much smaller
% than that) and `p` is for instance 0.95 or 0.99.
%
% Several modes are provided, for computing likelihoods, derivatives and moments
% respectively, see likFunctions.m for the details. In general, care is taken
% to avoid numerical issues when the arguments are extreme. The moments
% \int f^k likLogistic(y,f) N(f|mu,var) df are calculated via a cumulative 
% Gaussian scale mixture approximation.
%
% (C) 2019 by Truong X. Nghiem.
%
% See also LIKFUNCTIONS.M.

if nargin<4, varargout = {'0'}; return; end   % report number of hyperparameters

% a = exp(hyp);
% a = 15;

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
    s2zero = ~(nargin>4&&numel(s2)>0&&norm(s2)>eps);  % s2==0 ?
    if s2zero                                         % log probability evaluation
        lp = logLogisticDiff(a*(mu-l), a*(mu-h));           % log of likelihood
    else                                                              % prediction
        lp = likIntervalLogistic(a, hyp, y, mu, s2, 'infLaplace');
    end
    ymu = {}; ys2 = {};
    if nargout>1
        p = exp(lp);
        ymu = 2*p-1;                                                % first y moment
        if nargout>2
            ys2 = 4*p.*(1-p);                                        % second y moment
        end
    end
    varargout = {lp,ymu,ys2};
else                                                            % inference mode
    switch infMethod
        case 'infLaplace'
            if nargin<7                                             % no derivative mode
                f = mu;
                lp = logLogisticDiff(a*(f-l), a*(f-h));           % log of likelihood
                dlp = {}; d2lp = {}; d3lp = {};                         % return arguments
                if nargout>1                                           % first derivatives
                    dlpl = zeros(N,1);
                    ok = isfinite(l);
                    dlpl(ok) = logistic(a*(l(ok)-f(ok)));
                    
                    dlph = zeros(N,1);
                    ok = isfinite(h);
                    dlph(ok) = logistic(a*(f(ok)-h(ok)));
                    
                    dlp = a * (dlpl - dlph);        % derivative of log likelihood
                    
                    if nargout>2                          % 2nd derivative of log likelihood
                        d2lpl = dlpl.*(1-dlpl);  % dlpl - dlpl.^2;
                        d2lph = dlph.*(1-dlph);  % dlph - dlph.^2;
                        d2lp = -a^2 * (d2lpl + d2lph);
                        if nargout>3                        % 3rd derivative of log likelihood
                            d3lp = a^3 * (d2lpl.*(1-2*dlpl) + d2lph.*(2*dlph-1));
                        end
                    end
                end
                varargout = {lp,dlp,d2lp,d3lp};
            else                                                    % derivative mode
                varargout = {[],[],[]};                         % derivative w.r.t. hypers
            end
        
        case 'infEP'
            if nargin<7                                             % no derivative mode
                % We will use likLogistic to calculate
                % Z = likLogistic(a*(t-l)) - likLogistic(a*(t-h))
                % Consider likLogistic(a*(t-l))
                % By changing the variable to use likLogistic, we can
                % calculate lZ_l = lZl, dlZ_l = a*dlZl, d2lZ_l = a^2*d2lZl
                % by calling [lZl, dlZl, d2lZl] = likLogistic([], 1,
                % a*(mu-l), (a*sigma)^2, 'infEP')
                % Similarly for likLogistic(a*(t-h))
                % Then from these we can calculate lZ, dlZ, d2lZ using
                % arithmetic and basic calculus (note the change of
                % variables).
                
                % Because the bounds of the intervals can be Inf, we need
                % to take special care about them:
                % + If l = -Inf then its lZ_l = 0 (Z_l = 1), dlZ_l = 0,
                % d2lZ_l = 0.
                % + If h = Inf then its lZ_h = -Inf (Z_h = 0), dlZ_h = 0,
                % d2lZ_h = 0.
                
                s2_a = a*a*s2;
                
                % For lower bounds
                finIdx = find(isfinite(l));
                lZ_l = zeros(N,1);
                dlZ_l = lZ_l;
                d2lZ_l = lZ_l;
                if ~isempty(finIdx)
                    [lZ_l(finIdx), dlZ_l(finIdx), d2lZ_l(finIdx)] = likLogistic([], 1, a*(mu(finIdx)-l(finIdx)), s2_a(finIdx), 'infEP');
                    dlZ_l(finIdx) = dlZ_l(finIdx) * a;
                    d2lZ_l(finIdx) = d2lZ_l(finIdx) * (a*a);
                end
                
                % For upper bounds
                finIdx = find(isfinite(h));
                lZ_h = -inf(N,1);
                dlZ_h = zeros(N,1);
                d2lZ_h = dlZ_h;
                if ~isempty(finIdx)
                    [lZ_h(finIdx), dlZ_h(finIdx), d2lZ_h(finIdx)] = likLogistic([], 1, a*(mu(finIdx)-h(finIdx)), s2_a(finIdx), 'infEP');
                    dlZ_h(finIdx) = dlZ_h(finIdx) * a;
                    d2lZ_h(finIdx) = d2lZ_h(finIdx) * (a*a);
                end
                
                % Calculate final values from the above
                Z_l = exp(lZ_l);
                Z_h = exp(lZ_h);
                Zinv = 1./(Z_l-Z_h);
                
                lZ = log(Z_l - Z_h);
                
                % dlZ = dlZ_l + exp(lZ_h - lZ) .* (dlZ_l - dlZ_h);  % Equivalent to below
                dlZ = Zinv .* (dlZ_l.*Z_l - dlZ_h.*Z_h);
                
                d2lZ = Zinv .* (Z_l.*(d2lZ_l + dlZ_l.^2) - Z_h.*(d2lZ_h + dlZ_h.^2)) - dlZ.^2;
                                
                varargout = {lZ,dlZ,d2lZ};
            else                                                       % derivative mode
                varargout = {[]};                                     % deriv. wrt hyp.lik
            end
        otherwise
            error('Unsupported infererence method: %s', infMethod);
    end
end
end

function r = log1mexp(x)
    % log(1 - exp(x)) accurately
    if x < -0.6931471805599453094
        r = log1p(-exp(x));
    else
        r = log(-expm1(x));
    end
end

function r = logLogisticDiff(a, b)
    %Calculates log(logistic(a) - logistic(b))
    %where a > b.  The computation avoids numerical inaccuracy when a, b
    %are extreme (very small or very large).
    assert(all(a > b))
    
    r1 = a;
    ok = a > -35;
    r1(ok) = -log1p(exp(-a(ok)));

    r2 = -b;
    ok = b < 35;
    r2(ok) = -log1p(exp(b(ok)));
    
    r = log1mexp(b-a) + r1 + r2;
end

function p = logistic(f)
    % Calculate logistic function r = 1./(1 + exp(-f))
    p = 1./(1 + exp(-f));
end