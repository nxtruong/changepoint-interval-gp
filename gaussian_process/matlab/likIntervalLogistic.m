function [varargout] = likIntervalLogistic(a, hyp, y, mu, s2, infMethod, i)
% 
%     Logistic-like Likelihood for Intervals
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
                f = ones(size(mu));                                   % make y a vector
                % likLogistic(t) \approx 1/2 + \sum_{i=1}^5 (c_i/2) erf(lam_i/sqrt(2)t)
                lam = sqrt(2)*[0.44 0.41 0.40 0.39 0.36];    % approx coeffs lam_i and c_i
                c = [1.146480988574439e+02; -1.508871030070582e+03; 2.676085036831241e+03;
                    -1.356294962039222e+03;  7.543285642111850e+01                      ];
                fones = f*ones(1,5);
                mu_l = a*(mu-l);
                mu_h = a*(mu-h);
                s2_a = a*a*s2;
                
                finIdx = find(isfinite(l));
                lZc_l = zeros(size(fones));
                dlZc_l = lZc_l;
                d2lZc_l = lZc_l;
                if ~isempty(finIdx)
                    [lZc_l(finIdx,:),dlZc_l(finIdx,:),d2lZc_l(finIdx,:)] = likErf([], fones(finIdx,:), mu_l(finIdx)*lam, s2_a(finIdx)*(lam.^2), infMethod);
                end
                
                finIdx = find(isfinite(h));
                lZc_h = -inf(size(fones));
                dlZc_h = zeros(size(fones));
                d2lZc_h = dlZc_h;
                if ~isempty(finIdx)
                    [lZc_h(finIdx,:),dlZc_h(finIdx,:),d2lZc_h(finIdx,:)] = likErf([], fones(finIdx,:), mu_h(finIdx)*lam, s2_a(finIdx)*(lam.^2), infMethod);
                end
                
                %lZcmin = min(lZc_l, lZc_h);
                %lZcDelta = max(lZc_l, lZc_h) - lZcmin;
                %lZcsign = sign(lZc_l - lZc_h);
                
                % At this point, finIdx is for the upper thresholds h
                lZ_l = log_expA_x(lZc_l,c);       % A=lZc, B=dlZc, d=c.*lam', lZ=log(exp(A)*c)
                lZ_h = -inf(size(mu));
                if ~isempty(finIdx)
                    lZ_h(finIdx) = log_expA_x(lZc_h(finIdx,:),c);
                end
                
                dlZ_l  = expABz_expAx(lZc_l, c, dlZc_l, c.*lam') * a;  % ((exp(A).*B)*d)./(exp(A)*c)
                dlZ_h = zeros(size(mu));
                if ~isempty(finIdx)
                    dlZ_h(finIdx)  = expABz_expAx(lZc_h(finIdx,:), c, dlZc_h(finIdx,:), c.*lam') * a;
                end
                
                % d2lZ = ((exp(A).*Z)*e)./(exp(A)*c) - dlZ.^2 where e = c.*(lam.^2)'
                d2lZ_l = expABz_expAx(lZc_l, c, dlZc_l.^2+d2lZc_l, c.*(lam.^2)')*a*a - dlZ_l.^2;
                d2lZ_h = zeros(size(mu));
                if ~isempty(finIdx)
                    d2lZ_h(finIdx) = expABz_expAx(lZc_h(finIdx,:), c, dlZc_h(finIdx,:).^2+d2lZc_h(finIdx,:), c.*(lam.^2)')*a*a - dlZ_h(finIdx,:).^2;
                end
                
                %{
                % The following code is copied from GPML, however it's
                % found that it causes error vs. numerical derivative, so
                % it's not used.
                % The scale mixture approximation does not capture the correct asymptotic
                % behavior; we have linear decay instead of quadratic decay as suggested
                % by the scale mixture approximation. By observing that for large values
                % of -f*y ln(p(y|f)) for likLogistic is linear in f with slope y, we are
                % able to analytically integrate the tail region.
                
                val = abs(mu_l)-196/200*(s2_a)-4;       % empirically determined bound at val==0
                lam = 1./(1+exp(-10*val));                         % interpolation weights
                lZtail = min(s2_a/2-abs(mu_l),-0.1);  % apply the same to p(y|f) = 1 - p(-y|f)
                dlZtail = -sign(mu_l); d2lZtail = zeros(size(mu));
                id = mu_l>0; lZtail(id) = log(1-exp(lZtail(id)));  % label and mean agree
                dlZtail(id) = 0;
                
                lZ_l   = (1-lam).*  lZ_l + lam.*lZtail;      % interpolate between scale ..
                dlZ_l  = (1-lam).* dlZ_l + lam.*dlZtail;              % ..  mixture and   ..
                d2lZ = (1-lam).*d2lZ + lam.*d2lZtail;              % .. tail approximation
                
                val = abs(mu_h)-196/200*(s2_a)-4;       % empirically determined bound at val==0
                lam = 1./(1+exp(-10*val));                         % interpolation weights
                lZtail = min(s2_a/2-abs(mu_h),-0.1);  % apply the same to p(y|f) = 1 - p(-y|f)
                dlZtail = -sign(mu_h); d2lZtail = zeros(size(mu));
                id = mu_h<0; lZtail(id) = log(1-exp(lZtail(id)));  % label and mean agree
                dlZtail(id) = 0;

                lZ_h   = (1-lam).*  lZ_h + lam.*lZtail;      % interpolate between scale ..
                dlZ_h  = (1-lam).* dlZ_h + lam.*dlZtail;              % ..  mixture and   ..
                d2lZ = (1-lam).*d2lZ + lam.*d2lZtail;              % .. tail approximation
                %}
                
                Z_l = exp(lZ_l);
                Z_h = exp(lZ_h);
                Zinv = 1./(Z_l-Z_h);
                
                %lZ = log_expA_x(lZcmin, lZcsign(:).*c.*expm1(lZcDelta(:)));
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

%  computes y = log( exp(A)*x ) in a numerically safe way by subtracting the
%  maximal value in each row to avoid cancelation after taking the exp
function y = log_expA_x(A,x)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  y = log(exp(A-maxA*ones(1,N))*x) + maxA;  % exp(A) = exp(A-max(A))*exp(max(A))
end

%  computes y = ( (exp(A).*B)*z ) ./ ( exp(A)*x ) in a numerically safe way
%  The function is not general in the sense that it yields correct values for
%  all types of inputs. We assume that the values are close together.
function y = expABz_expAx(A,x,B,z)
  N = size(A,2);  maxA = max(A,[],2);      % number of columns, max over columns
  A = A-maxA*ones(1,N);                                 % subtract maximum value
  y = ( (exp(A).*B)*z ) ./ ( exp(A)*x );
end