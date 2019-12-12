function M = samplingmetric_vardist(s2train, s2test, distinv, alpha, disttype, N)
%SAMPLINGMETRIC_VARDIST Sample selection metric var/dist.
%   The sample selection metric is calculated for each training point as
%      metric = variance^alpha * d
%   where d for a training point x is
%      d(x) = max(dist(x, TEST))
%   where TEST is a subset of the test inputs depending on N:
%      N = 0: full test set
%      N = 1: test point with max variance
%      N > 1: subset of N test points with highest variance
%   and dist(x, TEST) depends on disttype
%      disttype = 'min': dist(x, TEST) = max(distinv(x, y) for y in TEST)
%      disttype = 'ave': dist(x, TEST) = mean(distinv(x, y) for y in TEST)

narginchk(6,inf);
assert(N >= 0);
assert(ischar(disttype));
ntest = numel(s2test);
assert(ntest == size(distinv,1));

% Find the TEST subset
if N <= 0
    Idx = 1:ntest;
else
    % Select N points with highest variance
    [~, Idx] = sort(s2test, 'descend');
    Idx = Idx(1:min(N,ntest));
end

switch lower(disttype)
    case 'min'
        D = max(distinv(Idx,:));
    case 'ave'
        D = mean(distinv(Idx,:));
    otherwise
        error('Unknown disttype');
end

M = (s2train.^alpha) .* D;
end

