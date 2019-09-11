% Test code

%% Generate data
truefunc = @(x) sin(x);

N = 100;
x = rand(N,1) * 4*pi;
y = truefunc(x) + randn(size(x))*0.0001;

% Create intervals
yint_l = y - rand(N,1)*0.1;
yint_h = y + rand(N,1)*0.1;
% Make some one-side open intervals
yintrand = rand(N,1);
yint_l(yintrand < 0.15) = -inf;
yint_h(yintrand > 0.85) = inf;
yint = RealInterval(yint_l, yint_h);

%% Create GP model
meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([1 1]);
% likfunc = {@likIntervalLogisticFull, [0.1, 20]}; hyp.lik = 0;
% likfunc = {@likIntervalLogistic, 5}; hyp.lik = [];
likfunc = {@likIntervalErf, 0.01}; hyp.lik = [];

% infMethod = @infLaplace;
infMethod = @infEP;

[nlZ, dnlZ] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, x, yint)

%% Optimize GP model

hyp = minimize(hyp, @gp, -40, infMethod, meanfunc, covfunc, likfunc, x, yint);

[nlZ, dnlZ] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, x, yint)

%% Prediction

xtest = linspace(1,3,20)' * pi;
[ymu, ys2, fmu, fs2] = gp(hyp, infMethod, meanfunc, covfunc, likfunc, x, yint, xtest);

%% Plotting
figure;
plot(xtest, truefunc(xtest), 'b', xtest, fmu, 'r');
hold on

[x1, i1] = sort(x);
yl1 = yint_l(i1);
yl1(isinf(yl1)) = -2;
yh1 = yint_h(i1);
yh1(isinf(yh1)) = 2;

plot(x1, yl1, 'b:', x1, yh1, 'b:');

scatter(xtest, fmu, 'r');
plot(xtest, fmu + 2*sqrt(fs2), 'r--');
plot(xtest, fmu - 2*sqrt(fs2), 'r--');