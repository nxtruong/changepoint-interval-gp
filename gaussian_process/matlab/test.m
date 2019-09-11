l = 0;
h = 4;
yint = RealInterval(l,h);

sigma = 0.05;

for mu = l:0.1:h
    delta = 1e-8;
    [lZ1,dlZ1,d2lZ1] = likIntervalLogistic(5, [], yint, mu, sigma^2, 'infEP');
    [lZ2,dlZ2,d2lZ2] = likIntervalLogistic(5, [], yint, mu+delta, sigma^2, 'infEP');
    
    [lZa, dlZa, d2lZa] = likIntervalLogistic1(5, [], yint, mu, sigma^2, 'infEP');
    [lZb, dlZb, d2lZb] = likIntervalLogistic1(5, [], yint, mu+delta, sigma^2, 'infEP');
    
    fprintf('mu = %g => num deriv err = %g; lZ err = %g; dlZ err = %g; d2lZ err = %g\n', mu, ...
        (dlZb-dlZa)/delta - d2lZa,...
        lZa - lZ1, dlZa - dlZ1, d2lZa - d2lZ1);
end

for mu = l:0.1:h
    delta = 1e-8;
    [lZ1,dlZ1,d2lZ1] = likIntervalErf(0.1, [], yint, mu, sigma^2, 'infEP');
    [lZ2,dlZ2,d2lZ2] = likIntervalErf(0.1, [], yint, mu+delta, sigma^2, 'infEP');

    fprintf('mu = %g => dlZ err = %g; d2lZ err = %g\n', mu, ...
        (lZ2-lZ1)/delta - dlZ1,...
        (dlZ2-dlZ1)/delta - d2lZ1);
end