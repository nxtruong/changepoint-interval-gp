classdef RealInterval
    %REALINTERVAL Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (GetAccess=public, SetAccess=protected)
        l  % Lower threshold, can be -Inf
        h  % upper threshod, can be +Inf
    end
    
    methods
        function obj = RealInterval(l, h)
            %REALINTERVAL Construct an instance of this class
            %   Detailed explanation goes here
            if nargin == 0
                return;
            end
            
            nl = length(l);
            nh = length(h);
            assert(nl == nh);
            n = max(nl, nh);
            if n > 1
                l = l(:);
                h = h(:);
            end
            assert(all(isfloat(l)) && all(isfloat(h)) && all(l < h), 'Invalid l and h.');
            assert(all(isfinite(l) | isfinite(h)), 'l and h cannot be both inf.');
            if n > 1
                obj(n,1) = obj;
                for k = 1:n
                    obj(k).l = l(k);
                    obj(k).h = h(k);
                end
            else
                obj.l = l;
                obj.h = h;
            end
        end
    end
end

