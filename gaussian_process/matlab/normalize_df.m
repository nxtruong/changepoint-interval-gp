function [df, norms] = normalize_df(df, cols)
%     normalize_df(df, cols=[])
% 
% In-place normalization of columns of a DataFrame `df`.
% By default, all columns are normalized, unless `cols` is specified (either by indices or symbols).
% `cols` can also be a dictionary of normalization parameters (see `norms` below),
% in which case the columns specified in this dictionary will be normalized with
% the given parameters.
% 
% Outputs: `(df, norms)`
% - `df` is the DataFrame
% - `norms` contains the normalization parameters. It is a dictionary from a
%   column's name (Symbol) to a named tuple of `(min, delta)`. The normalized value
%   `xn` from an original value `x` is calculated as `xn = (x-min)/delta`.
% 
% Note that normalized columns will be converted to Float64 type if not originally a Real type.

    normgiven = false;

    % Create a view into the selected columns of df
    if ~isempty(cols)
        % If cols is a Dict, it's normalization specs
        if isstruct(cols)
            normgiven = true;
            norms = cols;
            cols = fieldnames(norms);
        else
            if ischar(cols)
                cols = {cols};
            elseif ~iscell(cols)
                cols = df.Properties.VariableNames(cols);
            end
        end
    else
        cols = df.Properties.VariableNames;
    end
    
    if ~normgiven
        minvals = varfun(@(c) min(c), df(:,cols), 'OutputFormat', 'uniform');
        maxvals = varfun(@(c) max(c), df(:,cols), 'OutputFormat', 'uniform');
        norms = struct;
        for k = 1:numel(cols)
            norms.(cols{k}) = [minvals(k), maxvals(k)-minvals(k)];
        end
    end
    
    for k = 1:numel(cols)
        normparams = norms.(cols{k});
        if normparams(2) <= 0
            continue;
        end
        
        df{:, cols{k}} = (df{:, cols{k}} - normparams(1)) / normparams(2);
    end
end
