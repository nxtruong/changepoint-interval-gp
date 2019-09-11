function [train_inputs, train_outputs, test_inputs, test_outputs] = load_data(datadir, folddir, foldnum, rmchrom)
    df_inputs = readtable([datadir '/inputs.csv']);

    % Remove columns "chr*"
    if rmchrom
        chrcols = cellfun(@(x) startsWith(x, 'chr'), df_inputs.Properties.VariableNames);
        df_inputs(:, chrcols) = [];
    end

    % The output data (Y - targets)
    df_outputs = readtable([datadir '/outputs.csv']);

    % Load the fold data
    df_folds = readtable([folddir '/folds.csv']);

    % Build a list of sequenceID with the given fold number
    test_sequences = df_folds.sequenceID(df_folds.fold == foldnum);
    if isempty(test_sequences)
        error('No sequences with fold number $foldnum are available.');
    end

    % Create the training and test input DataFrames
    idx = cellfun(@(x) ismember(x, test_sequences), df_inputs.sequenceID);
    train_inputs = df_inputs(~idx, :);
    test_inputs = df_inputs(idx, :);

    % Create the training and test output DataFrames
    train_outputs = df_outputs(~idx, :);
    test_outputs = df_outputs(idx, :);    
end
