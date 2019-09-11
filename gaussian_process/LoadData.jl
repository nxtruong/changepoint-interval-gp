# Module to load data
module LoadData
export load_data, normalize_df!, normalize_df

using DataFrames
using TableReader

"""
    load_data(datadir, folddir, foldnum)

Loads the data and split into training and test data sets based on given fold
number.

- `datadir`: path to the main data directory (`inputs.csv.xz` and `outputs.csv`)
- `folddir`: path to the fold directory (`folds.csv`)
- `foldnum`: fold number, to be found in the `folds.csv` file
- `rmchrom`: whether to remove the `chrom` features (columns `chr*`)
"""
function load_data(datadir, folddir, foldnum; rmchrom=true)
    # The input data (X)
    df_inputs = readcsv(datadir * "/inputs.csv.xz")

    # Remove columns "chr*"
    if rmchrom
        chrcols = filter(x -> startswith(String(x),"chr"), names(df_inputs))
        deletecols!(df_inputs, chrcols)
        # df_inputs = df_inputs[:, filter(x -> !startswith(String(x),"chr"), names(df_inputs))]
    end

    # The output data (Y - targets)
    df_outputs = readcsv(datadir * "/outputs.csv.xz")

    # Load the fold data
    df_folds = readcsv(folddir * "/folds.csv")

    # Build a list of seuenceID with the given fold number
    test_sequences = Vector(df_folds[df_folds[:fold] .== foldnum, 1])
    if length(test_sequences) == 0
        error("No sequences with fold number $foldnum are available.")
    end

    # Create the training and test input DataFrames
    idx = map(x -> x ∈ test_sequences, df_inputs[:sequenceID])
    train_inputs = df_inputs[.!idx, :]
    test_inputs = df_inputs[idx, :]

    # Create the training and test output DataFrames
    train_outputs = df_outputs[.!idx, :]
    test_outputs = df_outputs[idx, :]

    # Return the data sets
    return (train_inputs, train_outputs, test_inputs, test_outputs)
end  # load_data


"""
    normalize_df!(df::DataFrame, cols=nothing)

In-place normalization of columns of a DataFrame `df`.
By default, all columns are normalized, unless `cols` is specified (either by indices or symbols).
`cols` can also be a dictionary of normalization parameters (see `norms` below),
in which case the columns specified in this dictionary will be normalized with
the given parameters.

Outputs: `(df, norms)`
- `df` is the DataFrame
- `norms` contains the normalization parameters. It is a dictionary from a
  column's name (Symbol) to a named tuple of `(min, delta)`. The normalized value
  `xn` from an original value `x` is calculated as `xn = (x-min)/delta`.

Note that normalized columns will be converted to Float64 type if not originally a Real type.
"""
function normalize_df!(df::DataFrame, cols=nothing)
    normgiven = false

    # Create a view into the selected columns of df
    if cols != nothing
        # If cols is a Dict, it's normalization specs
        if isa(cols, Dict)
            normgiven = true
            norms = cols
            cols = collect(keys(norms))
        else
            if !isa(cols, Vector)
                cols = [cols]
            end
        end
        sdf = view(df, cols)
    else
        sdf = df
    end

    if !normgiven
        norms = Dict{Symbol, NamedTuple{(:min, :delta), Tuple{Float64,Float64}}}()
    end

    # Loop through the columns
    for (cname, cvals) in eachcol(sdf, true)
        if cname ∉ names(sdf)
            continue
        end

        if !(eltype(cvals) <: AbstractFloat)
            # Convert column type to Float64
            # Must work with the original DataFrame because the view sdf can't change a column's type
            df[cname] = convert.(Float64, df[cname])
            cvals = df[cname]  # Obtain the converted current column
        end

        if normgiven
            vmin = norms[cname].min
            delta = norms[cname].delta
        else
            vmin, vmax = convert.(Float64, extrema(cvals))
            delta = vmax-vmin
            norms[cname] = (min=vmin,delta=delta)
        end

        @. cvals = (cvals-vmin)/delta
    end

    return (df,norms)
end # function


"""
    normalize_df(df::DataFrame, cols=nothing, removecols=false)

Make a copy of a DataFrame `df` and normalize the selected columns.
It's similar to `normalize_df!` but it makes a copy instead.
If `removecols = true`, all unselected columns in `df` are not copied.
"""
function normalize_df(df::DataFrame, cols=nothing, removecols=false)
    # Make a copy of the DataFrame
    if removecols && cols != nothing
        # If cols is a Dict, it's normalization specs
        if isa(cols, Dict)
            colnames = collect(keys(cols))
        else
            if isa(cols, Vector)
                colnames = cols
            else
                colnames = [cols]
            end
        end
        sdf = df[:, colnames]
    else
        sdf = deepcopy(df)
    end

    return normalize_df!(sdf, cols)
end # function

end  # module LoadData
