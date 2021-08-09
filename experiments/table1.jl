using Dates, DataFrames, CSV, Statistics, Latexify, LaTeXStrings

##### helper functions #####
function is_valid_file(file)
    file = basename(file)
    return startswith(file, "3-") && endswith(file, ".out")
end

function filter_latest(files)
    idx = findlast(contains("algorithm=all"), files)
    return files[idx]
end

const METRICS = [
    :iter, :time, :obj, :dist, :train_acc, :val_acc, :test_acc, :sv
]

function aggregate_metrics(df)
    global METRICS
    gdf = groupby(df, :sparsity)
    combine(gdf,
        [:alg; METRICS] =>
        ( (alg,a,b,c,d,e,f,g,h) -> (
            alg=first(alg),
            iter=median(a),
            time=median(b),
            obj=median(c),
            dist=median(d),
            train_acc=median(e),
            val_acc=median(f),
            test_acc=median(g),
            sv=median(h),
        )) =>
    AsTable)
end

function main()
    # Get script arguments.
    idir = ARGS[1]
    odir = ARGS[2]

    dir = joinpath(idir, "synthetic")
    files = readdir(dir, join=true)
        
    # Filter for lastest results for Experiment 3.
    filter!(is_valid_file, files)
    file = filter_latest(files)

    println("""
        Processing: $(file)
    """
    )

    df = CSV.read(file, DataFrame, comment="alg", header=[
        "alg", "fold", "sparsity", "time", "sv", "iter", "obj", "dist", "gradsq", "train_acc", "val_acc", "test_acc"]
        )
    MM_df = aggregate_metrics(filter(:alg => x -> x == "MM", df))
    SD_df = aggregate_metrics(filter(:alg => x -> x == "SD", df))

    # Consolidate into a single table.
    # result = vcat(MM_df, SD_df)
    result = MM_df
    sort!(result, [:sparsity])
    select!(result, [:sparsity; METRICS])

    # Create a number formatter to handle scientific notation
    fancy = FancyNumberFormatter(4)

    # Eliminate duplicate values in first column to make reading a little easier.
    # unique_vals = unique(result.sparsity)
    # col1 = Vector{String}(undef, nrow(result))
    # cur_val = -1 # assume column we scan does not contain -1
    # for (i, row) in enumerate(eachrow(result))
    #     global cur_val

    #     # found a duplicate entry, so make it blank
    #     if row.sparsity == cur_val
    #         col1[i] = ""
    #     else # otherwise use the same value and update cur_val
    #         col1[i] = string(fancy(row.sparsity))
    #         cur_val = row.sparsity
    #     end
    # end
    # result.sparsity = col1

    # Create header and formatting function.
    header = [
        latexstring(L"s", " (\\%)"),
        "Iter.",
        "Time (s)",
        "Objective",
        "Squared Distance",
        "Train (\\%)",
        "Valid. (\\%)",
        "Test (\\%)",
        "SV"
    ]
    fmt(x) = x # default: no formatting
    fmt(x::Number) = latexstring(x) # make numbers a LaTeXString
    fmt(x::AbstractFloat) = latexstring(fancy(x)) # scientific notation for floats

    # Pass to latexify to create the table.
    table1 = latexify(result, env=:table, booktabs=true, latex=false,
        head=header, fmt=fmt, adjustment=:r)

    open(joinpath(odir, "Table1.tex"), "w") do io
        write(io, table1)
    end
end