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
    :iter, :time, :obj, :test_acc
]

function aggregate_metrics(df)
    global METRICS
    gdf = groupby(df, :sparsity)
    combine(gdf,
        [:alg, :iter, :time, :obj, :train_acc, :val_acc, :test_acc] =>
        ( (alg,a,b,c,d,e,f) -> (
            alg=first(alg),
            iter=median(a),
            time=median(b),
            obj=median(c),
            test_acc=0.25*mean(d)+0.25*mean(e)+0.5*mean(f),
        )) =>
    AsTable)
end

function subset_max_accuracy(df)
    result = df[argmax(df.test_acc), :]
    result.time = sum(df.time)
    result.iter = sum(df.iter)
    return DataFrame(result)
end

function table2(idir, datasets)
    result = DataFrame()

    for (i, dataset) in enumerate(datasets)
        dir = joinpath(idir, dataset)
        files = readdir(dir, join=true)
        
        # Filter for Experiment 3.
        filter!(is_valid_file, files)

        # Filter for latest results.
        file = filter_latest(files)

        # Process the raw dataframe.
        println("""
            Processing: $(file)
        """
        )

        df = CSV.read(file, DataFrame, comment="alg", header=[
            "alg", "fold", "sparsity", "time", "sv", "iter", "obj", "dist", "train_acc", "val_acc", "test_acc"]
            )
        MM_df = aggregate_metrics(filter(:alg => x -> x == "MM", df))
        SD_df = aggregate_metrics(filter(:alg => x -> x == "SD", df))
        for df in (MM_df, SD_df)
            tmp = subset_max_accuracy(insertcols!(df, 1, :dataset => dataset))
            result = vcat(result, tmp)
        end
    end

    # Tidy up table.
    sort!(result, [:dataset, :alg])
    select!(result, [:dataset; :alg; :sparsity; METRICS])

    # Create a number formatter to handle scientific notation
    fancy = FancyNumberFormatter(4)

    # Eliminate duplicate values in first column to make reading a little easier.
    unique_vals = unique(result.dataset)
    col1 = Vector{String}(undef, nrow(result))
    cur_val = -1 # assume column we scan does not contain -1
    for (i, row) in enumerate(eachrow(result))
        # found a duplicate entry, so make it blank
        if row.dataset == cur_val
            col1[i] = ""
        else # otherwise use the same value and update cur_val
            col1[i] = string(fancy(row.dataset))
            cur_val = row.dataset
        end
    end
    result.dataset = col1

    # Create header and formatting function.
    header = [
        "Dataset", "Alg.", latexstring(L"s", " (\\%)"), "Total Iter.", "Total Time (s)",
        L"h_{\rho}(\bbeta)", "Test (\\%)"
    ]
    fmt(x) = x # default: no formatting
    fmt(x::Number) = latexstring(x) # make numbers a LaTeXString
    fmt(x::AbstractFloat) = latexstring(fancy(x)) # scientific notation for floats

    # modify dataset column to use \texttt
    result.dataset = map(x -> "\\texttt{$(x)}", result.dataset)

    # Pass to latexify to create the table.
    return latexify(result, env=:table, booktabs=true, latex=false,
        head=header, fmt=fmt, adjustment=:r)
end

# Get script arguments.
idir = ARGS[1]
odir = ARGS[2]
datasets = [
    "breast-cancer-wisconsin", "iris", "letter-recognition", "optdigits",
    "spiral", "splice", "synthetic", "TCGA-PANCAN-HiSeq"
]

tab2 = table2(idir, datasets)

open(joinpath(odir, "Table2.tex"), "w") do io
    write(io, tab2)
end
