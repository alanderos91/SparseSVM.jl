using Dates, DataFrames, CSV, Statistics, Latexify, LaTeXStrings

##### helper functions #####
function is_valid_file(file)
    file = basename(file)
    return startswith(file, "4-") && endswith(file, ".out")
end

function filter_latest(files)
    idx = findlast(contains("algorithm=all"), files)
    return files[idx]
end

function aggregate_metrics(df)
    gdf = groupby(df, :value)
    combine(gdf,
        [:alg, :time, :train_acc, :val_acc, :test_acc] =>
        ( (alg,a,b,c,d) -> (
            alg=first(alg),
            time=median(a),
            train_acc=mean(b),
            val_acc=mean(c),
            test_acc=mean(d),
        )) =>
    AsTable)
end

function subset_max_accuracy(df)
    idx = argmax(0.25*df.train_acc + 0.25*df.val_acc + 0.5*df.test_acc)
    result = df[idx, :]
    result.time = sum(df.time)
    return DataFrame(result)
end

function table3(idir, datasets)
    ALGORITHMS = ("MM", "SD", "L2R", "L1R", "SVC")

    result = DataFrame()

    for (i, dataset) in enumerate(datasets)
        dir = joinpath(idir, dataset)
        files = readdir(dir, join=true)
        
        # Filter for Experiment 4.
        filter!(is_valid_file, files)

        # Filter for latest results.
        file = filter_latest(files)

        # Process the raw dataframe.
        println("""
            Processing: $(file)
        """
        )

        df = CSV.read(file, DataFrame, comment="fold", header=[
            "alg", "fold", "value", "time", "train_acc", "val_acc", "test_acc", "sparsity"]
            )
        alg_vals = unique(df.alg)
        # For each algorithm tested...
        for alg in ALGORITHMS
            if alg ∉ df.alg continue end

            # Aggregate performance metrics over CV folds.
            tmp = aggregate_metrics(filter(:alg => isequal(alg), df))
            if alg in ("MM", "SD")
                tmp.value .*= 100
            end

            # Select representative results based on prediction accuracy.
            result = vcat(result,
                subset_max_accuracy(insertcols!(tmp, 1, :dataset => dataset)))
        end
    end

    # Tidy up table.
    sort!(result, [:dataset])
    select!(result,
        [:dataset, :alg, :value, :time, :train_acc, :val_acc, :test_acc, :sparsity])

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
        "Dataset", "Alg.", latexstring(L"s", " (\\%) / ", L"C"),
        "Total Time (s)", "Tr (\\%)", "V (\\%)", "T (\\%)", "Sparsity"
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

tab3 = table3(idir, datasets)

open(joinpath(odir, "Table3.tex"), "w") do io
    write(io, tab3)
end
