include("common.jl")

function main(args)
    SELECTED_K = [500, 375, 250, 125, 73, 43, 25, 15, 9, 5, 3, 2, 1]
    dir, outdir, i = args[1], args[2], parse(Int, args[3])

    df = CSV.read(joinpath("results", "synthetic", dir, "cv-result.out"), DataFrame)
    df = filter(row -> row.nvars in SELECTED_K, df)
    _tmp = groupby(summarize_over_folds(df), [:replicate, :lambda])
    println(i,"=",collect(eachindex(_tmp))[i])
    tmp = _tmp[i]
    mmdf = filter(row -> row.algorithm == "MMSVD", tmp)
    # sddf = filter(row -> row.algorithm == "SD", tmp)
    keys = map(Symbol, mmdf.variables)
    # dfs = (mmdf, sddf)

    cols = []
    add_column_with_se!(cols, "Iterations", keys, mmdf, :iterations, format=("{:.0f}", "{:.0f}"))
    add_column_with_se!(cols, "Loss", keys, mmdf, :risk, format=("{:.3f}", "{:.3f}"))
    add_column_with_se!(cols, "Support Vectors", keys, mmdf, :nsv, format=("{:.0f}", "{:.0f}"))
    add_column_with_se!(cols, "Train (\\%)", keys, mmdf, :train, format=("{:.0f}", "{:.0f}"))
    add_column_with_se!(cols, "Validation (\\%)", keys, mmdf, :validation, format=("{:.0f}", "{:.0f}"))
    add_column_with_se!(cols, "Test (\\%)", keys, mmdf, :test, format=("{:.0f}", "{:.0f}"))

    tab = hcat(cols...)
    open(joinpath(outdir, "Table2.tex"), "w") do io
        write(io, to_tex(tab))
    end

    return tab
end

main(ARGS)
