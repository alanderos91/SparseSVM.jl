include("common.jl")

get_grids(df) = unique(df.sparsity), unique(df.lambda)

function set_sigdigits(output, metric, n)
    output[!,metric] .= round.(output[!,metric], sigdigits=n)
end

function main(args)
    dir, outdir = args[1], args[2]
    examples = args[3:end]

    # load repeated cv results and summarize over folds within reach replicate
    tmp = []
    for example in examples
        filepath = joinpath("results", example, dir, "cv-result.out")
        if ispath(filepath)
            tmpdf = summarize_over_folds(CSV.read(filepath, DataFrame))
            tmpdf[!,:example] .= example
            push!(tmp, tmpdf)
        end

        filepath = joinpath("results", example, dir, "cv-libsvm-result.out")
        if ispath(filepath)
            tmpdf = summarize_over_folds(CSV.read(filepath, DataFrame))
            tmpdf[!,:example] .= example
            push!(tmp, tmpdf)
        end
    end
    df = vcat(tmp...)

    # find optimal model for each CV replicate
    gdf = groupby(df, [:example, :algorithm, :replicate])
    metric = :validation
    optimal_model = []
    for i in eachindex(gdf)
        ss, ls = get_grids(gdf[i])
        data = reshape(gdf[i][!,metric], length(ss), length(ls))
        push!(optimal_model, i => SparseSVM.search_hyperparameters(ss, ls, data, minimize=false))
    end

    # select record corresponding to optimal model in each CV replicate
    tmp = []
    for (key, val) in optimal_model
        ss, ls = get_grids(gdf[key])
        i, j, _ = val
        row_idx = findfirst(row -> row.sparsity==ss[i] && row.lambda==ls[j], eachrow(gdf[key]))
        tmpdf = DataFrame()
        push!(tmpdf, gdf[key][row_idx,:])
        tmpdf.time .= sum(gdf[key].time)
        push!(tmp, tmpdf)
    end

    # summarize over replicates
    output = combine(groupby(vcat(tmp...), [:example, :algorithm]),
        :lambda => mean => :lambda,
        :variables => mean => :variables,
        :time => mean => :time,
        :nnz => mean => :nnz,
        :anz => mean => :anz,
        :nsv => mean => :nsv,
        :train => mean => :train,
        :validation => mean => :validation,
        :test => mean => :test,
    )

    # significant digits
    set_sigdigits(output, :time, 3)
    set_sigdigits(output, :train, 4)
    set_sigdigits(output, :validation, 4)
    set_sigdigits(output, :test, 4)

    # # strip repeated row information
    # for example in examples
    #     i = findfirst(isequal(example), output.example)
    #     for j in setdiff(findall(isequal(example), output.example), i)
    #         output[j,:example] = ""
    #     end
    # end

    rows = []
    for example in examples
        subset = filter(row -> row.example == example, output)
        keys = map(x -> Symbol(x), eachindex(unique(subset.algorithm)))

        cols = []
        add_column!(cols, "Example", keys, subset, :example, format="{:s}")
        add_column!(cols, "Algorithm", keys, subset, :algorithm, format="{:s}")
        add_column!(cols, "\$\\lambda\$", keys, subset, :lambda, format="{:4.2f}")
        add_column!(cols, "\$k\$", keys, subset, :variables, format="{:.0f}")
        add_column!(cols, "Total Time [s]", keys, subset, :time, format="{:3.3f}")
        add_column!(cols, "Total", keys, subset, :nnz, format="{:.0f}")
        add_column!(cols, "Average", keys, subset, :anz, format="{:.0f}")
        add_column!(cols, "Support Vectors", keys, subset, :nsv, format="{:.0f}")
        add_column!(cols, "Train", keys, subset, :train, format="{:.0f}")
        add_column!(cols, "Validation", keys, subset, :validation, format="{:.0f}")
        add_column!(cols, "Test", keys, subset, :test, format="{:.0f}")

        push!(rows, hcat(cols...))
    end

    tab = vcat(rows...)
    open(joinpath(outdir, "Table4.tex"), "w") do io
        write(io, to_tex(tab))
    end

    return output, tab
end

main(ARGS)