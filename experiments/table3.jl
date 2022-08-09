include("common.jl")

function main(args)
    dir, outdir = args[1], args[2]
    examples = args[3:end]

    # load repeated cv results and summarize over folds within reach replicate
    tmp = []
    for example in examples
        filepath = joinpath("results", example, dir, "cv-comparison.out")
        if ispath(filepath)
            tmpdf = CSV.read(filepath, DataFrame)
            tmpdf[!,:example] .= example
            push!(tmp, tmpdf)
        end
    end
    output = vcat(tmp...)

    output.sparsity .*= 100
    output.train .*= 100
    output.test .*= 100
    output.margin = 1 ./ output.norm

    rows = []
    for example in examples
        subset = filter(row -> row.example == example, output)
        keys = map(x -> Symbol(uppercasefirst(x)), unique(subset.model))

        cols = []
        add_column!(cols, "Example", keys, subset, :example, format="{:s}")
        add_column!(cols, "\$\\lambda\$", keys, subset, :lambda, format="{:4.2f}")
        add_column!(cols, "\$k\$", keys, subset, :nvars, format="{:d}")
        add_column!(cols, "Active Variables", keys, subset, :nnz, format="{:.0f}")
        add_column!(cols, "Active Variables (average)", keys, subset, :anz, format="{:.0f}")
        add_column!(cols, "Support Vectors", keys, subset, :nsv, format="{:.0f}")
        add_column!(cols, "Margin", keys, subset, :margin, format="{:4.2f}")
        add_column!(cols, "Train (\\%)", keys, subset, :train, format="{:3.0f}")
        add_column!(cols, "Test (\\%)", keys, subset, :test, format="{:3.0f}")

        push!(rows, hcat(cols...))
    end

    tab = vcat(rows...)
    open(joinpath(outdir, "Table3.tex"), "w") do io
        write(io, to_tex(tab))
    end

    return output, tab
end

main(ARGS)
