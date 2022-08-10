include("common.jl")

using CairoMakie

function main(args)
    dir, outdir = args[1], args[2]
    examples = args[3:end]

    fz = 16.0
    kwargs = (;
        width=400,
        height=400,
        yticks=LinearTicks(11),
        xlabelsize=2*fz,
        ylabelsize=2*fz,
        xticklabelsize=1.5*fz,
        yticklabelsize=1.5fz,
    )

    # load repeated cv results and summarize over folds within reach replicate
    for example in examples
        filepath = joinpath("results", example, dir, "cv-result.out")
        if !ispath(filepath) continue end

        fulldf = summarize_over_folds(CSV.read(filepath, DataFrame))
        mmdf = filter(row -> row.algorithm == "MMSVD", fulldf)
        mmdf.algorithm .= "MM"
        sddf = filter(row -> row.algorithm == "SD", fulldf)

        sparsity_grid, lambda_grid, k_grid = sort!(unique(fulldf.sparsity)), sort!(unique(fulldf.lambda)), sort!(unique(fulldf.variables), rev=true)
        ns, nl = length(sparsity_grid), length(lambda_grid)
        for algorithm_df in (mmdf, sddf)
            accuracy_dist, s_dist, loglambda_dist, k_dist = Float64[], Float64[], Float64[], Int[]
            for df in groupby(algorithm_df, [:replicate])
                nt = reshape(df.validation, ns, nl)
                (i, _, (accuracy, s, lambda)) = SparseSVM.search_hyperparameters(sparsity_grid, lambda_grid, nt, minimize=false)
                push!(s_dist, s*100)
                push!(loglambda_dist, log10(lambda))
                push!(k_dist, k_grid[i])
                push!(accuracy_dist, accuracy)
            end

            fig = Figure(resolution=(800,600))
            fig[1:2,1] = GridLayout()
            g = fig[2,2] = GridLayout()
            fig[1:2,3] = GridLayout()
            metrics = (s_dist, k_dist, loglambda_dist, accuracy_dist)
            xlabels = ["Sparsity (%)", latexstring("Target Variables ", L"k"), L"\log(\lambda)", "Valdiation (%)"]
            list_of_ax = []
            Label(g[1,1:4], example, tellwidth=false, textsize=2*fz)
            for (j, (data, xlab)) in enumerate(zip(metrics, xlabels))
                ax = Axis(g[2,j]; xlabel=xlab, ylims=(0,10), kwargs...)
                hist!(ax, data)
                density!(ax, data, color=(:black, 0.0), strokecolor=:black, strokewidth=5)
                push!(list_of_ax, ax)
            end
            linkyaxes!(list_of_ax...)
            resize_to_layout!(fig)

            algorithm = first(algorithm_df.algorithm)
            save(joinpath(outdir, "stability-$(example)-$(algorithm).pdf"), fig, pt_per_unit=2)
        end
    end
end

main(ARGS)