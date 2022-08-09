# using Revise, DataFrames, CSV, Statistics, Plots, StatsPlots

# const DATASETS = ["synthetic", "spiral", "letter-recognition", "TCGA-PANCAN-HiSeq"]
# const PALETTE = palette(:tab10)
# const MM_COLOR = PALETTE[1]
# const SD_COLOR = PALETTE[2]

# default(:foreground_color_legend, nothing)
# default(:background_color_legend, nothing)
# default(:fontfamily, "Computer Modern")
# default(:dpi, 600)
# default(:legendfontsize, 8)

# function is_valid_file(file)
#     file = basename(file)
#     return startswith(file, "3-") && endswith(file, ".out")
# end

# function filter_latest(files)
#     idx = findlast(contains("algorithm=all"), files)
#     return files[idx]
# end

# # standard errors
# ste(x) = std(x) / 10

# # summarize over 10-fold CV
# function get_summary(df)
#     combine(df,
#         :obj => mean, :obj => ste,
#         :dist => mean, :dist => ste,
#         :gradsq => mean, :gradsq => ste,
#         :train_acc => mean, :train_acc => ste,
#         :val_acc => mean, :val_acc => ste,
#         :test_acc => mean, :test_acc => ste
#     )
# end

# # Set example to highlight in Fig. 3
# const DATASET = "TCGA-PANCAN-HiSeq"

# function main()
#     # Get script arguments.
#     idir = ARGS[1]
#     odir = ARGS[2]
#     global DATASET

#     dir = joinpath(idir, DATASET)
#     files = readdir(dir, join=true) # Read results directory.
#     filter!(is_valid_file, files)   # Filter for Experiment 3.
#     file = filter_latest(files)     # Filter for latest results.

#     # Process the raw dataframe.
#     println("""
#         Processing: $(file)
#     """
#     )

#     tmp = CSV.read(file, DataFrame,  
#                 comment="alg",
#                 header=["alg", "fold", "sparsity", "time", "sv", "iter", "obj", "dist", "gradsq", "train_acc", "val_acc", "test_acc"])

#     df = Dict(alg => groupby(filter(:alg => isequal(alg), tmp), :sparsity) for alg in unique(tmp.alg))

#     data = vcat(
#         insertcols!(get_summary(df["MM"]), 2, :alg => :MM),
#         insertcols!(get_summary(df["SD"]), 2, :alg => :SD),
#     )

#     sparsity = unique(data.sparsity)
#     x_index = 1:length(sparsity)
#     x_ticks = (x_index, sparsity)
#     xs = repeat(x_index, 2)

#     options = (
#         xticks=x_ticks,
#         xrot=22.5,
#         xlabel="sparsity (%)",
#         mark=[:circle :utriangle],
#         ms=6,
#         ls=:dot,
#         lw=3,
#         size=(450,225),
#         grid=false,
#     )

#     @df data plot(xs, :obj_mean .+ 5e-10; ylabel="objective", group=:alg, legend=:bottomright, yscale=:log10, options...)
#     savefig(joinpath(odir, "Fig3A.png"))

#     @df data plot(xs, :train_acc_mean; ylabel="train accuracy (%)", group=:alg, legend=:bottomleft, ylim=(0,105), options...)
#     savefig(joinpath(odir, "Fig3B.png"))

#     @df data plot(xs, :dist_mean .+ 5e-7; ylabel="sq dist", group=:alg, legend=:right, yscale=:log10, options...)
#     savefig(joinpath(odir, "Fig3C.png"))

#     @df data plot(xs, :val_acc_mean; ylabel="validation accuracy (%)", group=:alg, legend=:bottomleft, ylim=(0,105), options...)
#     savefig(joinpath(odir, "Fig3D.png"))

#     @df data plot(xs, :gradsq_mean .+ 5e-9; ylabel="sq norm(grad)", group=:alg, legend=:bottomright, yscale=:log10, options...)
#     savefig(joinpath(odir, "Fig3E.png"))

#     @df data plot(xs, :test_acc_mean; ylabel="test accuracy (%)", group=:alg, legend=:bottomleft, ylim=(0,105), options...)
#     savefig(joinpath(odir, "Fig3F.png"))
# end

# main()

include("common.jl")

using CairoMakie

function make_figure(mmdf, sddf)
    # plot parameters
    fz = 16
    lw, msz = 5, 20
    w, h = 550, 200
    pad_percent = 5
    color = [:navy, :orange]
    marker = [:circle, :utriangle]
    axis_kwargs = (;
        width=w,
        height=h,
        xlabelsize=2*fz,
        ylabelsize=2*fz,
        xticklabelsize=1.5*fz,
        yticklabelsize=1.5fz,
    )
    xs = mmdf.sparsity
    idx = 1:4:length(xs)

    fig = Figure(resolution=(800,600))
    g = fig[1,1] = GridLayout()

    # this is the layout we will use
    metric = [
        :risk :gradient :distance
        :train :validation :test
    ]
    ylabel = [
        "Loss" "Norm(gradient)" "Distance"
        "Train (%)" "Validation (%)" "Test (%)"
    ]
    yscale = [
        log10 log10 Makie.pseudolog10
        identity identity identity
    ]

    ax = [
        Axis(g[i,j];
            xlabel="Active Variables",
            ylabel=ylabel[i,j],
            yscale=yscale[i,j],
            axis_kwargs...,
        ) for i in axes(metric, 1), j in axes(metric, 2)
    ]

    # set the common x-axis
    linkxaxes!(ax...)
    for current_ax in ax
        current_ax.xticklabelrotation = pi/4
        current_ax.xticks = idx
        current_ax.xtickformat = _ -> map(xs[idx]) do x
            v = round(Int, 20265*(1-x))
            if v >= 1000
                string(round(v/1000, sigdigits=3), "k")
            else
                string(v)
            end
        end
    end

    # additional formmating for bottom row
    linkyaxes!(ax[2,:]...)
    for j in axes(metric, 2)
        current_ax = ax[2,j]
        ylims!(ax[2,j], 0-pad_percent, 100+pad_percent)
        current_ax.yticks = LinearTicks(5)
    end

    for j in axes(metric, 2), i in axes(metric, 1)
        current_ax = ax[i,j]
        ycol = metric[i,j]

        for (k, df) in enumerate((mmdf, sddf))
            ys = df[!,ycol]
            stairs!(current_ax, ys; color=color[k], linewidth=lw)
            scatter!(current_ax, ys; color=color[k], marker=marker[k], markersize=msz)
        end
    end

    # add legend entry
    mm_entry = [
        LineElement(color=color[1], linewidth=lw),
        MarkerElement(color=color[1], marker=marker[1], markersize=1.5*msz)
    ]
    sd_entry = [
        LineElement(color=color[2], linewidth=lw),
        MarkerElement(color=color[2], marker=marker[2], markersize=1.5*msz)
    ]
    fig[2,1] = Legend(fig, [mm_entry, sd_entry], ["MM", "SD"], "Algorithm",
        framevisible=false,
        titlesize=2.5*fz,
        labelsize=2*fz,
        patchsize=(80,60),
        orientation=:horizontal,
        nbanks=1,
        tellwidth=false,
        tellheight=true,
        colgap=250,
    )

    fig[1:2,0] = GridLayout()
    fig[1:2,2] = GridLayout()

    resize_to_layout!(fig)

    return fig
end

function main(args)
    dir, outdir = args[1], args[2]

    data = CSV.read(joinpath("results", "TCGA-HiSeq", dir, "cv-result.out"), DataFrame)
    mmdf = summarize_over_folds(filter(row -> row.algorithm == "MMSVD", data))
    sddf = summarize_over_folds(filter(row -> row.algorithm == "SD", data))

    # Figure 3 w/ λ = 1
    lambda = 1.0
    fig3 = make_figure(
        filter(row -> row.lambda == lambda, mmdf),
        filter(row -> row.lambda == lambda, sddf),
    )
    save(joinpath(outdir, "Fig3.pdf"), fig3, pt_per_unit=2)

    # Supplemental Figure w/ λ = 0.1
    lambda = 0.1
    extra1 = make_figure(
        filter(row -> row.lambda == lambda, mmdf),
        filter(row -> row.lambda == lambda, sddf),
    )
    save(joinpath(outdir, "SFig1.pdf"), extra1, pt_per_unit=2)

    # Supplemental Figure w/ λ = 10.0
    lambda = 10.0
    extra2 = make_figure(
        filter(row -> row.lambda == lambda, mmdf),
        filter(row -> row.lambda == lambda, sddf),
    )
    save(joinpath(outdir, "SFig2.pdf"), extra2, pt_per_unit=2)

    return fig3, extra1, extra2
end

main(ARGS)