using Revise, DataFrames, CSV, Statistics, Plots, StatsPlots

const DATASETS = ["synthetic", "spiral", "letter-recognition", "TCGA-PANCAN-HiSeq"]
const PALETTE = palette(:tab10)
const MM_COLOR = PALETTE[1]
const SD_COLOR = PALETTE[2]

default(:foreground_color_legend, nothing)
default(:background_color_legend, nothing)
default(:fontfamily, "Computer Modern")
default(:dpi, 600)
default(:legendfontsize, 8)

function is_valid_file(file)
    file = basename(file)
    return startswith(file, "3-") && endswith(file, ".out")
end

function filter_latest(files)
    idx = findlast(contains("algorithm=all"), files)
    return files[idx]
end

# standard errors
ste(x) = std(x) / 10

# summarize over 10-fold CV
function get_summary(df)
    combine(df,
        :obj => mean, :obj => ste,
        :dist => mean, :dist => ste,
        :gradsq => mean, :gradsq => ste,
        :train_acc => mean, :train_acc => ste,
        :val_acc => mean, :val_acc => ste,
        :test_acc => mean, :test_acc => ste
    )
end

# Set example to highlight in Fig. 3
const DATASET = "TCGA-PANCAN-HiSeq"

function main()
    # Get script arguments.
    idir = ARGS[1]
    odir = ARGS[2]
    global DATASET

    dir = joinpath(idir, DATASET)
    files = readdir(dir, join=true) # Read results directory.
    filter!(is_valid_file, files)   # Filter for Experiment 3.
    file = filter_latest(files)     # Filter for latest results.

    # Process the raw dataframe.
    println("""
        Processing: $(file)
    """
    )

    tmp = CSV.read(file, DataFrame,  
                comment="alg",
                header=["alg", "fold", "sparsity", "time", "sv", "iter", "obj", "dist", "gradsq", "train_acc", "val_acc", "test_acc"])

    df = Dict(alg => groupby(filter(:alg => isequal(alg), tmp), :sparsity) for alg in unique(tmp.alg))

    data = vcat(
        insertcols!(get_summary(df["MM"]), 2, :alg => :MM),
        insertcols!(get_summary(df["SD"]), 2, :alg => :SD),
    )

    sparsity = unique(data.sparsity)
    x_index = 1:length(sparsity)
    x_ticks = (x_index, sparsity)
    xs = repeat(x_index, 2)

    options = (
        xticks=x_ticks,
        xrot=22.5,
        xlabel="sparsity (%)",
        mark=[:circle :utriangle],
        ms=6,
        ls=:dot,
        lw=3,
        size=(450,225),
        grid=false,
    )

    @df data plot(xs, :obj_mean .+ 5e-10; ylabel="objective", group=:alg, legend=:bottomright, yscale=:log10, options...)
    savefig(joinpath(odir, "Fig3A.png"))

    @df data plot(xs, :train_acc_mean; ylabel="train accuracy (%)", group=:alg, legend=:bottomleft, ylim=(0,105), options...)
    savefig(joinpath(odir, "Fig3B.png"))

    @df data plot(xs, :dist_mean .+ 5e-7; ylabel="sq dist", group=:alg, legend=:right, yscale=:log10, options...)
    savefig(joinpath(odir, "Fig3C.png"))

    @df data plot(xs, :val_acc_mean; ylabel="validation accuracy (%)", group=:alg, legend=:bottomleft, ylim=(0,105), options...)
    savefig(joinpath(odir, "Fig3D.png"))

    @df data plot(xs, :gradsq_mean .+ 5e-9; ylabel="sq norm(grad)", group=:alg, legend=:bottomright, yscale=:log10, options...)
    savefig(joinpath(odir, "Fig3E.png"))

    @df data plot(xs, :test_acc_mean; ylabel="test accuracy (%)", group=:alg, legend=:bottomleft, ylim=(0,105), options...)
    savefig(joinpath(odir, "Fig3F.png"))
end

main()
