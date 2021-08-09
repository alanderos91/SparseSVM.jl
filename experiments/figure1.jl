using Dates, DataFrames, CSV, Statistics, Plots, StatsPlots, LaTeXStrings

const DATASETS = ["synthetic", "spiral", "letter-recognition", "TCGA-PANCAN-HiSeq"]
const PALETTE = palette(:tab10)
const MM_COLOR = PALETTE[1]
const SD_COLOR = PALETTE[2]

default(:foreground_color_legend, nothing)
default(:background_color_legend, nothing)
default(:fontfamily, "Computer Modern")
default(:dpi, 600)
default(:legendfontsize, 8)

const SELECTED_COLS = [:iter, :time, :obj, :gradsq, :dist, :sv, :train_acc, :test_acc]

function is_valid_file(file)
    file = basename(file)
    return startswith(file, "1-") && endswith(file, ".out")
end

function filter_latest(files)
    MM_idx = findlast(contains("algorithm=MM"), files)
    SD_idx = findlast(contains("algorithm=SD"), files)
    return files[MM_idx], files[SD_idx]
end

function add_columns!(df, alg, dataset)
    dataset = occursin("TCGA", dataset) ?
    "HiSeq" : occursin("letter", dataset) ?
    "letters" : dataset
    insertcols!(df, 1, :algorithm => alg)
    insertcols!(df, 2, :dataset => dataset)
end

get_randomized_subset(df) = filter(:trial => x -> x > 0, df)

function figure1(df)
    subplot_label = Dict(
    :iter => "# iterations",
    :time => "time (s)",
    :sv => "# support vectors",
    :obj => "penalized objective",
    :dist => "squared distance",
    :gradsq => "squared norm gradient",
    :train_acc => "train accuracy (%)",
    :test_acc => "test accuracy (%)",
    )
    w, h = default(:size)

    yscale = [:log10, :log10, :log10, :log10, :log10, :log10, :identity, :identity]
    legend = [:topleft, nothing, nothing, nothing, nothing, nothing, nothing, nothing]
    ylimits = [:auto, :auto, :auto, :auto, :auto, :auto, (0, 105), (0, 105)]

    # Initialize the plot
    fig = plot(layout=grid(2,4), legend=false, grid=false, size=(4*w, 2*h))
    for (i, metric) in enumerate(SELECTED_COLS)
        # Add boxplots for each metric.
        @df df groupedboxplot!(fig, string.(:dataset), cols(metric),
            title=subplot_label[metric],
            legend=legend[i],
            bar_width=0.4,
            markerstrokewidth=0,
            group=:algorithm,
            yscale=yscale[i],
            ylims=ylimits[i],
            yticks=true,
            yminorticks=true,
            subplot=i,
            fontfamily="Computer Modern",
            thickness_scaling=2,
        )
    end
    return fig
end

function main()
    # Get script arguments.
    idir = ARGS[1]
    odir = ARGS[2]

    global DATASETS

    df = DataFrame()

    for (i, dataset) in enumerate(DATASETS)
        dir = joinpath(idir, dataset)
        files = readdir(dir, join=true)
        
        # Filter for Experiment 1.
        filter!(is_valid_file, files)

        # Filter for latest results.
        MM_file, SD_file = filter_latest(files)

        println("""
            Processing...
                - MM file: $(MM_file)
                - SD file: $(SD_file)
        """
        )

        MM_df = CSV.read(MM_file, DataFrame)
        add_columns!(MM_df, :MM, dataset)
        df = vcat(df, get_randomized_subset(MM_df))

        SD_df = CSV.read(SD_file, DataFrame)
        add_columns!(SD_df, :SD, dataset)
        df = vcat(df, get_randomized_subset(SD_df))
    end

    fig = figure1(df)

    savefig(fig, joinpath(odir, "Fig1.png"))
end

main()
