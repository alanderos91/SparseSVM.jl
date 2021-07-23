using Dates, DataFrames, CSV, Statistics, Plots, StatsPlots

const PALETTE = palette(:tab10)
const MM_COLOR = PALETTE[1]
const SD_COLOR = PALETTE[2]

default(:foreground_color_legend, nothing)
default(:background_color_legend, nothing)
default(:legendfontsize, 6)

const SELECTED_COLS = [:sv, :obj, :dist, :test_acc]

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
    insertcols!(df, 1, :algorithm => alg)
    insertcols!(df, 2, :dataset => dataset)
end

get_randomized_subset(df) = filter(:trial => x -> x > 0, df)

function figure1(df)
    subplot_label = Dict(
    :sv => "no. support vectors",
    :obj => "penalized objective",
    :dist => "squared distance",
    :test_acc => "test accuracy (%)",
    )
    w, h = default(:size)

    # Initialize the plot
    fig = plot(layout=grid(2,2), legend=false, grid=false, size=(4*w, h))
    for (i, metric) in enumerate(SELECTED_COLS)
        # Add boxplots for each metric.
        @df df groupedboxplot!(fig, string.(:dataset), cols(metric),
            title=subplot_label[metric],
            legend=:outerright,
            bar_width=0.4,
            markerstrokewidth=0,
            group=:algorithm,
            yscale=i==2 || i==3 ? :log10 : :identity,
            subplot=i,
            size=(2*w, h),
        )
    end
    return fig
end

# Get script arguments.
idir = ARGS[1]
odir = ARGS[2]
datasets = ARGS[3:end]

df = DataFrame()

for (i, dataset) in enumerate(datasets)
    global df

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
