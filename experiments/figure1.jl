using Dates, DataFrames, CSV, Statistics, Plots, StatsPlots

# Get script arguments.
idir = ARGS[1]
odir = ARGS[2]
datasets = ARGS[3:end]

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

subplot_label = Dict(
    :sv => "no. support vectors",
    :obj => "penalized objective",
    :dist => "distance",
    :test_acc => "test accuracy (%)",
)
w, h = default(:size)

# Initialize the plot
fig = plot(layout=grid(4,1), legend=false, grid=false)
for (i, metric) in enumerate(SELECTED_COLS)
    # Add boxplots for each metric.
    @df df groupedboxplot!(fig, string.(:dataset), cols(metric),
        title=subplot_label[metric],
        legend=:topleft,
        fillalpha=0.75,
        group=:algorithm,
        subplot=i,
        size=(w, 0.25*h),
    )
end
plot!(fig, size=(w, 2*h))

savefig(fig, joinpath(odir, "Fig1.png"))
