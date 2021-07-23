using Dates, DataFrames, CSV, Statistics, Plots, StatsPlots

const PALETTE = palette(:tab10)
const MM_COLOR = PALETTE[1]
const SD_COLOR = PALETTE[2]

default(:foreground_color_legend, nothing)
default(:background_color_legend, nothing)
default(:legendfontsize, 6)

##### helper functions #####
function is_valid_file(file)
    file = basename(file)
    return startswith(file, "2-") && endswith(file, ".out")
end

function filter_latest(files)
    MM_idx = findlast(contains("algorithm=MM"), files)
    SD_idx = findlast(contains("algorithm=SD"), files)
    return files[MM_idx], files[SD_idx]
end

function add_column!(df, alg)
    insertcols!(df, 1, :algorithm => alg)
end

# Positive and Negative Predictive Value (PPV, NPV); interpreted as posterior probability
function ppv(TP, FP, TN, FN, p)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = 100 * TPR * p / ( TPR * p + (1 - TNR) * (1 - p) )
    return isnan(PPV) ? zero(PPV) : PPV
end

function npv(TP, FP, TN, FN, p)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    NPV = 100 * TNR * (1 - p) / ( TNR * (1 - p) + (1 - TPR) * p )
    return isnan(NPV) ? zero(NPV) : NPV
end

function figure2a(MM_df, SD_df)
    k0 = 50 # should get this from table

    # underdetermined
    df1 = vcat(
        filter([:m, :n] => (m, n) -> m == 500 && n == 10^4, MM_df),
        filter([:m, :n] => (m, n) -> m == 500 && n == 10^4, SD_df),
    )

    # overdetermined
    df2 = vcat(
        filter([:m, :n] => (m, n) -> m == 10^4 && n == 500, MM_df),
        filter([:m, :n] => (m, n) -> m == 10^4 && n == 500, SD_df),
    )

    # Initialize the plot
    sparsity = unique(df1.sparsity)
    x_index = 1:length(sparsity)
    x_ticks = (x_index, sparsity)
    xs = repeat(x_index, 2)

    global MM_COLOR
    global SD_COLOR
    colors = [MM_COLOR SD_COLOR]

    w, h = default(:size)
    fig = plot(layout=grid(2, 2), legend=:bottomright, grid=false, size=(1.2*w, 1.2*h),
        xticks=x_ticks,
        xrotation=45,
        yticks=(0:20:100),
        ylims=(0.0, 105.0),
        )

    ##### Row 1: predictive power #####
    options = (
        color=colors,
        linestyle=:dash,
        linewidth=3,
        markershape=:circle,
        markeralpha=0.5,
        markersize=6,
        markerstrokewidth=0,
    )

    # underdetermined
    @df df1 plot!(fig, xs, :test_acc;
        title="m=500, n=10,000",
        ylabel="test accuracy (%)",
        group=:algorithm,
        subplot=1,
        options...
        )
    
    # overdetermined
    @df df2 plot!(fig, xs, :test_acc;
        title="m=10,000, n=500",
        group=:algorithm,
        subplot=2,
        options...
        )    

    ##### Row 2: estimation power #####
    options = (
        color=colors,
        linestyle=:dash,
        linewidth=3,
        markeralpha=0.5,
        markersize=6,
        markerstrokewidth=0,
        )

    # underdetermined
    @df df1 plot!(fig, xs, ppv.(:TP, :FP, :TN, :FN, k0/10^4);
        xlabel="sparsity (%)",
        ylabel="estimation power (%)",
        group=:algorithm,
        shape=:circle,
        subplot=3,
        label=["PPV" ""],
        options...
        )

    @df df1 plot!(fig, xs, npv.(:TP, :FP, :TN, :FN, k0/10^4);
        group=:algorithm,
        shape=:utriangle,
        subplot=3,
        label=["NPV" ""],
        options...
    )

    # overdetermined
    @df df2 plot!(fig, xs, ppv.(:TP, :FP, :TN, :FN, k0/10^4);
        xlabel="sparsity (%)",
        group=:algorithm,
        shape=:circle,
        subplot=4,
        label=["PPV" ""],
        options...
        )

    @df df2 plot!(fig, xs, npv.(:TP, :FP, :TN, :FN, k0/10^4);
        group=:algorithm,
        shape=:utriangle,
        subplot=4,
        label=["NPV" ""],
        options...
    )

    # add guides for true sparsity
    true_sparsity = findall(isapprox(1-k0/10^4), sparsity ./ 100)
    vline!(fig, true_sparsity, subplot=1, label=nothing, lw=1, ls=:solid, color=:black)
    vline!(fig, true_sparsity, subplot=3, label=nothing, lw=1, ls=:solid, color=:black)

    true_sparsity = findall(isapprox(1-k0/500), sparsity ./ 100)
    vline!(fig, true_sparsity, subplot=2, label=nothing, lw=1, ls=:solid, color=:black)
    vline!(fig, true_sparsity, subplot=4, label=nothing, lw=1, ls=:solid, color=:black)

    return fig
end

function figure2b(MM_df, SD_df)
    k0 = 50 # should get this from table

    function subset_max_accuracy(df, col)
        idx = [argmax(gdf.test_acc) for gdf in groupby(df, col)]
        DataFrame(
            gdf[idx[i],:] for (i,gdf) in enumerate(groupby(df, col))
        )
    end

    # underdetermined
    df1 = vcat(
        subset_max_accuracy(filter(:m => m -> m == 500, MM_df), :n),
        subset_max_accuracy(filter(:m => m -> m == 500, SD_df), :n),
    )

    # overdetermined
    df2 = vcat(
        subset_max_accuracy(filter(:n => n -> n == 500, MM_df), :m),
        subset_max_accuracy(filter(:n => n -> n == 500, SD_df), :m),
    )

    # Initialize the plot
    global MM_COLOR
    global SD_COLOR
    colors = [MM_COLOR SD_COLOR]

    w, h = default(:size)
    fig = plot(layout=grid(3, 2), legend=false, grid=false, size=(1.2*w, 1.2*h))
    options = (
        legend=:topleft,
        xscale=:log10,
        color=colors,
        linestyle=:dash,
        linewidth=3,
        markershape=:circle,
        markeralpha=0.5,
        markersize=6,
        markerstrokewidth=0,
    )

    ##### Row 1: iterations
    @df df1 plot!(fig, :n, :iter; group=:algorithm, subplot=1, title="m = 500", ylabel="iterations", options...)
    @df df2 plot!(fig, :m, :iter; group=:algorithm, subplot=2, title="n = 500", options...)

    ##### Row 2: time
    @df df1 plot!(fig, :n, :time; group=:algorithm, subplot=3, ylabel="time (s)", options...)
    @df df2 plot!(fig, :m, :time; group=:algorithm, subplot=4, options...)

    ##### Row 3: estimation power
    ms = repeat(unique(df2.m), 2)
    ns = repeat(unique(df1.n), 2)

    options = (
        legend=:right,
        ylims=(0.0, 105.0),
        xscale=:log10,
        color=colors,
        linestyle=:dash,
        linewidth=3,
        markeralpha=0.5,
        markersize=6,
        markerstrokewidth=0,
        )

    # underdetermined
    @df df1 plot!(fig, ns, ppv.(:TP, :FP, :TN, :FN, k0 ./ cols(:n)); group=:algorithm, subplot=5, xlabel="no. features (n)", ylabel="estimation power (%)", markershape=:circle, label=["PPV" ""], options...)
    @df df1 plot!(fig, ms, npv.(:TP, :FP, :TN, :FN, k0 / 500); group=:algorithm, subplot=5, markershape=:utriangle, label=["NPV" ""], options...)

    # overdetermined
    @df df2 plot!(fig, ns, ppv.(:TP, :FP, :TN, :FN, k0 ./ cols(:n)); group=:algorithm, subplot=6, xlabel="no. samples (m)", markershape=:circle, label=["PPV" ""], options...)
    @df df2 plot!(fig, ms, npv.(:TP, :FP, :TN, :FN, k0 / 500); group=:algorithm, subplot=6, markershape=:utriangle, label=["NPV" ""], options...)

    return fig
end

# Get script arguments.
idir = ARGS[1]
odir = ARGS[2]

dir = joinpath(idir, "experiment2")
files = readdir(dir, join=true)
    
# Filter for lastest results for Experiment 2.
filter!(is_valid_file, files)
MM_file, SD_file = filter_latest(files)

println("""
    Processing...
        - MM file: $(MM_file)
        - SD file: $(SD_file)
"""
)

MM_df = CSV.read(MM_file, DataFrame)
add_column!(MM_df, :MM)

SD_df = CSV.read(SD_file, DataFrame)
add_column!(SD_df, :SD)

fig2A = figure2a(MM_df, SD_df)
savefig(fig2A, joinpath(odir, "Fig2A.png"))

fig2B = figure2b(MM_df, SD_df)
savefig(fig2B, joinpath(odir, "Fig2B.png"))
