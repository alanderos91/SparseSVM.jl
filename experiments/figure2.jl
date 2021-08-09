using Dates, DataFrames, CSV, Statistics, Plots, StatsPlots, LaTeXStrings

const PALETTE = palette(:tab10)
const MM_COLOR = PALETTE[1]
const SD_COLOR = PALETTE[2]

default(:foreground_color_legend, nothing)
default(:background_color_legend, nothing)
default(:fontfamily, "Computer Modern")
default(:dpi, 600)
default(:legendfontsize, 8)

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

function subset_true_sparsity(df, col, k0)
    gdfs = groupby(df, col)
    f(n) = round(100 * (1 - k0 / n), sigdigits=4)
    idx = [findfirst(s -> isapprox(f(gdf.n[1]), s), gdf.sparsity) for gdf in gdfs]
    DataFrame(
        gdf[idx[i],:] for (i,gdf) in enumerate(gdfs)
    )
end

# FDR; adjusted for prevalence p
function FDR(TP, FP, TN, FN, p)
    TPR = TP / (TP + FN) # sensitivity
    FPR = FP / (TN + FP) # 1 - specificity
    v = 100 * FPR * (1-p) / ( TPR * p + FPR * (1-p) )
    v = isnan(v) ? zero(v) : isinf(v) ? 100 : v
    return v
end

# FOR; adjusted for prevalence p
function FOR(TP, FP, TN, FN, p)
    TNR = TN / (TN + FP) # specificity
    FNR = FN / (TP + FN) # 1 - sensitivity
    v = 100 * FNR * p / ( FNR * p + TNR * (1-p) )
    v = isnan(v) ? zero(v) : isinf(v) ? 100 : v
    return v
end

function figure2a(MM_df, SD_df)
    k0 = 50 # should get this from table
    dimensions = (
        (500, 10^4), # underdetermined 
        (10^4, 500), # overdetermined
    )

    # Initialize the plot
    global MM_COLOR
    global SD_COLOR
    colors = [MM_COLOR SD_COLOR]

    sparsity = unique(MM_df.sparsity)
    x_index = 1:length(sparsity)
    x_ticks = (x_index, sparsity)
    xs = repeat(x_index, 2)

    nrows = 3
    ncols = 2

    w, h = default(:size)
    fig = plot(layout=grid(nrows, ncols), grid=false, size=(1.2*w, 1.5*h),
        xticks=x_ticks,
        xrotation=45,
    )

    get_ylabel(idx, text) = idx == 1 ? text : ""

    for (idx, (n, p)) in enumerate(dimensions)
        col_shift = idx - 1

        df = vcat(
            filter([:m, :n] => (x, y) -> x == n && y == p, MM_df),
            filter([:m, :n] => (x, y) -> x == n && y == p, SD_df),
        )
        prevalence = k0 / p
        true_idx = findall(isapprox(1 - prevalence), sparsity ./ 100)

        options = (
            color=colors,
            linestyle=:dash,
            linewidth=3,
            shape=[:circle :utriangle],
            markeralpha=0.5,
            markersize=6,
            markerstrokewidth=0,
        )

        ##### Row 1: Predictive Power
        sp = 1 + col_shift
        @df df plot!(fig, xs, :test_acc;
            group=:algorithm,
            title=latexstring("n=$n,\\ p=$p"),
            ylabel=get_ylabel(idx, "test accuracy (%)"),
            ylims=(0, 105),
            legend=:bottomright,
            subplot=sp,
            options...
        )
        vline!(fig, true_idx, subplot=sp, label=nothing, lw=1, ls=:solid, color=:black)

        ##### Row 2: FDR #####
        sp = 1 + ncols + col_shift
        @df df plot!(fig, xs, FDR.(:TP, :FP, :TN, :FN, prevalence);
            group=:algorithm,
            ylabel=get_ylabel(idx, "FDR (%)"),
            ylims=(0, 105),
            legend=:topright,
            subplot=sp,
            options...
        )
        vline!(fig, true_idx, subplot=sp, label=nothing, lw=1, ls=:solid, color=:black)

        ##### Row 3: FOR #####
        sp = 1 + 2*ncols + col_shift
        @df df plot!(fig, xs, FOR.(:TP, :FP, :TN, :FN, prevalence);
            group=:algorithm,
            # xlabel="sparsity (%)",
            ylabel=get_ylabel(idx, "FOR (%)"),
            ylims=(0, 10),
            legend=:topright,
            subplot=sp,
            options...
        )
        vline!(fig, true_idx, subplot=sp, label=nothing, lw=1, ls=:solid, color=:black)
    end

    return fig
end

function figure2b(MM_df, SD_df)
    k0 = 50 # should get this from table

    dimension = ( (:m, :n, 500), (:n, :m, 500) )

    df = [
        # underdetermined
        vcat(
            subset_true_sparsity(filter(:m => x -> x == 500, MM_df), :n, k0),
            subset_true_sparsity(filter(:m => x -> x == 500, SD_df), :n, k0),
        ),
        # overdetermined
        vcat(
            subset_true_sparsity(filter(:n => x -> x == 500, MM_df), :m, k0),
            subset_true_sparsity(filter(:n => x -> x == 500, SD_df), :m, k0),
        )
    ]

    # Initialize the plot
    global MM_COLOR
    global SD_COLOR
    colors = [MM_COLOR SD_COLOR]

    nrows = 4
    ncols = 2

    w, h = default(:size)
    fig = plot(layout=grid(nrows, ncols), legend=:topleft, grid=false, size=(1.2*w, 1.5*h))
    options = (
        xscale=:log10,
        color=colors,
        linestyle=:dash,
        linewidth=3,
        markershape=[:circle :utriangle],
        markeralpha=0.5,
        markersize=6,
        markerstrokewidth=0,
    )
    
    xlabel = ["# features (p)", "# samples (n)"]
    get_ylabel(idx, text) = idx == 1 ? text : ""

    for idx in eachindex(df)
        data = df[idx]
        col_shift = idx - 1
        fixed_dim, free_dim, fixed_val = dimension[idx]
        title = latexstring(fixed_dim == :m ? :n : :p, " = ", fixed_val)
        dimension_vals = repeat(unique(data[!, free_dim]), 2)
        
        feature_vals = unique(data[!, :n])
        if length(feature_vals) > 1
            prevalence = k0 ./ repeat(feature_vals, 2)
        else
            prevalence = k0 / feature_vals[1]
        end

        ##### Row 1: Iterations #####
        sp = 1 + col_shift
        @df data plot!(fig, dimension_vals, :iter; group=:algorithm, subplot=sp, title=title, ylabel=get_ylabel(idx, "iterations"), options...)

        ##### Row 2: Time #####
        sp = 1 + ncols + col_shift
        @df data plot!(fig, dimension_vals, :time; group=:algorithm, subplot=sp, ylabel=get_ylabel(idx, "time (s)"), yscale=:log10, options...)

        ##### Row 3: FDR #####
        sp = 1 + 2*ncols + col_shift
        @df data plot!(fig, dimension_vals, FDR.(:TP, :FP, :TN, :FN, prevalence); group=:algorithm, subplot=sp, ylabel=get_ylabel(idx, "FDR (%)"), ylims=(0, 105), options...)

        ##### Row 4: FOR #####
        sp = 1 + 3*ncols + col_shift
        @df data plot!(fig, dimension_vals, FOR.(:TP, :FP, :TN, :FN, prevalence); group=:algorithm, subplot=sp, ylabel=get_ylabel(idx, "FOR (%)"), ylims=(0, 10), xlabel=xlabel[idx], options...)
    end

    return fig
end

function main()
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
end
