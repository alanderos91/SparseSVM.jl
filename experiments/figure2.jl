include("common.jl")

using CairoMakie

function calculate_performance_metrics(df, new_col, cols)
    TP, FP, TN, FN = map(col -> df[!,col], cols)

    P = TP + FN
    N = TN + FP
    T = TP + FP + TN + FN

    prevalence = @. P / T
    sensitivity = @. TP / P
    specificity = @. TN / N
    accuracy = @. (TP + N) / T

    a = @. sensitivity * prevalence
    b = @. (1-specificity) * (1-prevalence)
    PPV = @. a / (a + b)

    a = @. specificity * (1-prevalence)
    b = @. (1-sensitivity) * prevalence
    NPV = @. a / (a + b)

    df[!,new_col[1]] = prevalence
    df[!,new_col[2]] = sensitivity
    df[!,new_col[3]] = specificity
    df[!,new_col[4]] = accuracy
    df[!,new_col[5]] = PPV
    df[!,new_col[6]] = NPV

    return df
end

function get_true_sparsity(df)
    a = 100 * (1 - df.ncausal[1] / df.nvars[1])
    b = findfirst(isapprox(a), df.sparsity)
    return a, b
end

function add_plot(ax, dfs, cols, labels, default_kwargs)
    mm, sd = dfs
    xcol, ycol = cols
    xlabel, ylabel = labels

    # MM path
    stairs!(ax, mm[!,xcol], mm[!,ycol];
        color=(:navy, 1.0),
        default_kwargs...,
    )
    scatter!(ax, mm[!,xcol], mm[!,ycol];
        color=(:navy, 1.0),
        marker=:circle,
        label="MM",
        default_kwargs...,
    )

    # SD path
    stairs!(ax, sd[!,xcol], sd[!,ycol];
        color=(:orange, 1.0),
        default_kwargs...,
    )
    scatter!(ax, sd[!,xcol], sd[!,ycol];
        color=(:orange, 1.0),
        marker=:utriangle,
        label="SD",
        default_kwargs...,
    )

    # Change tick labels so that we plot the x values on an irregular grid.
    ax.xlabel = xlabel
    ax.ylabel = ylabel

    return nothing
end

function add_irregular_plot(ax, dfs, idx, cols, labels, default_kwargs)
    mm, sd = dfs
    xcol, ycol = cols
    xlabel, ylabel = labels
    _, true_sparsity_idx = get_true_sparsity(mm[idx,:])

    # MM path
    stairs!(ax, mm[idx,ycol];
        color=(:navy, 1.0),
        default_kwargs...,
    )
    scatter!(ax, mm[idx,ycol];
        color=(:navy, 1.0),
        marker=:circle,
        label="MM",
        default_kwargs...,
    )

    # SD path
    stairs!(ax, sd[idx,ycol];
        color=(:orange, 1.0),
        default_kwargs...,
    )
    scatter!(ax, sd[idx,ycol];
        color=(:orange, 1.0),
        marker=:utriangle,
        label="SD",
        default_kwargs...,
    )

    # Highlight the true sparsity level.
    vlines!(ax, true_sparsity_idx, color=:black, linestyle=:dot, linewidth=5)

    # Change tick labels so that we plot the x values on an irregular grid.
    ax.xticklabelrotation = pi/4
    ax.xticks = 1:length(idx)
    if eltype(mm[!,xcol]) <: AbstractFloat
        ax.xtickformat = _ -> map(x -> string(round(x, sigdigits=4)), mm[idx,xcol])
    elseif eltype(mm[!,xcol]) <: Integer
        ax.xtickformat = _ -> map(mm[idx,xcol]) do x
            if x ≥ 1000
                v = string(round(x / 1000, sigdigits=2), "k")
            else
                v = string(round(Int, x))
            end
            return v
        end
    end
    ax.xlabel = xlabel
    ax.ylabel = ylabel

    return nothing
end

function add_irregular_plot2(ax, dfs, idx, cols, labels, default_kwargs)
    mm, sd = dfs
    xcol, ycol = cols
    xlabel, ylabel = labels
    _, true_sparsity_idx = get_true_sparsity(mm[idx,:])

    # MM path
    stairs!(ax, mm[idx,ycol];
        color=(:navy, 1.0),
        default_kwargs...,
    )
    scatter!(ax, mm[idx,ycol];
        color=(:navy, 1.0),
        marker=:circle,
        label="MM",
        default_kwargs...,
    )

    # SD path
    stairs!(ax, sd[idx,ycol];
        color=(:orange, 1.0),
        default_kwargs...,
    )
    scatter!(ax, sd[idx,ycol];
        color=(:orange, 1.0),
        marker=:utriangle,
        label="SD",
        default_kwargs...,
    )

    # Change tick labels so that we plot the x values on an irregular grid.
    ax.xticklabelrotation = pi/4
    ax.xticks = 1:length(idx)
    if eltype(mm[!,xcol]) <: AbstractFloat
        ax.xtickformat = _ -> map(x -> string(round(x, sigdigits=4)), mm[idx,xcol])
    elseif eltype(mm[!,xcol]) <: Integer
        ax.xtickformat = _ -> map(mm[idx,xcol]) do x
            if x ≥ 1000
                v = string(round(x / 1000, sigdigits=2), "k")
            else
                v = string(round(Int, x))
            end
            return v
        end
    end
    ax.xlabel = xlabel
    ax.ylabel = ylabel

    return nothing
end

function add_row_label(loc, label, fz)
    Box(loc, color=:gray90)
    Label(loc, label, rotation=pi/2, tellheight=false, textsize=2*fz)
end

function add_col_label(loc, label, fz)
    Box(loc, color=:gray90)
    Label(loc, label, tellwidth=false, textsize=2*fz)
end

function get_subset(df, fixed_col, fixed_value)
    filter(row -> row.ncausal == row.k && getproperty(row, fixed_col) == fixed_value, df)
end

function main(args)
    fz = 16
    w, h = 450, 250
    default_kwargs = (;
        markersize=20,
        linewidth=5,
    )
    axis_kwargs = (;
        width=w,
        height=h,
        xlabelsize=2*fz,
        ylabelsize=2*fz,
        xticklabelsize=1.5*fz,
        yticklabelsize=1.5fz,
    )
    dir, outdir = args[1], args[2]
    pad = 5e-2

    grouping_cols = [:nsamples, :nvars, :ncausal]
    mmdf = CSV.read(joinpath(dir, "MMSVD.out"), DataFrame) |> Base.Fix2(unique!, [grouping_cols; :k])
    sddf = CSV.read(joinpath(dir, "SD.out"), DataFrame) |> Base.Fix2(unique!, [grouping_cols; :k])

    ##### IMPORTANT #####
    # Filter out cases where k = 0 implying w = 0.
    # Predictions will be sgn(x_iᵀ*0) = ±0.0 will artificially arrive at the correct label prediction.
    # This is because decisions f(-0.0) = -1 and f(0.0) = +1 due to MLDataUtils.LabelEnc.MarginBased,
    # and the labels were generated exactly using sgn(x_iᵀ*w).
    # In otherwords, it is a useful check but otherwise meaningless.
    filter!(row -> row.k != 0, mmdf)
    filter!(row -> row.k != 0, sddf)
    
    data_cols = (:TP, :FP, :TN, :FN)
    metrics = (:Prevalence, :Sensitivity, :Specificity, :Accuracy, :PPV, :NPV)
    for df in (mmdf, sddf), col_group in (:w, :train, :test)
        inputs = map(x -> Symbol(col_group, x), data_cols)
        outputs = map(x -> Symbol(col_group, x), metrics)
        calculate_performance_metrics(df, outputs, inputs)
    end

    mmgdf = groupby(mmdf, grouping_cols)
    sdgdf = groupby(sddf, grouping_cols)

    fig = Figure(resolution=(800,600))

    # Figure 2A
    fig2A = fig[1,1] = GridLayout()
    example_dfs = ( (mmgdf[5], sdgdf[5]), (mmgdf[14], sdgdf[14]) )
    xcols = (:k, :k, :k, :k)
    ycols = (:trainAccuracy, :testAccuracy, :wPPV, :wNPV)
    lookup_label = Dict(
        :k => latexstring("Active Variables ", L"k"),
        :trainAccuracy => "Accuracy",
        :testAccuracy => "Accuracy",
        :wPPV => "PPV",
        :wNPV => "NPV",
    )
    
    for (i, (mm, _)) in enumerate(example_dfs)
        loc = fig2A[1+i,1]
        n, p = mm.nsamples[1], mm.nvars[1]
        label = "$n samples\n$p variables"
        add_row_label(loc, label, fz)
    end

    add_col_label(fig2A[1,2], "Training Set", fz)
    add_col_label(fig2A[1,3], "Test Set", fz)
    add_col_label(fig2A[1,4:5], "Coefficients", fz)

    ax = [Axis(fig2A[1+i,1+j]; axis_kwargs...) for i in 1:2, j in 1:4]
    for i in 1:2
        for j in 1:2
            ylims!(ax[i,j], 0.5-pad, 1.0+pad)
            ax[i,j].yticks = 0.5:0.1:1.0
        end
        linkaxes!(ax[i,1:2]...)

        for j in 3:4
            ylims!(ax[i,j], 0.0-pad, 1.0+pad)
            ax[i,j].yticks = 0.0:0.2:1.0
        end
        linkaxes!(ax[i,3:4]...)
    end
    
    for (j, (xcol, ycol)) in enumerate(zip(xcols, ycols)), (i, dfs) in enumerate(example_dfs)
        idx = [1:2:16; 17:nrow(dfs[1])]
        xlabel, ylabel = lookup_label[xcol], lookup_label[ycol]
        add_irregular_plot(ax[i,j], dfs, idx, (xcol, ycol), (xlabel, ylabel), default_kwargs)
    end

    # Figure 2B
    fig2B = fig[2,1] = GridLayout()

    nvars = nsamples = 500
    for (i, label) in enumerate(("$nvars variables", "$nsamples samples"))
        loc = fig2B[1+i,1]
        add_row_label(loc, label, fz)
    end

    add_col_label(fig2B[1,2:3], "Training Set", fz)
    add_col_label(fig2B[1,4:5], "Coefficients", fz)

    yscales = [log10, log10, identity, identity]
    ax = [Axis(fig2B[1+i,1+j]; yscale=yscales[j], xscale=log10, xticks=[1e3,1e4,1e5], axis_kwargs...) for i in 1:2, j in 1:4]
    foreach(j -> linkaxes!(ax[1:2,j]...), 1:4)
    linkaxes!(ax[1,3:4]...)

    example_dfs = (
        (get_subset(mmdf, :nvars, nvars), get_subset(sddf, :nvars, nvars)),             # overdetermined
        (get_subset(mmdf, :nsamples, nsamples), get_subset(sddf, :nsamples, nsamples)), # underdetermined 
    )
    xcols =(:nsamples, :nvars)
    xlabels = ("Total Samples", "Total Variables")

    for (i, (dfs, xcol, xlabel)) in enumerate(zip(example_dfs, xcols, xlabels))
        add_plot(ax[i,1], dfs, (xcol, :iters), (xlabel, "Iterations"), default_kwargs)
        add_plot(ax[i,2], dfs, (xcol, :time), (xlabel, "Time [s]"), default_kwargs)
        add_plot(ax[i,3], dfs, (xcol, :wPPV), (xlabel, "PPV"), default_kwargs)
        add_plot(ax[i,4], dfs, (xcol, :wNPV), (xlabel, "NPV"), default_kwargs)

        ylims!(ax[i,4], 0-pad, 1+pad)
        ax[i,3].yticks = LinearTicks(6)
        ax[i,4].yticks = LinearTicks(6)
    end

    # Add manual Legend
    mm_entry = [
        LineElement(color=:navy, linewidth=default_kwargs.linewidth),
        MarkerElement(color=:navy, marker=:circle, markersize=1.5*default_kwargs.markersize)
    ]
    sd_entry = [
        LineElement(color=:orange, linewidth=default_kwargs.linewidth),
        MarkerElement(color=:orange, marker=:utriangle, markersize=1.5*default_kwargs.markersize)
    ]
    fig[3,1] = Legend(fig, [mm_entry, sd_entry], ["MM", "SD"], "Algorithm",
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

    # Add padding on the left and right
    fig[1:2,0] = GridLayout()
    fig[1:2,2] = GridLayout()

    resize_to_layout!(fig)

    save(joinpath(outdir, "Fig2.pdf"), fig, pt_per_unit=2)

    # Supplement: # of support vectors vs sparsity
    fig = Figure(resolution=(800,600))
    g = fig[1,1] = GridLayout()
    fig[1:2,0] = GridLayout()
    fig[1:2,2] = GridLayout()

    example_dfs = [
                    (mmgdf[1], sdgdf[5])      (mmgdf[10], sdgdf[10])
                    (mmgdf[1], sdgdf[5])      (mmgdf[14], sdgdf[14])
                ]

    for i in axes(example_dfs, 1)
        loc = g[1+i,1]
        n = example_dfs[i,1][1].nsamples[1]
        label = "$n samples"
        add_row_label(loc, label, fz)
    end

    for j in axes(example_dfs, 2)
        loc = g[1,1+j]
        p = example_dfs[1,j][1].nsamples[1]
        label = "$p variables"
        add_col_label(loc, label, fz)
    end

    ax = [Axis(g[1+i,1+j]; axis_kwargs...) for i in axes(example_dfs, 1), j in axes(example_dfs, 2)]
    for j in axes(ax, 2), i in axes(ax, 1)
        dfs = example_dfs[i,j]
        idx = [1:2:16; 17:nrow(dfs[1])]
        add_irregular_plot2(ax[i,j], dfs, idx, (:k, :nSV), ("Active Variables", "Support Vectors"), default_kwargs)
    end

    # Add manual Legend
    mm_entry = [
        LineElement(color=:navy, linewidth=default_kwargs.linewidth),
        MarkerElement(color=:navy, marker=:circle, markersize=1.5*default_kwargs.markersize)
    ]
    sd_entry = [
        LineElement(color=:orange, linewidth=default_kwargs.linewidth),
        MarkerElement(color=:orange, marker=:utriangle, markersize=1.5*default_kwargs.markersize)
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

    resize_to_layout!(fig)

    save(joinpath(outdir, "SFig1.pdf"), fig, pt_per_unit=2)

    return nothing
end

main(ARGS)
