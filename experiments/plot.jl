using DataFrames, CSV, Statistics, Plots
gr(dpi=300, legend=nothing)

for exp in ("signal-dense", "signal-sparse", "signal-vsparse")
    dfMM = CSV.read("experiments/$(exp)/experiment2-MM.out", DataFrame)
    dfSD = CSV.read("experiments/$(exp)/experiment2-SD.out", DataFrame)

    if exp == "signal-dense"
        sopt = 100 * (1 - 500 / 501)
    elseif exp == "signal-sparse"
        sopt = 100 * (1 - 50 / 501)
    else
        sopt = 100 * (1 - 5 / 501)
    end

    xs = 100 * (1 .- dfMM.k / maximum(dfMM.k))
    fig = plot(layout=grid(2,3))
    for (df, alg) in ((dfMM, "MM"), (dfSD, "SD"))
        FDR = zeros(size(df, 1))
        FNR = zeros(size(df, 1))
        for (i, row) in enumerate(eachrow(df))
            d1 = row.FP + row.TN; iszero(d1) && (d1 = 1)
            d2 = row.FN + row.TP; iszero(d2) && (d2 = 1)
            FDR[i] = row.FP / d1
            FNR[i] = row.FN / d2
        end
        sym = alg == "MM" ? :+ : :x

        scatter!(fig, xs, df.MSE1, label=alg, ylab="MSE", subplot=1, m=(1, sym))
        vline!(fig, [sopt], subplot=1, label=nothing)

        scatter!(fig, xs, df.MSE2, label=alg, ylab="bias", title=exp, subplot=2, m=(1, sym))
        vline!(fig, [sopt], subplot=2, label=nothing)

        scatter!(fig, xs, df.percent, label=alg, ylab="Acc (%)", subplot=3, m=(1, sym), ylim=(0, 100))
        vline!(fig, [sopt], subplot=3, label=nothing)
        
        scatter!(fig, xs, FDR, label=alg, ylab="FDR", subplot=4, m=(1, sym))
        vline!(fig, [sopt], subplot=4, label=nothing)

        scatter!(fig, xs, FNR, label=alg, ylab="FNR", xlab="sparsity (%)", subplot=5, m=(1, sym))
        vline!(fig, [sopt], subplot=5, label=nothing)

        scatter!(fig, xs, df.time, label=alg, ylab="time (s)", subplot=6, m=(1, sym))
        vline!(fig, [sopt], subplot=6, label=nothing)
    end
    savefig(fig, "experiments/$(exp).png")
end
