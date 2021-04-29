using CSV, DataFrames, Statistics

match_file(alg, x) = occursin(".out", x) && occursin("experiment1", x) && occursin("$(alg)", x)

d = Dict()
datasets = ("synthetic", "iris", "letter-recognition", "MNIST")
algorithms = ("SD", "MM")
metrics = ("iters", "obj", "dist")

for alg in algorithms, dataset in datasets
    # Read the main log file
    files = readdir("experiments/$(dataset)")
    idx = findfirst(x -> match_file(alg, x), files)
    file = "experiments/$(dataset)/$(files[idx])"
    println("Reading $(file)...")
    df = CSV.read(file, DataFrame)

    # Scrape metrics from individual log files
    println("Scraping missing metrics...")
    run(`experiments/scrape.sh "$(dataset)" "$(alg)"`)

    # add information for iteration count, objective, and distance
    for metric in metrics
        tmp = CSV.read("experiments/$(dataset)/experiment1-$(alg)-$(metric).log", DataFrame, header=false)
        df[!, Symbol(metric)] = sum(eachcol(tmp))
    end

    # Add to the dictionary
    println("Done.")
    d["$(alg)/$(dataset)"] = df
end
println()

for (key, df) in d
    println(key)
    println(describe(df))
    println()
end
