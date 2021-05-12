using SparseSVM
using CSV, DataFrames, Random, Statistics

include("load_data.jl")
include("common.jl")

function run_experiment(algorithm::AlgOption, dataset, k_grid;
    percent_train::Real=0.8,
    tol::Real=1e-6,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    seed::Int=1234,
    )
    # Load the data.
    df, X, classes = load_data(dataset, seed=seed)
    class_mapping = Dict(c => i for (i, c) in enumerate(classes))
    y = Vector{Float64}(df.Class)
    nsamples, nfeatures = size(X)

    # Load info for ground truth.
    truth = CSV.read("data/$(dataset)-info.csv", DataFrame, header=true)
    J = Int.(truth.index)
    coef = truth.coefficient

    # Process options.
    f = get_algorithm_func(algorithm)
    
    # Create the train and test data sets.
    ntrain = round(Int, percent_train * nsamples)
    train_set = 1:ntrain
    test_set = ntrain+1:nsamples
    Xtrain, Xtest = X[train_set, :], X[test_set, :]
    ytrain, ytest = y[train_set], y[test_set]

    # Pre-compile.
    b_init = randn(MersenneTwister(seed), nfeatures)
    classifier = make_classifier(BinarySVMClassifier, nfeatures, classes)
    copyto!(classifier.b, b_init)
    @timed trainMM(classifier, f, ytrain, Xtrain, tol, k_grid[1],
        nouter=2, ninner=2, mult=mult)

    # Write header to main results file.
    results = open("experiments/$(dataset)/experiment2-$(string(algorithm)).out", "w")
    write(results, "k\tMSE1\tMSE2\tTP\tFP\tTN\tFN\tscore\tpercent\ttime\n")
    for k in k_grid
        # Feed STDOUT output to a log file.
        io = open("experiments/$(dataset)/experiment2-k=$(k)-$(string(algorithm)).log", "w")
        redirect_stdout(io)

        # Train using the chosen algorithm.
        copyto!(classifier.b, b_init)
        r = @timed trainMM(classifier, f, ytrain, Xtrain, tol, k,
        nouter=nouter, ninner=ninner, mult=mult)

        # Test the classifier against the test data.
        ypredict = map(xi -> classify(classifier, xi), eachrow(Xtest))
        score = sum(ypredict .== ytest)
        percent = round(score / length(ytest) * 100, sigdigits=2)

        # Check quality of solution against ground truth.
        b = classifier.b
        b0 = zero(b)
        b0[J] .= coef 

        MSE1 = mse(b, b0)
        MSE2 = b[end] .^ 2 # assume bias = 0
        (TP, FP, TN, FN) = discovery_metrics(b[1:nfeatures-1], b0[1:nfeatures-1])

        # Write to results file.
        write(results, "$(k)\t$(MSE1)\t$(MSE2)\t$(TP)\t$(FP)\t$(TN)\t$(FN)\t$(score)\t$(percent)\t$(r.time)\n")

        # Close log file.
        close(io)
    end

    # Close file.
    close(results)
end

# Make sure we set up BLAS threads correctly
using LinearAlgebra
BLAS.set_num_threads(10)

k_grid = 501:-1:0

for dataset in ARGS
    for opt in (MM, SD)
        run_experiment(opt, dataset, k_grid, tol=1e-6, ninner=10^4, nouter=50, mult=1.2)
    end
end