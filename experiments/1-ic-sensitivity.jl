using SparseSVM
using CSV, DataFrames, Random

include("load_data.jl")
include("common.jl")

function run_experiment(algorithm::AlgOption, dataset, ctype=MultiSVMClassifier;
    ntrials::Int=100,
    percent_train::Real=0.8,
    tol::Real=1e-6,
    k::Int=-1,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    seed::Int=1234,
    )

    # Load the data
    df, X, classes = load_data(dataset, seed=seed)
    class_mapping = Dict(c => i for (i, c) in enumerate(classes))
    if ctype == MultiSVMClassifier
        y = [class_mapping[c] for c in df.Class]
    else
        y = Vector{Float64}(df.Class)
    end
    nsamples, nfeatures = size(X)

    # Process options
    f = get_algorithm_func(algorithm)
    k = k < 0 ? round(Int, 0.5 * nfeatures) : min(k, nfeatures)
    
    # Create the train and test data sets.
    ntrain = round(Int, percent_train * nsamples)
    train_set = 1:ntrain
    test_set = ntrain+1:nsamples
    Xtrain, Xtest = X[train_set, :], X[test_set, :]
    ytrain, ytest = y[train_set], y[test_set]
    
    # Pre-compile
    classifier = make_classifier(ctype, nfeatures, classes)
    @timed trainMM(classifier, f, ytrain, Xtrain, tol, k, nouter=5, ninner=5, mult=mult)
    
    # Run the experiment several times.
    results = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm)).out", "w")
    write(results, "trial\tscore\tpercent\ttime\n")
    for trial in 1:ntrials
        # Feed STDOUT output to a log file
        io = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm))-$(trial).log", "w")
        redirect_stdout(io)

        # Train using the chosen MM algorithm
        classifier = make_classifier(ctype, nfeatures, classes)
        r = @timed trainMM(classifier, f, ytrain, Xtrain, tol, k, nouter=nouter, ninner=ninner, mult=mult)
        
        # Test the classifier against the test data
        ypredict = map(xi -> classify(classifier, xi), eachrow(Xtest))
        score = sum(ypredict .== ytest)
        percent = round(score / length(ytest) * 100, sigdigits=2)

        # Write to results file.
        write(results, "$(trial)\t$(score)\t$(percent)\t$(r.time)\n")

        # Close log file.
        close(io)
    end

    # Close file.
    close(results)
end

# Make sure we set up BLAS threads correctly
using LinearAlgebra
BLAS.set_num_threads(10)

##### Example 1: synthetic #####
if "synthetic" in ARGS
    for opt in (SD, MM)
        run_experiment(opt, "synthetic", BinarySVMClassifier, ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50, mult=1.2)
    end
end

##### Example 2: iris #####
if "iris" in ARGS
    for opt in (SD, MM)
        run_experiment(opt, "iris", ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50, mult=1.2)
    end
end

##### Example 3: letter-recognition #####
if "letter-recognition" in ARGS
    # for opt in (SD, MM)
    for opt in (MM,)
        run_experiment(opt, "letter-recognition", ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50, mult=1.2)
    end
end

##### Example 4: MNIST-digits #####
if "MNIST-digits" in ARGS
    for opt in (SD, MM)
        run_experiment(opt, "MNIST", ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50  , mult=1.2)
    end
end
