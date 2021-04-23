using SparseSVM
using CSV, DataFrames, Random

include("load_data.jl")

# Algorithm Option Definitions
@enum AlgOption SD=1 MM=2

function get_algorithm_func(opt::AlgOption)
    if opt == SD
        return sparse_steepest!
    elseif opt == MM
        return sparse_direct!
    else
        error("Unknown option $(opt)")
    end
end

function run_experiment(algorithm::AlgOption, dataset;
    ntrials::Int=100,
    percent_train::Real=0.8,
    tol::Real=1e-6,
    k::Int=-1,
    nouter::Int=20,
    mult::Real=1.5,
    seed::Int=1234,
    )

    # Load the data
    df, X, classes = load_data(dataset, seed=seed)
    class_mapping = Dict(c => i for (i, c) in enumerate(classes))
    y = [class_mapping[c] for c in df.Class]
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
    classifier = MultiSVMClassifier(nfeatures, classes)
    @timed trainMM(classifier, f, ytrain, Xtrain, tol, k, nouter=nouter, mult=mult)
    
    # Run the experiment several times.
    results = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm)).out", "w")
    write(results, "trial\tscore\tpercent\ttime\n")
    for trial in 1:ntrials
        # Feed STDOUT output to a log file
        io = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm))-$(trial).log", "w")
        redirect_stdout(io)

        # Train using the chosen MM algorithm
        classifier = MultiSVMClassifier(nfeatures, classes)
        r = @timed trainMM(classifier, f, ytrain, Xtrain, tol, k, nouter=nouter, mult=mult)
        
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

##### Example 1: synthetic #####
if "synthetic" in ARGS
    @warn "Example 'synthetic' not yet implemented."
end

##### Example 2: iris #####
if "iris" in ARGS
    for opt in (SD, MM)
        run_experiment(opt, "iris", ntrials=100, tol=1e-6)
    end
end

##### Example 3: letter-recognition #####
if "letter-recognition" in ARGS
    for opt in (SD, MM)
        run_experiment(opt, "letter-recognition", ntrials=100, tol=1e-6)
    end
end

##### Example 4: MNIST-digits #####
if "MNIST-digits" in ARGS
    @warn "Example 'MNIST-digits' not yet implemented."
end
