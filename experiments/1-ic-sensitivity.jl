using SparseSVM, MLDataUtils, KernelFunctions
using CSV, DataFrames, Random
using ProgressMeter

include("load_data.jl")
include("common.jl")

function run_experiment(algorithm::AlgOption, dataset, ctype=MultiClassifier;
    ntrials::Int=100,
    percent_train::Real=0.8,
    tol::Real=1e-6,
    k::Int=-1,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    seed::Int=1234,
    kernel::Union{Nothing,Kernel}=nothing,
    intercept=false,
    )
    # Load the data
    labeled_data = load_data(dataset, seed=seed)
    
    # Create the train and test data sets.
    train_subset, test_subset = splitobs(labeled_data, at=percent_train, obsdim=1)
    Xtrain, targets_train = train_subset
    Xtest, targets_test = test_subset

    # Process options
    f = get_algorithm_func(algorithm)
    nparams = kernel isa Nothing ? size(Xtrain, 2) : size(Xtrain, 1)
    k = k < 0 ? round(Int, 0.5 * nparams) : min(k, nparams)

    # Initialize classifier.
    classifier = make_classifier(ctype, Xtrain, targets_train, first(targets_train), kernel=kernel, intercept=intercept)
    local_rng = MersenneTwister(1903)
    initialize_weights!(local_rng, classifier)

    # Create closure.
    execute(ninner, nouter) = @timed trainMM(classifier, f, tol, k, nouter=nouter, ninner=ninner, mult=mult, init=false)
    
    # Run the experiment several times.
    original_stdout = stdout
    results = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm)).out", "w")
    write(results, "trial\tscore\tpercent\ttime\n")
    @showprogress 1 "Testing $(algorithm)..." for trial in 1:ntrials
        # Feed STDOUT output to a log file
        io = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm))-$(trial).log", "w")
        redirect_stdout(io)

        # Train using the chosen MM algorithm
        initialize_weights!(local_rng, classifier)
        r = execute(ninner, nouter)
        
        # Test the classifier against the test data
        targets_predict = classifier.(eachrow(Xtest))
        score = sum(targets_predict .== targets_test)
        percent = round(score / length(targets_test) * 100, digits=2)

        # Write to results file.
        write(results, "$(trial)\t$(score)\t$(percent)\t$(r.time)\n")

        # Close log file.
        close(io)
    end

    # Close file.
    redirect_stdout(original_stdout)
    close(results)
end

# Make sure we set up BLAS threads correctly
using LinearAlgebra
BLAS.set_num_threads(Sys.CPU_THREADS / 2)

##### Example 1: synthetic #####
if "synthetic" in ARGS
    println("Running 'synthetic' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "synthetic", BinaryClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=2, nouter=2, mult=1.2, intercept=true, kernel=nothing)

        # run
        run_experiment(opt, "synthetic", BinaryClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50, mult=1.2, intercept=true, kernel=nothing)
    end
end

##### Example 2: iris #####
if "iris" in ARGS
    println("Running 'iris' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "iris", MultiClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=2, nouter=2, mult=1.2)

        # run
        run_experiment(opt, "iris", MultiClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50, mult=1.2)
    end
end

##### Example 3: spiral #####
if "spiral300" in ARGS
    println("Running 'spiral300' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "spiral300", BinaryClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=2, nouter=2, mult=1.2, intercept=true, kernel=RBFKernel())

        # run
        run_experiment(opt, "spiral300", BinaryClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=10^4, nouter=50, mult=1.2, intercept=true, kernel=RBFKernel())
    end
end

##### Example 4: letter-recognition #####
if "letter-recognition" in ARGS
    println("Running 'letter-recognition' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "letter-recognition", MultiClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=2, nouter=2, mult=1.2)

        # run
        run_experiment(opt, "letter-recognition", MultiClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=10^5, nouter=50, mult=1.2)
    end
end

##### Example 5: MNIST-digits #####
if "MNIST-digits" in ARGS
    println("Running 'MNIST-digits' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "MNIST", MultiClassifier{Float64}, ntrials=100, tol=1e-6,
        ninner=2, nouter=2  , mult=1.2)

        # run
        run_experiment(opt, "MNIST", MultiClassifier{Float64}, ntrials=100, tol=1e-6,
            ninner=10^5, nouter=50  , mult=1.2)
    end
end
