using SparseSVM, MLDataUtils, KernelFunctions, LinearAlgebra
using CSV, DataFrames, Random, Statistics
using DelimitedFiles, ProgressMeter

include("load_data.jl")
include("common.jl")

function accuracy_score(classifier, X, targets)
    predictions = classifier.(eachrow(X))
    n = length(predictions)
    return sum(predictions .== targets) / n * 100
end

function run_experiment(algorithm::AlgOption, dataset, grid, ctype=MultiClassifier;
    nfolds::Int=10,
    percent_train::Real=0.8,
    tol::Real=1e-6,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    seed::Int=1234,
    strategy::MultiClassStrategy=OVR(),
    kernel::Union{Nothing,Kernel}=nothing,
    intercept=false,
    )
    # Load the data
    labeled_data = load_data(dataset, seed=seed)
    
    # Create the train and test data sets.
    cv_set, (test_X, test_targets) = splitobs(labeled_data, at=percent_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)
    gridvals = sort!(unique(grid), rev=false) # iterate from least sparse to most sparse models
    nvals = length(gridvals)

    # Open output file.
    results = open("experiments/$(dataset)/experiment3-$(string(algorithm)).out", "w")
    writedlm(results, ["fold" "sparsity" "k" "iterations" "objective" "distance" "train_accuracy" "validation_accuracy" "test_accuracy"])

    # run cross-validation
    p = Progress(nfolds*nvals, 1, "Running CV... ")
    for (j, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # get training set and validation set
        ((train_X, train_targets), (val_X, val_targets)) = fold

        # create classifier and use same initial point
        classifier = make_classifier(ctype, train_X, train_targets, first(train_targets), kernel=kernel, intercept=intercept, strategy=strategy)
        initialize_weights!(MersenneTwister(1903), classifier)

        # create SVD if needed
        A, Asvd = SparseSVM.get_A_and_SVD(classifier)

        # follow path along sparsity sets
        for (i, s) in enumerate(gridvals)
            # compute sparsity parameter k
            nparams = kernel isa Nothing ? size(train_X, 2) : size(train_X, 1)
            k = round(Int, nparams*(1-s))

            # train classifier enforcing k nonzero parameters
            iters, obj, dist = trainMM!(classifier, A, f, tol, k, fullsvd=Asvd, nouter=nouter, ninner=ninner, mult=mult, init=false, verbose=false)

            # compute evaluation metrics
            train_acc = accuracy_score(classifier, train_X, train_targets)
            val_acc = accuracy_score(classifier, val_X, val_targets)
            test_acc = accuracy_score(classifier, test_X, test_targets)

            writedlm(results, Any[j s*100 k iters obj dist train_acc val_acc test_acc])
            next!(p, showvalues=[(:fold, j), (:k, k)])
        end
    end
    close(results)

    return nothing
end
# Make sure we set up BLAS threads correctly
BLAS.set_num_threads(10)

##### Example 1: synthetic #####
if "synthetic" in ARGS
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
    println("Running 'synthetic' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "synthetic", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-6, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing)

        # run
        run_experiment(opt, "synthetic", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-6, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=nothing)
    end
end

##### Example 2: iris #####
if "iris" in ARGS
    println("Running 'iris' benchmark")
    grid = [0.0, 0.25, 0.5, 0.75]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "iris", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-6, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "iris", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-6, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 3: spiral #####
if "spiral300" in ARGS
    println("Running 'spiral300' benchmark")
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "spiral300", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-6, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=RBFKernel())

        # run
        run_experiment(opt, "spiral300", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-6, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=RBFKernel())
    end
end

##### Example 4: letter-recognition #####
if "letter-recognition" in ARGS
    println("Running 'letter-recognition' benchmark")
    grid = [i/16 for i in 0:15]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "letter-recognition", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "letter-recognition", grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=50, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 5: MNIST-digits #####
# if "MNIST-digits" in ARGS
#     println("Running 'MNIST-digits' benchmark")
#     grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
#     # for opt in (SD, MM)
#     for opt in (SD,) # MM algorithm requires too much memory
#         # precompile
#         run_experiment(opt, "MNIST", grid, MultiClassifier{Float64},
#             intercept=true, nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
#             kernel=RBFKernel(), strategy=OVO())

#         # run
#         run_experiment(opt, "MNIST", grid, MultiClassifier{Float64},
#             nfolds=10, tol=1e-4, ninner=10^5, nouter=50, mult=1.2,
#             intercept=true, kernel=RBFKernel(), strategy=OVO())
#     end
# end
