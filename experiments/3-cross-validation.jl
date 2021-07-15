using SparseSVM, MLDataUtils, KernelFunctions, LinearAlgebra
using CSV, DataFrames, Random, Statistics
using DelimitedFiles, ProgressMeter

include("common.jl")

function run_experiment(algorithm::AlgOption, dataset, grid, ctype=MultiClassifier;
    nfolds::Int=10,
    percent_train::Real=0.8,
    tol::Real=1e-6,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    scale::Symbol=:zscore,
    strategy::MultiClassStrategy=OVR(),
    kernel::Union{Nothing,Kernel}=nothing,
    intercept=false,
    )
    # Load the data
    df = SparseSVM.dataset(dataset)
    y, X = Vector(df[!, 1]), Matrix{Float64}(df[!, 2:end])

    if dataset == "TCGA-PANCAN-HiSeq"
        # The TCGA data has a few columns full of 0s that should be dropped
        idxs = findall(i -> isequal(0, maximum(X[:,i])), 1:size(X,2))
        selected = setdiff(1:size(X,2), idxs)

        # Limit to the first 10000 genes
        X = X[:, selected[1:10000]]
    end

    _rescale_!(Val(scale), X)
    labeled_data = (X, y)
    
    # Create the train and test data sets.
    cv_set, (test_X, test_targets) = splitobs(labeled_data, at=percent_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)
    gridvals = sort!(unique(grid), rev=false) # iterate from least sparse to most sparse models
    nvals = length(gridvals)

    # Generate short name for kernel option.
    if kernel isa RBFKernel
        kernelopt = "RBF"
    elseif kernel isa Nothing
        kernelopt = "NONE"
    else
        error("Add short name for $(kernel).")
    end

    # Open output file.
    !isdir("results/$(dataset)") && mkdir("results/$(dataset)")
    results = open("results/$(dataset)/3-kernel=$(kernelopt)-$(string(algorithm)).out", "w")
    writedlm(results, ["fold" "sparsity" "iterations" "objective" "distance" "train_accuracy" "validation_accuracy" "test_accuracy"])

    # run cross-validation
    p = Progress(nfolds*nvals, 1, "Running CV... ")
    for (j, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # get training set and validation set
        ((train_X, train_targets), (val_X, val_targets)) = fold

        # create classifier and use same initial point
        classifier = make_classifier(ctype, train_X, train_targets, first(train_targets), kernel=kernel, intercept=intercept, strategy=strategy)

        # create design matrix and SVD (if needed)
        A, Asvd = SparseSVM.get_A_and_SVD(classifier)

        # initialize weights with univariate solution
        initialize_weights!(classifier, A)

        # follow path along sparsity sets
        for (i, s) in enumerate(gridvals)
            # train classifier enforcing k nonzero parameters
            iters, obj, dist = trainMM!(classifier, A, f, tol, s, fullsvd=Asvd, nouter=nouter, ninner=ninner, mult=mult, init=false, verbose=false)

            # compute evaluation metrics
            train_acc = round(accuracy_score(classifier, train_X, train_targets)*100, sigdigits=4)
            val_acc = round(accuracy_score(classifier, val_X, val_targets)*100, sigdigits=4)
            test_acc = round(accuracy_score(classifier, test_X, test_targets)*100, sigdigits=4)
            s_percent = round(s*100, sigdigits=4)

            writedlm(results, Any[j s_percent iters obj dist train_acc val_acc test_acc])
            next!(p, showvalues=[(:fold, j), (:sparsity, s_percent)])
        end
    end
    close(results)

    return nothing
end
# Make sure we set up BLAS threads correctly
BLAS.set_num_threads(10)

##### Example 1: synthetic #####
if "synthetic" in ARGS
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99; 0.996]
    println("Running 'synthetic' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "synthetic", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing)

        # run
        run_experiment(opt, "synthetic", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=10^4, nouter=50, mult=1.2,
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
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "iris", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 3: spiral #####
if "spiral" in ARGS
    println("Running 'spiral' benchmark")
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "spiral", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=RBFKernel(), strategy=OVO())

        # run
        run_experiment(opt, "spiral", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=RBFKernel(), strategy=OVO())
    end
end

##### Example 4: letter-recognition #####
if "letter-recognition" in ARGS
    println("Running 'letter-recognition' benchmark")
    grid = [i/16 for i in 0:15]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "letter-recognition", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "letter-recognition", grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=50, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 5: breast-cancer-wisconsin #####
if "breast-cancer-wisconsin" in ARGS
    println("Running 'breast-cancer-wisconsin' benchmark")
    grid = [i/10 for i in 0:9]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "breast-cancer-wisconsin", grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:none,
            intercept=true, kernel=nothing)

        # run
        run_experiment(opt, "breast-cancer-wisconsin", grid, BinaryClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=50, mult=1.2, scale=:none,
            intercept=true, kernel=nothing)
    end
end

##### Example 6: splice #####
if "splice" in ARGS
    println("Running 'splice' benchmark")
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "splice", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.1, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "splice", grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.1, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 7: TCGA-PANCAN-HiSeq #####
if "TCGA-PANCAN-HiSeq" in ARGS
    println("Running 'TCGA-PANCAN-HiSeq' benchmark")
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "TCGA-PANCAN-HiSeq", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.05, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "TCGA-PANCAN-HiSeq", grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=200, mult=1.05, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 8: optdigits #####
if "optdigits" in ARGS
    println("Running 'optdigits' benchmark")
    grid = [0.0; 0.5:0.05:0.95; 0.96:0.01:0.99]
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "optdigits", grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:none,
            intercept=true, kernel=nothing, strategy=OVO(), percent_train=0.68)

        # run
        run_experiment(opt, "optdigits", grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=50, mult=1.2, scale=:none,
            intercept=true, kernel=nothing, strategy=OVO(), percent_train=0.68)
    end
end
