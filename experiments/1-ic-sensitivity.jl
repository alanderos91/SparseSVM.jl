using SparseSVM, MLDataUtils, KernelFunctions, LinearAlgebra
using CSV, DataFrames, Random, Statistics
using DelimitedFiles, ProgressMeter

include("common.jl")

function run_experiment(algorithm::AlgOption, dataset, ctype=MultiClassifier;
    ntrials::Int=100,
    percent_train::Real=0.8,
    tol::Real=1e-6,
    s::Float64=0.5,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    scale::Symbol=:zscore,
    strategy=MultiClassStrategy=OVR(),
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
    (train_X, train_targets), (test_X, test_targets) = splitobs(labeled_data, at=percent_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)

    # Generate short name for kernel option.
    if kernel isa RBFKernel
        kernelopt = "RBF"
    elseif kernel isa Nothing
        kernelopt = "NONE"
    else
        error("Add short name for $(kernel).")
    end

    # Initialize classifier.
    classifier = make_classifier(ctype, train_X, train_targets, first(train_targets),
        kernel=kernel, intercept=intercept, strategy=strategy)
    local_rng = MersenneTwister(1903)

    # Create closure.
    A, Asvd = SparseSVM.get_A_and_SVD(classifier)
    execute(ninner, nouter) = @timed trainMM!(classifier, A, f, tol, s, fullsvd=Asvd, nouter=nouter, ninner=ninner, mult=mult, init=false, verbose=false)
    
    # Run the experiment several times.
    # original_stdout = stdout
    results = open("results/$(dataset)/1-kernel=$(kernelopt)-s=$(s)-$(string(algorithm)).out", "w")
    writedlm(results, ["trial" "sparsity" "time" "iterations" "objective" "distance" "train_accuracy" "test_accuracy"])
    @showprogress 1 "Testing $(algorithm)..." for trial in 1:ntrials
        # # Feed STDOUT output to a log file
        # io = open("results/$(dataset)/1-kernel=$(kernelopt)-k=$(k)-$(string(algorithm))-trial=$(trial).log", "w")
        # redirect_stdout(io)

        # Train using the chosen MM algorithm
        randomize_weights!(classifier, local_rng)
        r = execute(ninner, nouter)

        t = r.time
        iters, obj, dist = r.value
        
        # Test the classifier against the test data
        train_acc = round(accuracy_score(classifier, train_X, train_targets)*100, sigdigits=4)
        test_acc = round(accuracy_score(classifier, test_X, test_targets)*100, sigdigits=4)
        sparsity = round(s*100, sigdigits=4)

        # Write to results file.
        writedlm(results, Any[trial sparsity t iters obj dist train_acc test_acc])

        # Close log file.
        # close(io)
    end

    # Close file.
    # redirect_stdout(original_stdout)
    close(results)
end

# Make sure we set up BLAS threads correctly
BLAS.set_num_threads(10)

##### Example 1: synthetic #####
if "synthetic" in ARGS
    println("Running 'synthetic' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "synthetic", BinaryClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing)

        # run
        run_experiment(opt, "synthetic", BinaryClassifier{Float64},
            tol=1e-4, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=nothing)
    end
end

##### Example 2: iris #####
if "iris" in ARGS
    println("Running 'iris' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "iris", MultiClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "iris", MultiClassifier{Float64},
            tol=1e-4, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 3: spiral #####
if "spiral" in ARGS
    println("Running 'spiral' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "spiral", MultiClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=RBFKernel(), strategy=OVO())

        # run
        run_experiment(opt, "spiral", MultiClassifier{Float64},
            tol=1e-4, ninner=10^4, nouter=50, mult=1.2,
            intercept=true, kernel=RBFKernel(), strategy=OVO())
    end
end

##### Example 4: letter-recognition #####
if "letter-recognition" in ARGS
    println("Running 'letter-recognition' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "letter-recognition", MultiClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "letter-recognition", MultiClassifier{Float64}, 
            tol=1e-4, ninner=10^5, nouter=50, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 5: breast-cancer-wisconsin #####
if "breast-cancer-wisconsin" in ARGS
    println("Running 'breast-cancer-wisconsin' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "breast-cancer-wisconsin", BinaryClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:none,
            intercept=true, kernel=nothing)

        # run
        run_experiment(opt, "breast-cancer-wisconsin", BinaryClassifier{Float64}, 
            tol=1e-4, ninner=10^5, nouter=50, mult=1.2, scale=:none,
            intercept=true, kernel=nothing)
    end
end

##### Example 6: splice #####
if "splice" in ARGS
    println("Running 'splice' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "splice", MultiClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.1, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "splice", MultiClassifier{Float64}, 
            tol=1e-4, ninner=10^5, nouter=100, mult=1.1, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 7: TCGA-PANCAN-HiSeq #####
if "TCGA-PANCAN-HiSeq" in ARGS
    println("Running 'TCGA-PANCAN-HiSeq' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "TCGA-PANCAN-HiSeq", MultiClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.05, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())

        # run
        run_experiment(opt, "TCGA-PANCAN-HiSeq", MultiClassifier{Float64}, 
            tol=1e-4, ninner=10^5, nouter=200, mult=1.05, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 8: optdigits #####
if "optdigits" in ARGS
    println("Running 'optdigits' benchmark")
    for opt in (SD, MM)
        # precompile
        run_experiment(opt, "optdigits", MultiClassifier{Float64},
            tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:none,
            intercept=true, kernel=nothing, strategy=OVO(), percent_train=0.68)

        # run
        run_experiment(opt, "optdigits", MultiClassifier{Float64}, 
            tol=1e-4, ninner=10^5, nouter=50, mult=1.2, scale=:none,
            intercept=true, kernel=nothing, strategy=OVO(), percent_train=0.68)
    end
end
