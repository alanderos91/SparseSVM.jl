# load common packages + functions
include("common.jl")

function run_experiment(fname, algorithm::AlgOption, dataset, grid, ctype=MultiClassifier;
        nfolds::Int=10,
        proportion_train::Real=0.8,
        tol::Real=1e-6,
        nouter::Int=20,
        ninner::Int=1000,
        mult::Real=1.5,
        scale::Symbol=:zscore,
        strategy::MultiClassStrategy=OVR(),
        kernel::Union{Nothing,Kernel}=nothing,
        intercept::Bool=false,
    )
    # Put options into a list.
    options = (
        :nfolds => nfolds,
        :fname => fname,
        :dataset => dataset,
        :ctype => ctype,
        :proportion_train => proportion_train,
        :tol => tol,
        :nouter => nouter,
        :ninner => ninner,
        :mult => mult,
        :scale => scale,
        :kernel => kernel,
        :intercept => intercept,
        :strategy => strategy,
        :algorithm => string(algorithm),
    )

    # Load the data.
    df = SparseSVM.dataset(dataset)
    y, X = Vector(df[!, 1]), Matrix{Float64}(df[!, 2:end])

    if dataset == "TCGA-PANCAN-HiSeq"
        # The TCGA data has a few columns full of 0s that should be dropped.
        idxs = findall(i -> isequal(0, maximum(X[:,i])), 1:size(X,2))
        selected = setdiff(1:size(X,2), idxs)

        # Limit to the first 10000 genes.
        X = X[:, selected[1:10000]]
    end

    _rescale_!(Val(scale), X)
    labeled_data = (X, y)
    
    # Create the train and test data sets.
    cv_set, (test_X, test_targets) = splitobs(labeled_data, at=proportion_train, obsdim=1)

    # Process options.
    f = get_algorithm_func(algorithm)
    alg = string(algorithm)
    gridvals = sort!(unique(grid), rev=false) # iterate from least sparse to most sparse models
    nvals = length(gridvals)

    # Open output file.
    dir = joinpath("results", dataset)
    !isdir(dir) && mkdir(dir)
    open(joinpath(dir, "$(fname).log"), "a+") do io
        for (key, val) in options
            writedlm(io, [key val], '=')
        end
        println(io)
    end
    results = open(joinpath(dir, "$(fname).out"), "a+")
    writedlm(results, [
        "alg" "fold" "sparsity" "time" "sv" "iter" "obj" "dist" "train_acc" "val_acc" "test_acc"]
    )

    # Run cross-validation.
    p = Progress(nfolds*nvals, 1, "Running CV w/ $(alg)... ")
    for (j, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # Get training set and validation set.
        ((train_X, train_targets), (val_X, val_targets)) = fold

        # Create classifier and use same initial point.
        classifier = make_classifier(ctype, train_X, train_targets, first(train_targets), kernel=kernel, intercept=intercept, strategy=strategy)

        # Create design matrix and SVD (if needed).
        _r = @timed begin
            A, Asvd = SparseSVM.get_A_and_SVD(classifier)

            # Initialize weights with univariate solution.
            initialize_weights!(classifier, A)
        end

        # Follow path along sparsity sets.
        for (i, s) in enumerate(gridvals)
            # Train classifier enforcing an s-sparse solution.
            r = @timed trainMM!(classifier, A, f, tol, s,
                fullsvd=Asvd, nouter=nouter, ninner=ninner, mult=mult, init=false, verbose=false)
            iters, obj, dist = r.value
            t = i > 1 ? r.time : r.time + _r.time # account for init cost

            # Get number of support vectors.
            sv = count_support_vecs(classifier)

            # Compute evaluation metrics.
            train_acc = round(accuracy_score(classifier, train_X, train_targets)*100, sigdigits=4)
            val_acc = round(accuracy_score(classifier, val_X, val_targets)*100, sigdigits=4)
            test_acc = round(accuracy_score(classifier, test_X, test_targets)*100, sigdigits=4)
            sparsity = round(s*100, sigdigits=4)

            # Append results to file.
            writedlm(results, Any[
                alg j sparsity t sv iters obj dist train_acc val_acc test_acc
            ])
            flush(results)

            # Update progress bar.
            next!(p, showvalues=[(:fold, j), (:sparsity, sparsity)])
        end
    end

    close(results)

    return nothing
end

##### Example 1: synthetic #####
if "synthetic" in ARGS
    grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    dataset = "synthetic"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")
    
    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing)
    end
    cleanup_precompile(fname)
    
    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=10^4, nouter=100, mult=1.2,
            intercept=true, kernel=nothing)
    end
end

##### Example 2: iris #####
if "iris" in ARGS
    grid = [0.0, 0.25, 0.5, 0.75]
    dataset = "iris"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")
    
    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=10^4, nouter=100, mult=1.2,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 3: spiral #####
if "spiral" in ARGS
    grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    dataset = "spiral"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")

    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2,
            intercept=true, kernel=RBFKernel(), strategy=OVO())
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=10^4, nouter=100, mult=1.2,
            intercept=true, kernel=RBFKernel(), strategy=OVO())
    end
end

##### Example 4: letter-recognition #####
if "letter-recognition" in ARGS
    grid = [i/16 for i in 0:15]
    dataset = "letter-recognition"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")

    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 5: breast-cancer-wisconsin #####
if "breast-cancer-wisconsin" in ARGS
    grid = [i/10 for i in 0:9]
    dataset = "breast-cancer-wisconsin"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")

    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, BinaryClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:none,
            intercept=true, kernel=nothing)
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, BinaryClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:none,
            intercept=true, kernel=nothing)
    end
end

##### Example 6: splice #####
if "splice" in ARGS
    grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    dataset = "splice"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")

    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.1, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 7: TCGA-PANCAN-HiSeq #####
if "TCGA-PANCAN-HiSeq" in ARGS
    grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    dataset = "TCGA-PANCAN-HiSeq"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")

    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.05, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=200, mult=1.05, scale=:minmax,
            intercept=true, kernel=nothing, strategy=OVO())
    end
end

##### Example 8: optdigits #####
if "optdigits" in ARGS
    grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    dataset = "optdigits"
    fname = generate_filename(3, "all")
    println("Running '$(dataset)' benchmark")

    for opt in (SD, MM)
        # precompile
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64},
            nfolds=10, tol=1e-4, ninner=2, nouter=2, mult=1.2, scale=:none,
            intercept=true, kernel=nothing, strategy=OVO(), proportion_train=0.68)
    end
    cleanup_precompile(fname)

    for opt in (SD, MM)
        # run
        run_experiment(fname, opt, dataset, grid, MultiClassifier{Float64}, 
            nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:none,
            intercept=true, kernel=nothing, strategy=OVO(), proportion_train=0.68)
    end
end
