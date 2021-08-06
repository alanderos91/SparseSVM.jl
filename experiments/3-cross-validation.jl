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
        "alg" "fold" "sparsity" "time" "sv" "iter" "obj" "dist" "gradsq" "train_acc" "val_acc" "test_acc"]
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
            iters, obj, dist, gradsq = r.value
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
                alg j sparsity t sv iters obj dist gradsq train_acc val_acc test_acc
            ])
            flush(results)

            # Update progress bar.
            next!(p, showvalues=[(:fold, j), (:sparsity, sparsity)])
        end
    end

    close(results)

    return nothing
end

##### MAIN #####
include("examples.jl")

# Check for unknown examples.
examples = String[]
for example in ARGS
    if example in EXAMPLES
        push!(examples, example)
    else
        @warn "Unknown example `$(example)`. Check spelling."
    end
end

# Run selected examples.
for example in examples
    println("Running '$(example)' benchmark")
    fname = generate_filename(3, "all")

    # options
    ctype, kwargs = OPTIONS[example]
    grid = SPARSITY_GRID[example]

    # precompile
    tmpkwargs = (kwargs..., ninner=2, nouter=2,)
    for opt in (SD, MM)
        run_experiment(fname, opt, example, grid, ctype; nfolds=10, tmpkwargs...)
    end
    cleanup_precompile(example, fname)

    # run
    for opt in (SD, MM)
        run_experiment(fname, opt, example, grid, ctype; nfolds=10, kwargs...)
    end
end
