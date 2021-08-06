# load common packages + functions
include("common.jl")
include("LIBSVM_wrappers.jl")

function init_ours(F, ctype, train_X, train_targets, tol, kernel, intercept, ninner, nouter, mult)
    # Create classifier and use same initial point
    classifier = make_classifier(ctype, train_X, train_targets, first(train_targets), kernel=kernel, intercept=intercept, strategy=SparseSVM.OVO())
    
    # Create design matrix and SVD (if needed)
    A, Asvd = SparseSVM.get_A_and_SVD(classifier)

    # initialize weights with univariate solution
    initialize_weights!(classifier, A)

    # Create closure to run our algorithm.
    function run_ours(val)
        trainMM!(classifier, A, F, tol, val,
            fullsvd=Asvd,
            nouter=nouter,
            ninner=ninner,
            mult=mult,
            init=false,
            verbose=false)
        return classifier
    end

    return run_ours
end

function init_theirs(F, ctype, train_X, train_targets, tol, kernel, intercept, ninner, nouter, mult)
    # Create classifier object. F should be one of our wrappers.
    classifier = F(tol=tol, intercept=intercept)

    # Create closure to run LIBSVM algorithm.
    function run_theirs(val)
        LIBSVM.set_params!(classifier, cost=val)
        fit!(classifier, train_X, train_targets)
        return classifier
    end

    return run_theirs
end

function cv(results, algname, init_f, grid, cv_set, test_set, nfolds; message = "Running CV... ")
    # Extract test set pieces.
    (test_X, test_targets) = test_set

    # Initialize progress bar and run CV.
    nvals = length(grid)
    p = Progress(nfolds*nvals, 1, message)
    for (j, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # get training set and validation set
        ((train_X, train_targets), (val_X, val_targets)) = fold

        _r = @timed init_f(train_X, train_targets)
        run_f = _r.value

        for (i, val) in enumerate(grid)
            # Run the training algorithm with hyperparameter = val.
            r = @timed run_f(val)
            classifier = r.value
            t = i > 1 ? r.time : r.time + _r.time # account for init cost

            # Compute evaluation metrics.
            train_acc = round(accuracy_score(classifier, train_X, train_targets)*100, sigdigits=4)
            val_acc = round(accuracy_score(classifier, val_X, val_targets)*100, sigdigits=4)
            test_acc = round(accuracy_score(classifier, test_X, test_targets)*100, sigdigits=4)
            sparsity = measure_model_sparsity(classifier)

            # Append results to file.
            writedlm(results, Any[
                algname j val t train_acc val_acc test_acc sparsity
            ])
            flush(results)

            # Update progress bar
            next!(p, showvalues=[(:fold, j), (:val, val)])
        end
    end
end

function run_experiment(fname, dataset, our_grid, their_grid, ctype=MultiClassifier;
    nfolds::Int=10,
    proportion_train::Real=0.8,
    tol::Real=1e-6,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    scale::Symbol=:zscore,
    kernel::Union{Nothing,KernelFunctions.Kernel}=nothing,
    intercept::Bool=false,
    kwargs...
)
    # Put options into a list
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
    cv_set, test_set = splitobs(labeled_data, at=proportion_train, obsdim=1)

    # Open file for results. Save settings on a separate log.
    dir = joinpath("results", dataset)
    !isdir(dir) && mkdir(dir)
    open(joinpath(dir, "$(fname).log"), "w") do io
        for (key, val) in options
            writedlm(io, [key val], '=')
        end
    end
    results = open(joinpath(dir, "$(fname).out"), "w")
    writedlm(results, ["alg" "fold" "value" "time" "train_acc" "val_acc" "test_acc" "sparsity"])

    # Benchmark MM.
    fMM(X, y) = init_ours(sparse_direct!, ctype, X, y, tol, kernel, intercept, ninner, nouter, mult)
    cv(results, "MM", fMM, our_grid, cv_set, test_set, nfolds,
        message="Running MM... ")

    # Benchmark SD.
    fSD(X, y) = init_ours(sparse_steepest!, ctype, X, y, tol, kernel, intercept, ninner, nouter, mult)
    cv(results, "SD", fSD, our_grid, cv_set, test_set, nfolds,
        message="Running SD... ")

    if kernel isa Nothing
        # Benchmark L2-regularized, L2-loss SVC.
        fL2R(X, y) = init_theirs(LIBSVM_L2, nothing, X, y, tol, nothing, intercept, nothing, nothing, nothing)
        cv(results, "L2R", fL2R, their_grid, cv_set, test_set, nfolds,
            message="Running L2R_L2LOSS_SVC... ")

        # Benchmark L1-regularized, L2-loss SVC.
        fL1R(X, y) = init_theirs(LIBSVM_L1, nothing, X, y, tol, nothing, intercept, nothing, nothing, nothing)
        cv(results, "L1R", fL1R, their_grid, cv_set, test_set, nfolds,
            message="Running L1R_L2LOSS_SVC... ")
    else
        # Benchmark non-linear SVC with Radial Basis Kernel.
        fSVC(X, y) = init_theirs(LIBSVM_RB, nothing, X, y, tol, nothing, intercept, nothing, nothing, nothing)
        cv(results, "SVC", fSVC, their_grid, cv_set, test_set, nfolds,
            message="Running SVC w/ RBF kernel... ")
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
    
    # options
    ctype, kwargs = OPTIONS[example]
    our_grid = SPARSITY_GRID[example]
    their_grid = MARGIN_GRID[example] 

    # precompile
    tmpkwargs = (kwargs..., ninner=2, nouter=2,)
    fname = generate_filename(4, "all")
    for opt in (SD, MM)
        run_experiment(fname, example, our_grid, their_grid, ctype;
            nfolds=10, tmpkwargs...)
    end
    cleanup_precompile(example, fname)

    # run
    fname = generate_filename(4, "all")
    for opt in (SD, MM)
        run_experiment(fname, example, our_grid, their_grid, ctype;
            nfolds=10, kwargs...)
    end
end

# function run_L2R()
#     C_vals = [2.0 ^ k for k in 0:-1:-10]
#     F = fL2R(X, y)
#     for C in C_vals
#         println("Running w/ C = $(C)...\n"); flush(stdout)
#         F(C)
#         println("\nFinished running w/ C = $(C).\n"); flush(stdout)
#     end
# end

# open("/home/alanderos/Desktop/tmp.out", "w") do io
#     redirect_stdout(run_L2R, io)
# end

# svm = F(1.0)
# p, d = LIBSVM.LIBLINEAR.linear_predict(svm.model.fit, X')