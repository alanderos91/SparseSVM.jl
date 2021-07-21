# load common packages + functions
include("common.jl")
using LIBSVM
import SparseSVM: Classifier
import LIBSVM: set_params!, fit!

struct LIBSVMClassifier{M} <: SparseSVM.Classifier
    model::M
end

function (classifier::LIBSVMClassifier)(X::AbstractMatrix)
    LIBSVM.predict(classifier.model, X)
end

LIBSVM.set_params!(classifier::LIBSVMClassifier; new_params...) = LIBSVM.set_params!(classifier.model; new_params...)
LIBSVM.fit!(classifier, X, y) = LIBSVM.fit!(classifier.model, X, y)

function LIBSVM_L2(;C::Real=1.0, tol::Real=1e-4, intercept::Bool=false, kwargs...)
    model = LIBSVMClassifier(LinearSVC(
        solver=LIBSVM.Linearsolver.L2R_L2LOSS_SVC,
        cost=C,
        tolerance=tol,
        bias= intercept ? 1.0 : -1.0,
    ))
    return model
end

function LIBSVM_L1(;C::Real=1.0, tol::Real=1e-4, intercept::Bool=false, kwargs...)
    model = LIBSVMClassifier(LinearSVC(
        solver=LIBSVM.Linearsolver.L1R_L2LOSS_SVC,
        cost=C,
        tolerance=tol,
        bias= intercept ? 1.0 : -1.0,
    ))
    return model
end

function LIBSVM_RB(;C::Real=1.0, tol::Real=1e-4, intercept::Bool=false, kwargs...)
    model = LIBSVMClassifier(SVC(
        kernel=LIBSVM.Kernel.RadialBasis,
        cost=C,
        tolerance=tol,
    ))
    return model
end

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

            # Append results to file.
            writedlm(results, Any[
                algname j val t train_acc val_acc test_acc
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
    writedlm(results, ["fold" "value" "time" "train_acc" "val_acc" "test_acc"])

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

function cleanup_precompile(fname)
    rm(joinpath("results", dataset, "$(fname).out"))
    rm(joinpath("results", dataset, "$(fname).log"))
end

##### Example 1: synthetic #####
if "synthetic" in ARGS
    our_grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ]
    dataset = "synthetic"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, BinaryClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, BinaryClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^4, nouter=100, mult=1.2,
        intercept=true, kernel=nothing)
end


##### Example 2: iris #####
if "iris" in ARGS
    our_grid = [
        0.0, 0.25, 0.5, 0.75 
    ]
    their_grid = [
        1.0, 1e-1, 1e-2, 1e-3
    ]
    dataset = "iris"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^4, nouter=100, mult=1.2,
        intercept=true, kernel=nothing)
end

##### Example 3: spiral #####
if "spiral" in ARGS
    our_grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ]
    dataset = "spiral"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=RBFKernel())
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^4, nouter=100, mult=1.2,
        intercept=true, kernel=RBFKernel())
end

##### Example 4: letter-recognition #####
if "letter-recognition" in ARGS
    our_grid = [
        i/16 for i in 0:15
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
        1e-4
    ]
    dataset = "letter-recognition"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:minmax,
        intercept=true, kernel=nothing)
end

##### Example 5: breast-cancer-wisconsin #####
if "breast-cancer-wisconsin" in ARGS
    our_grid = [
        i/10 for i in 0:9
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
    ]
    dataset = "breast-cancer-wisconsin"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, BinaryClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, BinaryClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:none,
        intercept=true, kernel=nothing)
end

##### Example 6: splice #####
if "splice" in ARGS
    our_grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ]
    dataset = "splice"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:minmax,
        intercept=true, kernel=nothing)
end

##### Example 7: TCGA-PANCAN-HiSeq #####
if "TCGA-PANCAN-HiSeq" in ARGS
    our_grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ]
    dataset = "TCGA-PANCAN-HiSeq"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^5, nouter=200, mult=1.05, scale=:minmax,
        intercept=true, kernel=nothing)
end

##### Example 8: optdigits #####
if "optdigits" in ARGS
    our_grid = [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ]
    their_grid = [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ]
    dataset = "optdigits"

    println("Running '$(dataset)' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    cleanup_precompile(fname)

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, dataset, our_grid, their_grid, MultiClassifier{Float64},
        nfolds=10, tol=1e-4, ninner=10^5, nouter=100, mult=1.2, scale=:none,
        intercept=true, kernel=nothing, proportion_train=0.68)
end
