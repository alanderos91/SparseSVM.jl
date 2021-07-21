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

        run_f = init_f(train_X, train_targets)

        for (i, val) in enumerate(grid)
            # Run the training algorithm with hyperparameter = val.
            r = @timed run_f(val)
            classifier = r.value
            t = r.time

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

    println("Running 'synthetic' benchmark")

    # precompile
    fname = generate_filename(4, "all")
    run_experiment(fname, "synthetic", our_grid, their_grid, BinaryClassifier{Float64},
        nfolds=3, tol=1e-1, ninner=2, nouter=2, mult=1.2,
        intercept=true, kernel=nothing)
    rm(joinpath("results", "synthetic", "$(fname).out"))
    rm(joinpath("results", "synthetic", "$(fname).log"))

    # run
    fname = generate_filename(4, "all")
    run_experiment(fname, "synthetic", our_grid, their_grid, BinaryClassifier{Float64},
        nfolds=10, tol=1e-6, ninner=10^4, nouter=100, mult=1.2,
        intercept=true, kernel=nothing)
end
