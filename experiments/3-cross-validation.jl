using SparseSVM, MLDataUtils, KernelFunctions, LinearAlgebra
using CSV, DataFrames, Random, Statistics
using ProgressMeter

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
    kernel::Union{Nothing,Kernel}=nothing,
    intercept=false,
    )
    # Load the data
    labeled_data = load_data(dataset, seed=seed)
    
    # Create the train and test data sets.
    cv_set, (test_X, test_targets) = splitobs(labeled_data, at=percent_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)
    gridvals = sort(grid, rev=true) # iterate from least sparse to most sparse models

    # allocate output
    kvals = zeros(Int, length(grid), nfolds)
    train_score = zeros(length(grid), nfolds)
    val_score = zeros(length(grid), nfolds)
    test_score = zeros(length(grid), nfolds)

    # run cross-validation
    p = Progress(nfolds, 1, "Running CV...")
    for (j, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # get training set and validation set
        ((train_X, train_targets), (val_X, val_targets)) = fold

        # create classifier and use same initial point
        classifier = make_classifier(ctype, train_X, train_targets, first(train_targets), kernel=kernel, intercept=intercept)
        initialize_weights!(MersenneTwister(1903), classifier)

        # create SVD if needed
        A = SparseSVM.get_design_matrix(classifier.data, classifier.intercept)
        Asvd = svd(A, full=true)

        # follow path along sparsity sets
        for (i, s) in enumerate(gridvals)
            # compute sparsity parameter k
            nparams = kernel isa Nothing ? size(train_X, 2) : size(train_X, 1)
            k = round(Int, nparams*s)

            # train classifier enforcing k nonzero parameters
            trainMM!(classifier, A, f, tol, k, fullsvd=Asvd, nouter=nouter, ninner=ninner, mult=mult, init=false)

            # compute evaluation metrics
            train_score[i,j] = accuracy_score(classifier, train_X, train_targets)
            val_score[i,j] = accuracy_score(classifier, val_X, val_targets)
            test_score[i,j] = accuracy_score(classifier, test_X, test_targets)
            kvals[i,j] = k
        end

        next!(p, showvalues=[(:fold, j)])
    end

    scores = (grid=gridvals, k=kvals, train=train_score, val=val_score, test=test_score)

    return scores
end

    # Initialize classifier.
    # classifier = make_classifier(ctype, Xtrain, targets_train, first(targets_train), kernel=kernel, intercept=intercept)
    # local_rng = MersenneTwister(1903)
    # initialize_weights!(local_rng, classifier)

    # # Create closure.
    # execute(ninner, nouter) = @timed trainMM(classifier, f, tol, k, nouter=nouter, ninner=ninner, mult=mult, init=false)
    
    # Run the experiment several times.
    # original_stdout = stdout
    # results = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm)).out", "w")
    # write(results, "trial\tscore\tpercent\ttime\n")
    # @showprogress 1 "Testing $(algorithm)..." for trial in 1:ntrials
    #     # Feed STDOUT output to a log file
    #     # io = open("experiments/$(dataset)/experiment1-k=$(k)-$(string(algorithm))-$(trial).log", "w")
    #     # redirect_stdout(io)

    #     # Train using the chosen MM algorithm
    #     initialize_weights!(local_rng, classifier)
    #     r = execute(ninner, nouter)
        
    #     # Test the classifier against the test data
    #     targets_predict = classifier.(eachrow(Xtest))
    #     score = sum(targets_predict .== targets_test)
    #     percent = round(score / length(targets_test) * 100, digits=2)

    #     # Write to results file.
    #     # write(results, "$(trial)\t$(score)\t$(percent)\t$(r.time)\n")

    #     # Close log file.
    #     # close(io)
    # end

    # Close file.
    # redirect_stdout(original_stdout)
    # close(results)
# end
