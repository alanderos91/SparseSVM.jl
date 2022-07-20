"""
    cv(algorithm, problem, grids; [at], [kwargs...])

Split data in `problem` into train-validation and test sets, then run cross-validation over the `grids`. Note the following:

- The train-validation set is shuffled and further split into train and validation sets for each fold.
- The training set is used to fit a SVM which is assessed over a validation set.
- Scores due to the validation set should be used to select candidate solutions.
- Out-of-sample performance is determined over the testing set.
- Crucially, samples from the test set are **never** used to train or validate a SVM.

# Keyword Arguments

- `at`: A value between `0` and `1` indicating the proportion of samples/instances used for cross-validation, with remaining samples used for a test set (default=`0.8`).
"""
function cv(algorithm::AbstractMMAlg, problem, grids::G; at::Real=0.8, kwargs...) where G
    # Split data into cross-validation and test sets.
    @unpack p, labels, intercept = problem
    X = get_problem_data(problem)
    dataset_split = splitobs((labels, view(X, :, 1:p)), at=at, obsdim=1)
    SparseSVM.cv(algorithm, problem, grids, dataset_split; kwargs...)
end

"""
    cv(algorithm, problem, grids, dataset_split; [kwargs...])

Run k-fold cross-validation over various sparsity levels.

The given `problem` should enter with initial model parameters in `problem.coeff`.
Hyperparameter values are specified in `grids`, and data subsets are given as `dataset_split = (cv_set, test_set)`.

# Keyword Arguments

- `nfolds`: The number of folds to run in cross-validation.
- `scoref`: A function that evaluates a classifier over training, validation, and testing sets (default uses misclassification error).
- `show_progress`: Toggles progress bar.

Additional arguments are propagated to `fit` and `anneal`. See also [`SparseSVM.fit`](@ref) and [`SparseSVM.anneal`](@ref).
"""
function cv(algorithm::AbstractMMAlg, problem, grids::G, dataset_split::Tuple{S1,S2};
    maxiter::Int=10^4,
    gtol::Real=DEFAULT_GTOL,
    nfolds::Int=5,
    scoref::Function=DEFAULT_SCORE_FUNCTION,
    cb::Function=DEFAULT_CALLBACK,
    show_progress::Bool=true,
    progress_bar::Progress=Progress(nfolds * length(grids[1]) * length(grids[2]); desc="Running CV w/ $(algorithm)... ", enabled=show_progress),
    data_transform::Type{Transform}=ZScoreTransform,
    kwargs...) where {G,S1,S2,Transform}
    # Sanity checks.
    not_proportion(x) = x < 0 || x > 1
    if length(grids) != 2
        error("Argument 'grids' should contain two collections representing sparsity and lambda values, respectively.")
    end
    sparsity_grid, lambda_grid = grids
    if any(not_proportion, sparsity_grid)
        error("Values in sparsity grid should be in [0,1].")
    end
    if any(<(0), lambda_grid)
        error("Values in λ grid should be positive.")
    end

    # Initialize the output.
    cv_set, test_set = dataset_split
    nl, ns = length(lambda_grid), length(sparsity_grid)
    alloc_score_arrays(a, b, c) = Array{Float64,3}(undef, a, b, c)
    result = (;
        train=alloc_score_arrays(ns, nl, nfolds),
        validation=alloc_score_arrays(ns, nl, nfolds),
        test=alloc_score_arrays(ns, nl, nfolds),
        time=alloc_score_arrays(ns, nl, nfolds),
    )

    # Run cross-validation.
    for (k, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_Y, train_X = getobs(train_set, obsdim=1)
        val_Y, val_X = getobs(validation_set, obsdim=1)
        test_Y, test_X = getobs(test_set, obsdim=1)

        # Standardize ALL data based on the training set.
        # Adjustment of transformation is to detect NaNs, Infs, and zeros in transform parameters that will corrupt data, and handle them gracefully if possible.
        F = StatsBase.fit(data_transform, train_X, dims=1)
        __adjust_transform__(F)
        foreach(Base.Fix1(StatsBase.transform!, F), (train_X, val_X, test_X))
        
        # Create a problem object for the training set.
        train_problem = change_data(problem, train_Y, train_X)
        extras = __mm_init__(algorithm, train_problem, nothing)
        T = floattype(train_problem)

        for (j, lambda) in enumerate(lambda_grid)
            set_initial_coefficients_and_intercept!(train_problem, zero(T))

            for (i, sparsity) in enumerate(sparsity_grid)
                # Obtain solution as function of sparsity and lambda.
                if sparsity != 0.0
                    timed_result = @timed SparseSVM.fit!(algorithm, train_problem, lambda, sparsity, extras, (true, false,);
                        gtol=gtol, kwargs...
                    )
                else# sparsity == 0
                    timed_result = @timed SparseSVM.fit!(algorithm, train_problem, lambda, extras;
                        maxiter=maxiter, gtol=gtol,
                    )
                end

                hyperparams = (;sparsity=sparsity, lambda=lambda,)
                indices = (;sparsity=i, lambda=j, fold=k,)
                measured_time = timed_result.time # seconds
                result.time[i,j,k] = measured_time
                statistics = timed_result.value
                cb(statistics, problem, hyperparams, indices)

                # Evaluate the solution.
                r = scoref(train_problem, (train_Y, train_X), (val_Y, val_X), (test_Y, test_X))
                for (arr, val) in zip(result, r) # only touches first three arrays
                    arr[i,j,k] = val
                end
    
                # Update the progress bar.
                spercent = string(round(100*sparsity, digits=4), '%')
                next!(progress_bar, showvalues=[(:fold, k), (:lambda, lambda), (:sparsity, spercent)])
            end
        end
    end

    return result
end

function repeated_cv(algorithm::AbstractMMAlg, problem, grids::G; at::Real=0.8, kwargs...) where G
    # Split data into cross-validation and test sets.
    @unpack p, labels, intercept = problem
    X = get_problem_data(problem)
    dataset_split = splitobs((labels, view(X, :, 1:p)), at=at, obsdim=1)
    SparseSVM.repeated_cv(algorithm, problem, grids, dataset_split; kwargs...)
end

function repeated_cv(algorithm::AbstractMMAlg, problem, grids::G, dataset_split::Tuple{S1,S2};
    nfolds::Int=5,
    nreplicates::Int=10,
    show_progress::Bool=true,
    rng::AbstractRNG=StableRNG(1903),
    cb::Function=DEFAULT_CALLBACK,
    kwargs...) where {G,S1,S2}
    # Retrieve subsets and create index set into cross-validation set.
    cv_set, test_set = dataset_split

    progress_bar = Progress(nreplicates * nfolds * length(grids[1]) * length(grids[2]); desc="Repeated CV... ", enabled=show_progress)

    # Replicate CV procedure several times.
    keys = (:train,:validation,:test,:time)
    types = NTuple{4,Array{Float64,3}}
    replicate = Vector{NamedTuple{keys,types}}(undef, nreplicates)

    for rep in 1:nreplicates
        # Shuffle cross-validation data.
        cv_shuffled = shuffleobs(cv_set, obsdim=1, rng=rng)

        # Run k-fold cross-validation and store results.
        cb_rep = cb(rep)
        replicate[rep] = SparseSVM.cv(algorithm, problem, grids, (cv_shuffled, test_set);
            nfolds=nfolds, show_progress=show_progress, progress_bar=progress_bar, cb=cb_rep, kwargs...)
    end

    return replicate
end

function search_hyperparameters(sparsity_grid, lambda_grid, data;
        minimize::Bool=true
    )
    ns, nl = length(sparsity_grid), length(lambda_grid)
    if size(data) != (ns, nl)
        error("Data in NamedTuple is incompatible with ($ns,$nl) grid.")
    end

    if minimize
        best_i, best_j, best_triple = 0, 0, (Inf, Inf, Inf)
    else
        best_i, best_j, best_triple = 0, 0, (-Inf, -Inf, -Inf)
    end

    for (j, lambda) in enumerate(lambda_grid), (i, sparsity) in enumerate(sparsity_grid)
        pair_score = data[i,j]

        # Check if this is the best pair. Rank by pair_score -> sparsity -> lambda.
        if minimize
            #
            #   pair_score: We want to minimize the CV score; e.g. minimum prediction error.
            #   1-sparsity: higher sparsity => smaller model
            #   1/lambda: larger lambda => wider margin
            #
            proposal = (pair_score, 1-sparsity, 1/lambda)
            if proposal < best_triple
                best_i, best_j, best_triple = i, j, proposal
            end
        else
            #
            #   pair_score: We want to maximize the CV score; e.g. maximum prediction accuracy.
            #   sparsity: higher sparsity => smaller model
            #   lambda: larger lambda => wider margin
            #
            proposal = (pair_score, sparsity, lambda)
            if proposal > best_triple
                best_i, best_j, best_triple = i, j, proposal
            end
        end
    end

    return best_i, best_j, best_triple
end
