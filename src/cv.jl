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
    tol::Real=DEFAULT_GTOL,
    nfolds::Int=5,
    scoref::Function=DEFAULT_SCORE_FUNCTION,
    cb::Function=DEFAULT_CALLBACK,
    show_progress::Bool=true,
    data_transform::Type{Transform}=ZScoreTransform,
    kwargs...) where {G,S1,S2,Transform}
    # Sanity checks.
    if length(grids) != 2
        error("Argument 'grids' should contain two collections representing sparsity and lambda values, respectively.")
    end
    sparsity_grid, lambda_grid = grids
    if any(x -> x < 0 || x > 1, sparsity_grid)
        error("Values in sparsity grid should be in [0,1].")
    end
    if any(x -> x < 0, lambda_grid)
        error("Values in λ grid should be positive.")
    end

    # Initialize the output.
    cv_set, test_set = dataset_split
    nl, ns = length(lambda_grid), length(sparsity_grid)
    alloc_score_arrays(a, b, c) = [Matrix{Float64}(undef, a, b) for _ in 1:c]
    result = (;
        train=alloc_score_arrays(ns, nl, nfolds),
        validation=alloc_score_arrays(ns, nl, nfolds),
        test=alloc_score_arrays(ns, nl, nfolds),
        time=alloc_score_arrays(ns, nl, nfolds),
    )

    # Run cross-validation.
    if show_progress
        progress_bar = Progress(nfolds * ns * nl, 1, "Running CV w/ $(algorithm)... "; offset=3)
    end

    for (k, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        train_set, validation_set = fold
        train_Y, train_X = getobs(train_set, obsdim=1)
        val_Y, val_X = getobs(validation_set, obsdim=1)
        test_Y, test_X = getobs(test_set, obsdim=1)

        # Standardize ALL data based on the training set.
        # Adjustment of transformation is to detect NaNs, Infs, and zeros in transform parameters that will corrupt data, and handle them gracefully if possible.
        F = StatsBase.fit(Transform, train_X, dims=1)
        __adjust_transform__(F)
        foreach(X -> StatsBase.transform!(F, X), (train_X, val_X, test_X))
        
        # Create a problem object for the training set.
        train_problem = change_data(problem, train_Y, train_X)
        extras = __mm_init__(algorithm, train_problem, nothing)
        T = floattype(train_problem)

        for (j, lambda) in enumerate(lambda_grid)
            set_initial_coefficients_and_intercept!(train_problem, zero(T))

            for (i, s) in enumerate(sparsity_grid)
                # Obtain solution as function of s.
                if s != 0.0
                    result.time[k][i,j] = @elapsed SparseSVM.fit!(algorithm, train_problem, lambda, s, extras, (true, false,);
                        cb=cb, kwargs...
                    )
                else# s == 0
                    result.time[k][i,j] = @elapsed SparseSVM.init!(algorithm, train_problem, lambda, extras;
                        maxiter=maxiter, gtol=tol, nesterov_threshold=0,
                    )
                end
    
                # Evaluate the solution.
                r = scoref(train_problem, (train_Y, train_X), (val_Y, val_X), (test_Y, test_X))
                for (arr, val) in zip(result, r) # only touches first three arrays
                    arr[k][i,j] = val
                end
    
                # Update the progress bar.
                if show_progress
                    spercent = string(round(100*s, digits=6), '%')
                    next!(progress_bar, showvalues=[(:fold, k), (:lambda, lambda), (:sparsity, spercent)])
                end
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
    nreplicates::Int=10,
    show_progress::Bool=true,
    rng::AbstractRNG=StableRNG(1903),
    kwargs...) where {G,S1,S2}
    # Retrieve subsets and create index set into cross-validation set.
    cv_set, test_set = dataset_split

    if show_progress
        progress_bar = Progress(nreplicates, 1, "Repeated CV... ")
    end

    # Replicate CV procedure several times.
    keys = (:train,:validation,:test,:time)
    types = NTuple{4,Vector{Vector{Float64}}}
    replicate = Vector{NamedTuple{keys,types}}(undef, nreplicates)

    for rep in 1:nreplicates
        # Shuffle cross-validation data.
        cv_shuffled = shuffleobs(cv_set, obsdim=1, rng=rng)

        # Run k-fold cross-validation and store results.
        replicate[rep] = SparseSVM.cv(algorithm, problem, grids, (cv_shuffled, test_set);
            show_progress=show_progress, kwargs...)

        # Update the progress bar.
        if show_progress
            next!(progress_bar, showvalues=[(:replicate, rep),])
        end
    end

    return replicate
end
