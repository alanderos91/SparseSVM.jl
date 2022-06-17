"""
    cv(algorithm, problem, grid; [at], [kwargs...])

Split data in `problem` into train-validation and test sets, then run cross-validation over the `grid`. Note the following:

- The train-validation set is shuffled and further split into train and validation sets for each fold.
- The training set is used to fit a SVM which is assessed over a validation set.
- Scores due to the validation set should be used to select candidate solutions.
- Out-of-sample performance is determined over the testing set.
- Crucially, samples from the test set are **never** used to train or validate a SVM.

# Keyword Arguments

- `at`: A value between `0` and `1` indicating the proportion of samples/instances used for cross-validation, with remaining samples used for a test set (default=`0.8`).
"""
function cv(algorithm::AbstractMMAlg, problem, grid::G; at::Real=0.8, kwargs...) where G
    # Split data into cross-validation and test sets.
    @unpack p, labels, intercept = problem
    X = get_problem_data(problem)
    dataset_split = splitobs((labels, view(X, :, 1:p)), at=at, obsdim=1)
    SparseSVM.cv(algorithm, problem, grid, dataset_split; kwargs...)
end

"""
    cv(algorithm, problem, grid, dataset_split; [kwargs...])

Run k-fold cross-validation over various sparsity levels.

The given `problem` should enter with initial model parameters in `problem.coeff`.
Hyperparameter values are specified in `grid`, and data subsets are given as `dataset_split = (cv_set, test_set)`.

# Keyword Arguments

- `nfolds`: The number of folds to run in cross-validation.
- `scoref`: A function that evaluates a classifier over training, validation, and testing sets (default uses misclassification error).
- `show_progress`: Toggles progress bar.

Additional arguments are propagated to `fit` and `anneal`. See also [`SparseSVM.fit`](@ref) and [`SparseSVM.anneal`](@ref).
"""
function cv(algorithm::AbstractMMAlg, problem, grid::G, dataset_split::Tuple{S1,S2};
    lambda::Real=1.0,
    maxiter::Int=10^4,
    tol::Real=DEFAULT_GTOL,
    nfolds::Int=5,
    scoref::Function=DEFAULT_SCORE_FUNCTION,
    cb::Function=DEFAULT_CALLBACK,
    show_progress::Bool=true,
    kwargs...) where {G,S1,S2}
    # Initialize the output.
    cv_set, test_set = dataset_split
    ns = length(grid)
    alloc_score_arrays(a, b) = [Vector{Float64}(undef, a) for _ in 1:b]
    result = (;
        train=alloc_score_arrays(ns, nfolds),
        validation=alloc_score_arrays(ns, nfolds),
        test=alloc_score_arrays(ns, nfolds),
        time=alloc_score_arrays(ns, nfolds),
    )

    # Run cross-validation.
    if show_progress
        progress_bar = Progress(nfolds * ns, 1, "Running CV w/ $(algorithm)... "; offset=3)
    end

    for (k, fold) in enumerate(kfolds(cv_set, k=nfolds, obsdim=1))
        # Retrieve the training set and validation set.
        # TODO: Does this guarantee copies?
        train_set, validation_set = fold
        train_Y, train_X = getobs(train_set, obsdim=1)
        val_Y, val_X = getobs(validation_set, obsdim=1)
        test_Y, test_X = getobs(test_set, obsdim=1)
        
        # Standardize ALL data based on the training set.
        F = StatsBase.fit(ZScoreTransform, train_X, dims=1)
        has_nan = any(isnan, F.scale) || any(isnan, F.mean)
        has_inf = any(isinf, F.scale) || any(isinf, F.mean)
        has_zero = any(iszero, F.scale)
        if has_nan
            error("Detected NaN in z-score.")
        elseif has_inf
            error("Detected Inf in z-score.")
        elseif has_zero
            for idx in eachindex(F.scale)
                x = F.scale[idx]
                F.scale[idx] = ifelse(iszero(x), one(x), x)
            end
        end

        foreach(X -> StatsBase.transform!(F, X), (train_X, val_X, test_X))
        
        # Create a problem object for the training set.
        train_idx = extract_indices(problem, train_Y)
        train_problem = change_data(problem, train_Y, train_X)
        extras = __mm_init__(algorithm, train_problem, nothing)
        set_initial_coefficients!(train_problem, problem, train_idx)

        for (i, s) in enumerate(grid)
            # Obtain solution as function of s.
            if s != 0.0
                result.time[k][i] = @elapsed SparseSVM.fit!(algorithm, train_problem, s, extras, (true, false,);
                    cb=cb, kwargs...
                )
            else# s == 0
                result.time[k][i] = @elapsed SparseSVM.init!(algorithm, train_problem, lambda, extras;
                    maxiter=maxiter, gtol=tol, nesterov_threshold=0,
                )
            end
            copyto!(train_problem.coeff, train_problem.proj)
            # copy_to_buffer!(train_problem)

            # Evaluate the solution.
            r = scoref(train_problem, (train_Y, train_X), (val_Y, val_X), (test_Y, test_X))
            for (arr, val) in zip(result, r) # only touches first three arrays
                arr[k][i] = val
            end

            # Update the progress bar.
            if show_progress
                spercent = string(round(100*s, digits=6), '%')
                next!(progress_bar, showvalues=[(:fold, k), (:sparsity, spercent)])
            end
        end
    end

    return result
end

function repeated_cv(algorithm::AbstractMMAlg, problem, grid::G; at::Real=0.8, kwargs...) where G
    # Split data into cross-validation and test sets.
    @unpack p, labels, intercept = problem
    X = get_problem_data(problem)
    dataset_split = splitobs((labels, view(X, :, 1:p)), at=at, obsdim=1)
    SparseSVM.repeated_cv(algorithm, problem, grid, dataset_split; kwargs...)
end

function repeated_cv(algorithm::AbstractMMAlg, problem, grid::G, dataset_split::Tuple{S1,S2};
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
        replicate[rep] = SparseSVM.cv(algorithm, problem, grid, (cv_shuffled, test_set);
            show_progress=show_progress, kwargs...)

        # Update the progress bar.
        if show_progress
            next!(progress_bar, showvalues=[(:replicate, rep),])
        end
    end

    return replicate
end
