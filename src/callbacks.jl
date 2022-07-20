"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__((iter, state), problem, hyperparams) = nothing
__do_nothing_callback__(statistics, problem, hyperparams, indices) = nothing
__do_nothing_callback__(::Int) = __do_nothing_callback__

##### options for scoref argument #####

function prediction_errors(problem, train_set, validation_set, test_set)
    # Extract data for each set.
    Tr_Y, Tr_X = train_set
    V_Y, V_X = validation_set
    T_Y, T_X = test_set

    # Helper function to make predictions on each subset and evaluate errors.
    classification_error = function(problem, L, X)
        # Sanity check: Y and X have the same number of rows.
        length(L) != size(X, 1) && error("Labels ($(length(L))) not compatible with data X ($(size(X))).")

        # Translate data to the model's encoding.
        L_translated = convert_labels(problem, L)

        # Classify response in vertex space; may use @batch.
        Lhat = SparseSVM.classify(problem, X)

        # Sweep through predictions and tally the mistakes.
        nincorrect = sum(L_translated .!= Lhat)

        return nincorrect / length(L)
    end

    Tr = classification_error(problem, Tr_Y, Tr_X)
    V = classification_error(problem, V_Y, V_X)
    T = classification_error(problem, T_Y, T_X)

    return (Tr, V, T)
end

function prediction_accuracies(problem, train_set, validation_set, test_set)
    Tr, V, T = prediction_errors(problem, train_set, validation_set, test_set)
    return (1-Tr, 1-V, 1-T)
end

##### options for cb argument #####

struct VerboseCallback <: Function
    every::Int
end

VerboseCallback() = VerboseCallback(1)

function (F::VerboseCallback)((iter, state), problem::BinarySVMProblem, hyperparams)
    if iter == 0
        @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s\t%-12s\t%-8s\t%-8s\n", "iter", "rho", "risk", "loss", "objective", "margin", "|gradient|", "distance")
    end
    if iter % F.every == 0
        @printf("%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e\t%8.3e\t%4.3e\t%4.3e\n", iter, hyperparams.rho, state.risk, state.loss, state.objective, 1/(state.norm), state.gradient, state.distance)
    end

    return nothing
end

struct CVStatisticsCallback <: Function
    sparsity_grid::Vector
    lambda_grid::Vector
    nfolds::Int

    iters::Array{Float64,3}
    risk::Array{Float64,3}
    loss::Array{Float64,3}
    objective::Array{Float64,3}
    gradient::Array{Float64,3}
    norm::Array{Float64,3}
    distance::Array{Float64,3}

    nnz::Array{Float64,3}
    nsv::Array{Float64,3}

    function CVStatisticsCallback(sparsity_grid, lambda_grid, nfolds)
        ns = length(sparsity_grid)
        nl = length(lambda_grid)
        alloc() = Array{Float64,3}(undef, ns, nl, nfolds)

        return new(
            sparsity_grid, lambda_grid, nfolds, 
            alloc(), alloc(), alloc(), alloc(), alloc(), alloc(), alloc(),
            alloc(), alloc()
        )
    end
end

function (F::CVStatisticsCallback)(statistics::Vector, problem::MultiSVMProblem, hyperparams, indices)
    function get_statistic(replicate, field)
        getfield(last(replicate), field)
    end

    function count_nnz(svm)
        coeff = last(SparseSVM.get_params_proj(svm))
        count(!isequal(0), coeff)
    end

    i, j, k = indices.sparsity, indices.lambda, indices.fold

    F.iters[i,j,k] = mean(first, statistics)
    for field in (:risk, :loss, :objective, :gradient, :norm, :distance)
        arr = getfield(F, field)
        arr[i,j,k] = mean(Base.Fix2(get_statistic, field), statistics)
    end
    F.nnz[i,j,k] = mean(count_nnz, problem.svm)
    F.nsv[i,j,k] = length(support_vectors(problem))

    return nothing
end

function (F::CVStatisticsCallback)(statistics::Tuple, problem::BinarySVMProblem, hyperparams, indices)
    i, j, k = indices.sparsity, indices.lambda, indices.fold

    F.iters[i,j,k] = first(statistics)
    for field in (:risk, :loss, :objective, :gradient, :norm, :distance)
        arr = getfield(F, field)
        arr[i,j,k] = getfield(last(statistics), field)
    end
    F.nnz[i,j,k] = count(!isequal(0), last(SparseSVM.get_params_proj(problem)))
    F.nsv[i,j,k] = length(support_vectors(problem))

    return nothing
end

struct RepeatedCVCallback{T} <: Function
    callback_set::Vector{T}
end

function RepeatedCVCallback{T}(sparsity_grid, lambda_grid, nfolds, nreplicates) where T
    callback_set = Vector{T}(undef, nreplicates)
    for rep in 1:nreplicates
        callback_set[rep] = T(sparsity_grid, lambda_grid, nfolds)
    end
    return RepeatedCVCallback(callback_set)
end

function (F::RepeatedCVCallback)(rep::Int)
    return F.callback_set[rep]
end

function extract_cv_data(F::RepeatedCVCallback{CVStatisticsCallback})
    nreplicates = length(F.callback_set)
    ns, nl, nfolds = size(first(F.callback_set).iters)
    alloc() = Array{Float64,4}(undef, ns, nl, nfolds, nreplicates)

    fields = (:iters, :risk, :loss, :objective, :gradient, :norm, :distance, :nnz, :nsv)
    arrays = (alloc(), alloc(), alloc(), alloc(), alloc(), alloc(), alloc(), alloc(), alloc(),)

    for rep in 1:nreplicates
        replicate = F.callback_set[rep]
        for k in 1:nfolds, j in 1:nl, i in 1:ns
            for (arr, field) in zip(arrays, fields)
                data = getfield(replicate, field)
                arr[i,j,k,rep] = data[i,j,k]
            end
        end
    end

    return NamedTuple{fields}(arrays)
end

CVScoreT = NamedTuple{(:train,:validation,:test,:time),NTuple{4,Array{Float64,3}}}

function extract_cv_data(replicates::Vector{CVScoreT})
    nreplicates = length(replicates)
    ns, nl, nfolds = size(first(replicates).time)
    alloc() = Array{Float64,4}(undef, ns, nl, nfolds, nreplicates)

    fields = (:train, :validation, :test, :time)
    arrays = (alloc(), alloc(), alloc(), alloc(),)

    for (rep, replicate) in enumerate(replicates)
        for k in 1:nfolds, j in 1:nl, i in 1:ns
            for (arr, field) in zip(arrays, fields)
                data = getfield(replicate, field)
                arr[i,j,k,rep] = data[i,j,k]
            end
        end
    end

    return NamedTuple{fields}(arrays)
end