abstract type AbstractLIBSVMAlgorithm <: AbstractAlgorithm end

function get_libsvm_object(algorithm::AbstractLIBSVMAlgorithm, problem::AbstractSVM, lambda, kwargs)
    # Dispatch on selected kernel.
    create_classifier, kwargs = get_libsvm_object(algorithm, problem.kernel, kwargs)

    # LIBLINEAR use an intercept if bias < 0.
    bias = Float64(problem.intercept - 1)

    # Translate lambda penalty parameter (on weights) to cost penalty parameter (on loss)
    cost = 1 / (2 * problem.n * lambda)

    if create_classifier <: LIBSVM.LinearSVC
        kwargs = (;kwargs..., cost=cost, bias=bias)
    else
        kwargs = (;kwargs..., cost=cost)
    end

    return create_classifier(;kwargs...)
end

"""
Squared hinge loss with L2 penalty on weights.
"""
struct L2SVM <: AbstractLIBSVMAlgorithm end

function get_libsvm_object(::L2SVM, ::Nothing, kwargs)
    kwargs =(;kwargs..., solver=LIBSVM.Linearsolver.L2R_L2LOSS_SVC)    
    return LIBSVM.LinearSVC, kwargs
end

function get_libsvm_object(::L2SVM, ::RBFKernel, kwargs)
    kwargs = (;kwargs..., kernel=LIBSVM.Kernel.RadialBasis, gamma=1.0)
    return LIBSVM.SVC, kwargs
end

function get_libsvm_object(::L2SVM, kernel::TransformedKernel{<:RBFKernel,<:ScaleTransform}, kwargs)
    kwargs = (;kwargs..., kernel=LIBSVM.Kernel.RadialBasis, gamma=kernel.transform.s^2)
    return LIBSVM.SVC, kwargs
end

"""
Squared hinge loss with L1 penalty on weights.
"""
struct L1SVM <: AbstractLIBSVMAlgorithm end

function get_libsvm_object(::L1SVM, ::Nothing, kwargs)
    kwargs = (;kwargs..., solver=LIBSVM.Linearsolver.L1R_L2LOSS_SVC)
    return LIBSVM.LinearSVC, kwargs
end

function fit(algorithm::AbstractLIBSVMAlgorithm, problem::AbstractSVM, lambda::Real; kwargs...)
    L, X = get_labeled_data(problem)
    classifier = get_libsvm_object(algorithm, problem, lambda, kwargs)
    model = LIBSVM.fit!(classifier, X, L)
    transfer_model!(problem, model)
    if problem isa BinarySVMProblem
        statistics = (0, __evaluate_reg_objective__(problem, lambda, (;z=similar(problem.res.main))))
    else
        statistics = [(0, __evaluate_reg_objective__(svm, lambda, (;z=similar(svm.res.main)))) for svm in problem.svm]
    end
    return statistics
end

function cv(algorithm::AbstractLIBSVMAlgorithm, problem, lambda_grid::G; at::Real=0.8, kwargs...) where G
    # Split data into cross-validation and test sets.
    @unpack p, labels, intercept = problem
    X = get_problem_data(problem)
    dataset_split = splitobs((labels, view(X, :, 1:p)), at=at, obsdim=1)
    SparseSVM.cv(algorithm, problem, lambda_grid, dataset_split; kwargs...)
end

function cv(algorithm::AbstractLIBSVMAlgorithm, problem, lambda_grid::G, dataset_split::Tuple{S1,S2};
    nfolds::Int=5,
    scoref::Function=DEFAULT_SCORE_FUNCTION,
    cb::Function=DEFAULT_CALLBACK,
    show_progress::Bool=true,
    progress_bar::Progress=Progress(nfolds * length(lambda_grid); desc="Running CV w/ $(algorithm)... ", enabled=show_progress),
    data_transform::Type{Transform}=ZScoreTransform,
    kwargs...) where {G,S1,S2,Transform}
    # Sanity checks.    
    if any(<(0), lambda_grid)
        error("Values in λ grid should be positive.")
    end

    # Initialize the output.
    cv_set, test_set = dataset_split
    ns, nl = 1, length(lambda_grid)
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

        for (j, lambda) in enumerate(lambda_grid)
            i = 1

            timed_result = @timed SparseSVM.fit(algorithm, train_problem, lambda; kwargs...)

            hyperparams = (;sparsity=0.0, lambda=lambda,)
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
            next!(progress_bar, showvalues=[(:fold, k), (:lambda, lambda)])
        end
    end

    return result
end

function repeated_cv(algorithm::AbstractLIBSVMAlgorithm, problem, lambda_grid::G; at::Real=0.8, kwargs...) where G
    # Split data into cross-validation and test sets.
    @unpack p, labels, intercept = problem
    X = get_problem_data(problem)
    dataset_split = splitobs((labels, view(X, :, 1:p)), at=at, obsdim=1)
    SparseSVM.repeated_cv(algorithm, problem, lambda_grid, dataset_split; kwargs...)
end

function repeated_cv(algorithm::AbstractLIBSVMAlgorithm, problem, lambda_grid::G, dataset_split::Tuple{S1,S2};
    nfolds::Int=5,
    nreplicates::Int=10,
    show_progress::Bool=true,
    rng::AbstractRNG=StableRNG(1903),
    cb::Function=DEFAULT_CALLBACK,
    kwargs...) where {G,S1,S2}
    # Retrieve subsets and create index set into cross-validation set.
    cv_set, test_set = dataset_split

    progress_bar = Progress(nreplicates * nfolds * length(lambda_grid); desc="Repeated CV... ", enabled=show_progress)

    # Replicate CV procedure several times.
    keys = (:train,:validation,:test,:time)
    types = NTuple{4,Array{Float64,3}}
    replicate = Vector{NamedTuple{keys,types}}(undef, nreplicates)

    for rep in 1:nreplicates
        # Shuffle cross-validation data.
        cv_shuffled = shuffleobs(cv_set, obsdim=1, rng=rng)

        # Run k-fold cross-validation and store results.
        cb_rep = cb(rep)
        replicate[rep] = SparseSVM.cv(algorithm, problem, lambda_grid, (cv_shuffled, test_set);
            nfolds=nfolds, show_progress=show_progress, progress_bar=progress_bar, cb=cb_rep, kwargs...)
    end

    return replicate
end

##### Helpers for copying coefficients from LIBSVM/LIBLINEAR to SparseSVM #####
#=
    These functions assume problem and model are compatible. This means:

    - There is no check for intercepts. If LIBSVM/LIBLINEAR fit intercepts, they appear in the last components.
      Be careful in case we decide to change the formatting for our models.
    - LIBLINEAR always uses OVR and coefficients are layed out as n_classes × n_features.
    - LIBSVM always uses OVO and coefficients are layed out as n_support_vectors × n_classes-1.
=#

function transfer_model!(problem::BinarySVMProblem, model::LIBSVM.LinearSVC)
    coefficients = model.fit.w
    # Check if classes are flipped.
    if model.fit.labels[1] != poslabel(problem)
        coefficients .*= -1
    end
    copyto!(problem.coeff, coefficients)
    copyto!(problem.coeff_prev, coefficients)
    copyto!(problem.proj, coefficients)
    return nothing
end

function transfer_model!(problem::MultiSVMProblem, model::LIBSVM.LinearSVC)
    if problem.strategy isa OVO
        error("Cannot translate OVR coeffficients to OVO for MultiSVMProblem.")
    end
    intercept = problem.intercept
    _, n_features, n_classes = SparseSVM.probdims(problem)
    class_idx = sortperm(model.fit.labels)
    coefficients = reshape(model.fit.w, n_classes, n_features + intercept)'
    coefficients .= coefficients[:, class_idx]
    copyto!(problem.coeff, coefficients)
    copyto!(problem.coeff_prev, coefficients)
    copyto!(problem.proj, coefficients)
    return nothing
end

function transfer_model!(problem::BinarySVMProblem, model::LIBSVM.SVC)
    intercept = problem.intercept
    alpha, b = model.fit.coefs, model.fit.rho
    coefficients = similar(problem.coeff)
    fill!(coefficients, 0)
    idx = model.fit.SVs.indices
    for (i, w_i) in zip(idx, alpha)
        coefficients[i] = w_i
    end
    if intercept
        coefficients[end] = b[1]
    end
    copyto!(problem.coeff, coefficients)
    copyto!(problem.coeff_prev, coefficients)
    copyto!(problem.proj, coefficients)
    return nothing
end

# function transfer_model!(problem::MultiSVMProblem, model::LIBSVM.SVC)
#     if problem.strategy isa OVR
#         error("Translation of OVO coeffficients (LIBSVM.SVC) to OVR (SparseSVM) is not implemented.")
#     end
#     intercept = problem.intercept
#     alpha, b = model.fit.coefs, model.fit.rho
#     coefficients = similar(problem.coeff)
#     idx, nsv, pos = Vector{Int}[], model.fit.SVs.nSV, 1
#     for k in eachindex(coefficients)
#         blk = pos:pos+nsv[k]-1
#         push!(idx, model.fit.SVs.indices[blk])
#         pos += nsv[k]
#     end
#     for k in eachindex(coefficients)
#         pos = 1
#         svm = problem.svm[k]
#         subset = problem.subset[k]
#         slice = coefficients[k] = zeros(svm.n + intercept)
#         blk = pos:pos+nsv[k]-1
#         for l in 1:problem.c-1, (j, i) in enumerate(idx[k])
#             _i = findfirst(isequal(i), subset[k])
#             slice[_i] = alpha[blk[j],l]
#         end
#         pos += nsv[k]
#     end
#     for arr in (problem.coeff, problem.coeff_prev, problem.proj)
#         foreach(k -> copyto!(arr[k], coefficients[k]), eachindex(coefficients));
#     end
#     return nothing
# end
