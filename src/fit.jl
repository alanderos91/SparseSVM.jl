"""
    fit(algorithm, problem::BinarySVMProblem, lambda, sparsity; kwargs...)

Solve a classification `problem` at the specified `sparsity` level and regularization strength `lambda`.

The solution is obtained via a proximal distance `algorithm` that gradually anneals parameter estimates
toward the target sparsity set.
"""
function fit(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, sparsity::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing) # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, sparsity, extras, (true,false,); kwargs...)
end

"""
    fit(algorithm, problem::MultiSVMProblem, lambda, sparsity; kwargs...)

Solve a classification with nonbinary labels using multiple SVMs to define a decision boundary.

The same settings (that is, `algorithm`, `sparsity`, `lambda`, ...) are applied to each SVM.
"""
function fit(algorithm::AbstractMMAlg, problem::MultiSVMProblem, lambda::Real, sparsity::Real; kwargs...)
    extras = [__mm_init__(algorithm, svm, nothing) for svm in problem.svm] # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, sparsity, extras, (true,false,); kwargs...)
end

"""
    fit!(algorithm, problem, lambda, sparsity, [extras], [update_extras]; kwargs...)

Same as `fit` but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `sparsity` and `rho` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `dist < dtol || abs(dist - old) < rtol * (1 + old)`, where `dist` is the distance and `dtol` and `rtol` are tolerance parameters.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `nouter`: The number of outer iterations; i.e. the maximum number of `rho` values to use in annealing (default=`100`).
- `dtol`: An absolute tolerance parameter for the squared distance (default=`1e-6`).
- `rtol`: A relative tolerance parameter for the squared distance (default=`1e-6`).
- `rho_init`: The initial value for `rho` (default=1.0).
- `rho_max`: The maximum value for `rho` (default=1e8).
- `rhof`: A function `rhof(rho, iter, rho_max)` used to determine the next value for `rho` in the annealing sequence. The default multiplies `rho` by `1.2`.
- `cb`: A callback function for extending functionality.

See also: [`SparseSVM.anneal!`](@ref) for additional keyword arguments applied at the annealing step.
"""
function fit!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, sparsity::Real,
    extras::Union{Nothing,NamedTuple}=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    nouter::Int=100,
    dtol::Real=DEFAULT_DTOL,
    rtol::Real=DEFAULT_RTOL,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    rhof::Function=DEFAULT_ANNEALING,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm ", (typeof(algorithm)), ".")
    end

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem
    @unpack projection = extras
    
    # Fix model size.
    k = sparsity_to_k(problem, sparsity)

    # Use previous estimates in case of warm start.
    copyto!(coeff, coeff_prev)

    # Initialize rho and iteration count.
    rho, iters = rho_init, 0

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, lambda, rho, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, lambda, rho, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    state = __evaluate_objective__(problem, lambda, rho, extras)
    old = state.distance
    cb((0, state), problem, (;lambda=lambda, rho=rho, k=k,))

    for iter in 1:nouter
        # Solve minimization problem for fixed rho.
        (inner_iters, state) = SparseSVM.anneal!(algorithm, problem, lambda, rho, sparsity, extras, (false,true,); cb=cb, kwargs...)

        # Update total iteration count.
        iters += inner_iters

        # Check for convergence to constrained solution.
        dist = state.distance
        if dist < dtol || abs(dist - old) < rtol * (1 + old)
            break
        else
          old = dist
        end
                
        # Update according to annealing schedule.
        rho = ifelse(iter < nouter, rhof(rho, iter, rho_max), rho)
    end
    
    # Project solution to the constraint set.
    apply_projection(projection, problem, k)
    state = __evaluate_objective__(problem, lambda, rho, extras)

    return (iters, state)
end

function fit!(algorithm::AbstractMMAlg, problem::MultiSVMProblem, lambda::Real, sparsity::Real,
    extras::Union{Nothing,Vector}=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm ", (typeof(algorithm)), ".")
    end
    
    # Create closure to fit a particular SVM.
    __fit__! = let algorithm=algorithm, problem=problem, lambda=lambda, sparsity=sparsity, extras=extras, update_extras=update_extras, kwargs=kwargs
        function (k)
            return SparseSVM.fit!(algorithm, problem.svm[k], lambda, sparsity, extras[k], update_extras; kwargs...)
        end
    end

    # Fit each SVM to build the classifier.
    n = length(problem.svm)
    result = __fit__!(1)
    results = [result]
    for k in 2:n
        result = __fit__!(k)
        push!(results, result)
    end

    return results
end

"""
    anneal(algorithm, problem, rho, sparsity; kwargs...)

Solve the `rho`-penalized optimization problem at sparsity level `sparsity`.
"""
function anneal(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, rho::Real, sparsity::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing)
    SparseSVM.anneal!(algorithm, problem, lambda, rho, sparsity, extras, (true,true,); kwargs...)
end

"""
    anneal!(algorithm, problem, rho, sparsity, [extras], [update_extras]; kwargs...)

Same as `anneal(algorithm, problem, rho, sparsity)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `rho` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `grad < gtol`, where `grad` is the Euclidean norm of the gradient and `gtol` is a tolerance parameter.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `ninner`: The maximum number of iterations (default=`10^4`).
- `gtol`: An absoluate tolerance parameter on the squared Euclidean norm of the gradient (default=`1e-6`).
- `nesterov_threshold`: The number of early iterations before applying Nesterov acceleration (default=`10`).
- `cb`: A callback function for extending functionality.
"""
function anneal!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, rho::Real, sparsity::Real,
    extras::Union{Nothing,NamedTuple}=nothing,
    update_extras::NTuple{2,Bool}=(true,true);
    ninner::Int=10^4,
    gtol::Real=DEFAULT_GTOL,
    nesterov_threshold::Int=10,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm ", (typeof(algorithm)), ".")
    end

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem
    @unpack projection = extras

    # Fix model size(s) and hyperparameters.
    k = sparsity_to_k(problem, sparsity)
    hyperparams = (;lambda=lambda, rho=rho, k=k,)

    # Use previous estimates in case of warm start.
    copyto!(coeff, coeff_prev)

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, lambda, rho, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, lambda, rho, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    state = __evaluate_objective__(problem, lambda, rho, extras)
    old = state.objective

    if state.gradient < gtol
        return (0, state)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    for iter in 1:ninner
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __mm_iterate__(algorithm, problem, lambda, rho, k, extras)

        # Update loss, objective, distance, and gradient.
        apply_projection(projection, problem, k)
        state = __evaluate_objective__(problem, lambda, rho, extras)

        cb((iter, state), problem, hyperparams)

        # Assess convergence.
        obj = state.objective
        if state.gradient < gtol
            break
        elseif iter < ninner
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end

    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev, coeff)

    return (iters, state)
end

"""
```fit(algorithm, problem::BinarySVMProblem, lambda, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov_threshold=10])```

Fit a SVM using the L2-loss / L2-regularization model.
"""
function fit(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing) # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, extras, true; kwargs...)
end

function fit!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real,
    extras::Union{Nothing,NamedTuple}=nothing,
    update_extras::Bool=true;
    maxiter::Int=10^3,
    gtol::Real=DEFAULT_GTOL,
    nesterov_threshold::Int=10,
    cb::Function=DEFAULT_CALLBACK,
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm ", (typeof(algorithm)), ".")
    end

    # Fix hyperparameters.
    hyperparams = (;lambda=lambda, rho=zero(lambda), k=length(problem.coeff)-problem.intercept,)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem

    # Update data structures due to hyperparameters.
    update_extras && __mm_update_lambda__(algorithm, problem, lambda, extras)

    # Initialize coefficients.
    copyto!(coeff, coeff_prev)

    # Check initial values for loss, objective, distance, and norm of gradient.
    state = __evaluate_reg_objective__(problem, lambda, extras)
    cb((0, state), problem, hyperparams)
    old = state.objective

    if state.gradient < gtol
        return (0, state)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    for iter in 1:maxiter
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __reg_iterate__(algorithm, problem, lambda, extras)

        # Update loss, objective, and gradient.
        state = __evaluate_reg_objective__(problem, lambda, extras)

        cb((iter, state), problem, hyperparams)

        # Assess convergence.
        obj = state.objective
        if state.gradient < gtol
            break
        elseif iter < maxiter
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end

    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev, coeff)
    copyto!(proj, coeff)

    return (iters, state)
end

"""
```fit(algorithm, problem::MultiSVMProblem, lambda, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov_threshold=10])```

Fit multiple SVMs using the L2-loss / L2-regularization model.
"""
function fit(algorithm::AbstractMMAlg, problem::MultiSVMProblem, lambda::Real; kwargs...)
    extras = [__mm_init__(algorithm, svm, nothing) for svm in problem.svm] # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, extras, true; kwargs...)
end

function fit!(algorithm::AbstractMMAlg, problem::MultiSVMProblem, lambda::Real,
    extras::Union{Nothing,Vector}=nothing,
    update_extras::Bool=true;
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm ", (typeof(algorithm)), ".")
    end
    
    # Create closure to fit a particular SVM.
    __fit__! = let algorithm=algorithm, problem=problem, lambda=lambda, extras=extras, update_extras=update_extras, kwargs=kwargs
        function (k)
            return SparseSVM.fit!(algorithm, problem.svm[k], lambda, extras[k], update_extras; kwargs...)
        end
    end

    # Fit each SVM to build the classifier.
    n = length(problem.svm)
    result = __fit__!(1)
    results = [result]
    for k in 2:n
        result = __fit__!(k)
        push!(results, result)
    end

    return results
end
