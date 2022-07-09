"""
    fit(algorithm, problem, lambda, s; kwargs...)

Solve optimization problem at sparsity level `s`.

The solution is obtained via a proximal distance `algorithm` that gradually anneals parameter estimates
toward the target sparsity set.
"""
function fit(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, s::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing) # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, s, extras, (true,false,); kwargs...)
end

function fit(algorithm::AbstractMMAlg, problem::MultiSVMProblem, lambda::Real, s::Real; kwargs...)
    extras = [__mm_init__(algorithm, svm, nothing) for svm in problem.svm] # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, s, extras, (true,false,); kwargs...)
end

"""
    fit!(algorithm, problem, s, [extras], [update_extras]; kwargs...)

Same as `fit(algorithm, problem, s)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `rho` using `update_extras[1]` and `update_extras[2]`, respectively.

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
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.

See also: [`SparseSVM.anneal!`](@ref) for additional keyword arguments applied at the annealing step.
"""
function fit!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, s::Real,
    extras::Union{Nothing,NamedTuple}=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    nouter::Int=100,
    dtol::Real=DEFAULT_DTOL,
    rtol::Real=DEFAULT_RTOL,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    rhof::Function=DEFAULT_ANNEALING,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem
    @unpack projection = extras
    
    # Fix model size(s).
    k = sparsity_to_k(problem, s)

    # Use previous estimates in case of warm start.
    copyto!(coeff, coeff_prev)

    # Initialize rho and iteration count.
    rho, iters = rho_init, 0

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, lambda, rho, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, lambda, rho, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    init_result = __evaluate_objective__(problem, lambda, rho, extras)
    result = SubproblemResult(0, init_result)
    cb(0, problem, rho, k, result)
    old = sqrt(result.distance)

    for iter in 1:nouter
        # Solve minimization problem for fixed rho.
        result = SparseSVM.anneal!(algorithm, problem, lambda, rho, s, extras, (false,true,); verbose=verbose, cb=cb, kwargs...)

        # Update total iteration count.
        iters += result.iters

        cb(iter, problem, rho, k, result)

        # Check for convergence to constrained solution.
        dist = sqrt(result.distance)
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
    loss, obj, dist, gradsq = __evaluate_objective__(problem, lambda, rho, extras)

    if verbose
        print("\n\niters = ", iters)
        print("\n1/n ∑ᵢ max{0, 1 - yᵢ xᵢᵀ β}² = ", loss)
        print("\nobjective  = ", obj)
        print("\ndistance   = ", sqrt(dist))
        println("\n|gradient| = ", sqrt(gradsq))
    end

    return SubproblemResult(iters, loss, obj, dist, gradsq)
end

function fit!(algorithm::AbstractMMAlg, problem::MultiSVMProblem, lambda::Real, s::Real,
    extras::Union{Nothing,Vector}=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end
    
    n = length(problem.svm)
    total_iter, total_loss, total_objv, total_dist, total_grad = 0, 0.0, 0.0, 0.0, 0.0

    # Create closure to fit a particular SVM.
    function __fit__!(k)
        svm = problem.svm[k]
        r = SparseSVM.fit!(algorithm, svm, lambda, s, extras[k], update_extras; kwargs...)

        i, l, o, d, g = r
        total_iter += i
        total_loss += l
        total_objv += o
        total_dist += d
        total_grad += g
        return r
    end

    # Fit each SVM to build the classifier.
    result = __fit__!(1)
    results = [result]
    for k in 2:n
        result = __fit__!(k)
        push!(results, result)
    end
    total = SubproblemResult(total_iter, IterationResult(total_loss, total_objv, total_dist, total_grad))

    return (; total=total, result=results)
end

"""
    anneal(algorithm, problem, rho, s; kwargs...)

Solve the `rho`-penalized optimization problem at sparsity level `s`.
"""
function anneal(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, rho::Real, s::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing)
    SparseSVM.anneal!(algorithm, problem, lambda, rho, s, extras, (true,true,); kwargs...)
end

"""
    anneal!(algorithm, problem, rho, s, [extras], [update_extras]; kwargs...)

Same as `anneal(algorithm, problem, rho, s)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `rho` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `grad < gtol`, where `grad` is the Euclidean norm of the gradient and `gtol` is a tolerance parameter.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `ninner`: The maximum number of iterations (default=`10^4`).
- `gtol`: An absoluate tolerance parameter on the squared Euclidean norm of the gradient (default=`1e-6`).
- `nesterov_threshold`: The number of early iterations before applying Nesterov acceleration (default=`10`).
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.
"""
function anneal!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real, rho::Real, s::Real,
    extras::Union{Nothing,NamedTuple}=nothing,
    update_extras::NTuple{2,Bool}=(true,true);
    ninner::Int=10^4,
    gtol::Real=DEFAULT_GTOL,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem
    @unpack projection = extras

    # Fix model size(s).
    k = sparsity_to_k(problem, s)

    # Use previous estimates in case of warm start.
    copyto!(coeff, coeff_prev)

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, lambda, rho, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, lambda, rho, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    result = __evaluate_objective__(problem, lambda, rho, extras)
    cb(0, problem, rho, k, result)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "distance", "|gradient|", "rho")
    for iter in 1:ninner
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __mm_iterate__(algorithm, problem, lambda, rho, k, extras)

        # Update loss, objective, distance, and gradient.
        apply_projection(projection, problem, k)
        result = __evaluate_objective__(problem, lambda, rho, extras)

        cb(iter, problem, rho, k, result)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, sqrt(result.distance), sqrt(result.gradient), rho)
        end

        # Assess convergence.
        obj = result.objective
        if sqrt(result.gradient) < gtol
            break
        elseif iter < ninner
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end
    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev, coeff)

    return SubproblemResult(iters, result)
end

function fit(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing) # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, lambda, extras, true; kwargs...)
end

"""
```fit(algorithm, problem, lambda, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov_threshold=10], [verbose=false])```

Fit a SVM using the L2-loss / L2-regularization model.
"""
function fit!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, lambda::Real,
    extras::Union{Nothing,NamedTuple}=nothing,
    update_extras::Bool=true;
    maxiter::Int=10^3,
    gtol::Real=DEFAULT_GTOL,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem

    # Update data structures due to hyperparameters.
    update_extras && __mm_update_lambda__(algorithm, problem, lambda, extras)

    # Initialize coefficients.
    copyto!(coeff, coeff_prev)

    # Check initial values for loss, objective, distance, and norm of gradient.
    result = __evaluate_reg_objective__(problem, lambda, extras)
    cb(0, problem, lambda, 0, result)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "|gradient|")
    for iter in 1:maxiter
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __reg_iterate__(algorithm, problem, lambda, extras)

        # Update loss, objective, and gradient.
        result = __evaluate_reg_objective__(problem, lambda, extras)

        cb(iter, problem, lambda, 0, result)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, sqrt(result.gradient))
        end

        # Assess convergence.
        obj = result.objective
        if sqrt(result.gradient) < gtol
            break
        elseif iter < maxiter
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end
    if verbose print("\n\n") end

    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev, coeff)
    copyto!(proj, coeff)

    return SubproblemResult(iters, result)
end

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
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    n = length(problem.svm)
    total_iter, total_loss, total_objv, total_dist, total_grad = 0, 0.0, 0.0, 0.0, 0.0
    
    # Create closure to fit a particular SVM.
    function __fit__!(k)
        svm = problem.svm[k]
        r = SparseSVM.fit!(algorithm, svm, lambda, extras[k], update_extras; kwargs...)

        i, l, o, d, g = r
        total_iter += i
        total_loss += l
        total_objv += o
        total_dist += d
        total_grad += g
        return r
    end

    # Fit each SVM to build the classifier.
    result = __fit__!(1)
    results = [result]
    for k in 2:n
        result = __fit__!(k)
        push!(results, result)
    end
    total = SubproblemResult(total_iter, IterationResult(total_loss, total_objv, total_dist, total_grad))

    return (; total=total, result=results)
end
