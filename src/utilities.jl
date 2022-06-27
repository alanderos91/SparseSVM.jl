function shifted_response!(z, y, Xβ)
    @inbounds for i in eachindex(z)
        yi, Xβi = y[i], Xβ[i]
        z[i] = ifelse(yi*Xβi < 1, yi, Xβi)
    end
    return nothing
end

"""
Generic template for evaluating residuals.
This assumes that projections have been handled externally.
The following flags control how residuals are evaluated:

+ `need_main`: If `true`, evaluates regression residuals.
+ `need_dist`: If `true`, evaluates difference between parameters and their projection.

**Note**: The values for each flag should be known at compile-time!
"""
function __evaluate_residuals__(problem::BinarySVMProblem, extras, need_main::Bool, need_dist::Bool)
    @unpack y, coeff, proj, res = problem
    @unpack z = extras
    β, pₘ, r, q = coeff, proj, res.main, res.dist
    X = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)
    n, _, _ = probdims(problem)
    
    if need_main # 1/sqrt(n) * (zₘ - X*β)
        a = convert(T, 1 / sqrt(n))
        mul!(r, X, β)
        shifted_response!(z, y, r)
        axpby!(a, z, -a, r)
    end

    if need_dist # P(βₘ) - β
        @. q = pₘ - β
    end

    return nothing
end

"""
Evaluate the gradiant of the regression problem. Assumes residuals have been evaluated.
"""
function __evaluate_gradient__(problem, rho, extras)
    @unpack grad, res = problem
    ∇g, r, q = grad, res.main, res.dist
    X = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)
    n, _, _ = probdims(problem)

    # ∇gᵨ(β ∣ βₘ)ⱼ = -[aXᵀ bI] * [rₘ, qₘ] = -a*Xᵀrₘ - b*qₘ
    a, b = convert(T, 1 / sqrt(n)), convert(T, rho)
    mul!(∇g, X', r)
    axpby!(-b, q, -a, ∇g)

    return nothing
end

function __evaluate_reg_gradient__(problem, lambda, extras)
    @unpack coeff, res, grad = problem
    ∇g, r, β = grad, res.main, coeff
    X = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)
    n, _, _ = probdims(problem)

    # ∇gᵨ(β ∣ βₘ)ⱼ = -a*Xᵀrₘ + b*β
    a, b = convert(T, 1 / sqrt(n)), convert(T, lambda)
    mul!(∇g, X', r)
    axpby!(b, β, -a, ∇g)

    return nothing
end

"""
Evaluate the penalized least squares criterion. Also updates the gradient.
This assumes that projections have been handled externally.
"""
function __evaluate_objective__(problem, rho, extras)
    @unpack res, grad = problem
    r, q, ∇g = res.main, res.dist, grad

    __evaluate_residuals__(problem, extras, true, true)
    __evaluate_gradient__(problem, rho, extras)

    loss = dot(r, r) # 1/n * ∑ᵢ (zᵢ - X*β)²
    dist = dot(q, q) # ∑ⱼ (P(βₘ) - β)²
    gradsq = dot(∇g, ∇g)
    obj = 1//2 * (loss + rho * dist)

    return IterationResult(loss, obj, dist, gradsq)
end

function __evaluate_reg_objective__(problem, lambda, extras)
    @unpack coeff, res, grad = problem
    β, r, ∇g = coeff, res.main, grad
    T = floattype(problem)

    __evaluate_residuals__(problem, extras, true, false)
    __evaluate_reg_gradient__(problem, lambda, extras)

    loss = dot(r, r) # 1/n * ∑ᵢ (zᵢ - X*β)²
    gradsq = dot(∇g, ∇g)
    penalty = dot(β, β)
    objective = 1//2 * (loss + lambda * penalty)

    return IterationResult(loss, objective, 0.0, gradsq)
end

"""
Apply acceleration to the current iterate `x` based on the previous iterate `y`
according to Nesterov's method with parameter `r=3` (default).
"""
function __apply_nesterov__!(x, y, iter::Integer, needs_reset::Bool, r::Int=3)
    if needs_reset # Reset acceleration scheme
        copyto!(y, x)
        iter = 1
    else # Nesterov acceleration 
        γ = (iter - 1) / (iter + r - 1)
        @inbounds for i in eachindex(x)
            xi, yi = x[i], y[i]
            zi = xi + γ * (xi - yi)
            x[i], y[i] = zi, xi
        end
        iter += 1
    end

    return iter
end

"""
Map a sparsity level `s` to an integer `k`, assuming `n` elements.
"""
sparsity_to_k(problem::AbstractSVM, s) = __sparsity_to_k__(problem.kernel, problem, s)
__sparsity_to_k__(::Nothing, problem::AbstractSVM, s) = round(Int, (1-s) * problem.p)
__sparsity_to_k__(::Kernel, problem::AbstractSVM, s) = round(Int, (1-s) * problem.n)

get_projection_indices(problem::AbstractSVM) = __get_projection_indices__(problem.kernel, problem)
__get_projection_indices__(::Nothing, problem::AbstractSVM) = 1:problem.p
__get_projection_indices__(::Kernel, problem::AbstractSVM) = 1:problem.n

"""
Apply a projection to model coefficients.
"""
function apply_projection(projection, problem, k)
    idx = get_projection_indices(problem)
    @unpack coeff, proj = problem
    copyto!(proj, coeff)

    # projection step, might not be unique
    if problem.intercept
        projection(view(proj, idx), k)
    else
        projection(proj, k)
    end

    return proj
end

"""
Define a geometric progression recursively by the rule
```
    rho_new = min(rho_max, rho * multiplier)
```
The result is guaranteed to have type `typeof(rho)`.
"""
function geometric_progression(rho, iter, rho_max, multiplier::Real=1.2)
    return convert(typeof(rho), min(rho_max, rho * multiplier))
end

function prediction_error(model, train_set, validation_set, test_set)
    # Extract number of features to make predictions consistent.
    @unpack p = model

    # Extract data for each set.
    Tr_Y, Tr_X = train_set
    V_Y, V_X = validation_set
    T_Y, T_X = test_set

    # Helper function to make predictions on each subset and evaluate errors.
    _error = function(model, L, X)
        # Sanity check: Y and X have the same number of rows.
        length(L) != size(X, 1) && error("Labels ($(length(L))) not compatible with data X ($(size(X))).")

        # Classify response in vertex space; may use @batch.
        Lhat = SparseSVM.classify(model, X)

        # Sweep through predictions and tally the mistakes.
        nincorrect = sum(L .!= Lhat)

        return 100 * nincorrect / length(L)
    end

    Tr = _error(model, Tr_Y, view(Tr_X, :, 1:p))
    V = _error(model, V_Y, view(V_X, :, 1:p))
    T = _error(model, T_Y, view(T_X, :, 1:p))

    return (Tr, V, T)
end

function set_initial_coefficients!(::Nothing, train_coeff, coeff, idx)
    copyto!(train_coeff, coeff)
end

function set_initial_coefficients!(::Kernel, train_coeff, coeff, idx)
    for (i, idx_i) in enumerate(idx)
        train_coeff[i] = coeff[idx_i]
    end
end

function set_initial_coefficients!(train_problem::BinarySVMProblem, problem::BinarySVMProblem, idx)
    set_initial_coefficients!(train_problem.kernel, train_problem.coeff, problem.coeff, idx)
    set_initial_coefficients!(train_problem.kernel, train_problem.coeff_prev, problem.coeff_prev, idx)
    nothing
end

function set_initial_coefficients!(train_problem::MultiSVMProblem, problem::MultiSVMProblem, idxs)
    for (i, (train_svm, svm, idx)) in enumerate(zip(train_problem.svm, problem.svm, idxs))
        set_initial_coefficients!(train_svm, svm, idx)
        __copy_coefficients!__(train_problem.kernel, train_problem, train_svm, i)
    end
    nothing
end

function extract_indices(problem::BinarySVMProblem, labels)
    intersect(parentindices(problem.labels), parentindices(labels))
end

function extract_indices(problem::MultiSVMProblem, labels)
    [extract_indices(svm, labels) for svm in problem.svm]
end

__copy_coefficients!__(::Nothing, problem::MultiSVMProblem, subproblem::BinarySVMProblem, i::Integer) = nothing
__copy_coefficients!__(::Nothing, subproblem::BinarySVMProblem, problem::MultiSVMProblem, i::Integer) = nothing

function __copy_coefficients!__(::Kernel, problem::MultiSVMProblem, subproblem::BinarySVMProblem, i::Integer)
    @unpack subset, intercept = problem
    @unpack coeff, coeff_prev, proj = subproblem
    idx = subset[i]

    if !(coeff isa SubArray)
        for field in (:coeff, :coeff_prev, :proj)
            src = getfield(subproblem, field)
            dst = getfield(problem, field)
            nparams = length(idx)
            dst[idx, i] .= view(src, 1:nparams)
            if intercept
                dst[nparams+1,i] = src[nparams+1]
            end
        end
    end

    return nothing
end

function __copy_coefficients!__(::Kernel, subproblem::BinarySVMProblem, problem::MultiSVMProblem, i::Integer)
    @unpack subset, intercept = problem
    @unpack coeff, coeff_prev, proj = subproblem
    idx = subset[i]

    if !(coeff isa SubArray)
        for field in (:coeff, :coeff_prev, :proj)
            src = getfield(problem, field)
            dst = getfield(subproblem, field)
            nparams = length(idx)
            dst[1:nparams] .= view(src, idx, i)
            if intercept
                dst[nparams+1] = src[nparams+1,i]
            end
        end
    end

    return nothing
end

function copy_from_buffer!(problem::MultiSVMProblem)
    for (i, subproblem) in enumerate(problem.svm)
        __copy_coefficients!__(problem.kernel, problem, subproblem, i)
    end
end

function copy_to_buffer!(problem::MultiSVMProblem)
    for (i, subproblem) in enumerate(problem.svm)
        __copy_coefficients!__(problem.kernel, subproblem, problem, i)
    end
end


"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__(iter, problem, rho, k, history) = nothing
# __do_nothing_callback__(fold, problem, train_problem, data, lambda, sparsity, model_size, result) = nothing

__svd_wrapper__(A::StridedMatrix) = svd(A, full=false)
__svd_wrapper__(A::AbstractMatrix) = svd!(copy(A), full=false)

struct IterationResult
    loss::Float64
    objective::Float64
    distance::Float64
    gradient::Float64
end

# destructuring
Base.iterate(r::IterationResult) = (r.loss, Val(:objective))
Base.iterate(r::IterationResult, ::Val{:objective}) = (r.objective, Val(:distance))
Base.iterate(r::IterationResult, ::Val{:distance}) = (r.distance, Val(:gradient))
Base.iterate(r::IterationResult, ::Val{:gradient}) = (r.gradient, Val(:done))
Base.iterate(r::IterationResult, ::Val{:done}) = nothing

struct SubproblemResult
    iters::Int
    loss::Float64
    objective::Float64
    distance::Float64
    gradient::Float64
end

function SubproblemResult(iters, r::IterationResult)
    return SubproblemResult(iters, r.loss, r.objective, r.distance, r.gradient)
end

# destructuring
Base.iterate(r::SubproblemResult) = (r.iters, Val(:loss))
Base.iterate(r::SubproblemResult, ::Val{:loss}) = (r.loss, Val(:objective))
Base.iterate(r::SubproblemResult, ::Val{:objective}) = (r.objective, Val(:distance))
Base.iterate(r::SubproblemResult, ::Val{:distance}) = (r.distance, Val(:gradient))
Base.iterate(r::SubproblemResult, ::Val{:gradient}) = (r.gradient, Val(:done))
Base.iterate(r::SubproblemResult, ::Val{:done}) = nothing
