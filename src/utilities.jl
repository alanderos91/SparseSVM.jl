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
function __evaluate_gradient__(problem, ρ, extras)
    @unpack grad, res = problem
    ∇g, r, q = grad, res.main, res.dist
    X = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)
    n, _, _ = probdims(problem)

    # ∇gᵨ(β ∣ βₘ)ⱼ = -[aXᵀ bI] * [rₘ, qₘ] = -a*Xᵀrₘ - b*qₘ
    a, b = convert(T, 1 / sqrt(n)), convert(T, ρ)
    mul!(∇g, X', r)
    axpby!(-b, q, -a, ∇g)

    return nothing
end

function __evaluate_reg_gradient__(problem, λ, extras)
    @unpack coeff, res, grad = problem
    ∇g, r, β = grad, res.main, coeff
    X = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)
    n, _, _ = probdims(problem)

    # ∇gᵨ(β ∣ βₘ)ⱼ = -a*Xᵀrₘ + b*β
    a, b = convert(T, 1 / sqrt(n)), convert(T, λ)
    mul!(∇g, X', r)
    axpby!(b, β, -a, ∇g)

    return nothing
end

"""
Evaluate the penalized least squares criterion. Also updates the gradient.
This assumes that projections have been handled externally.
"""
function __evaluate_objective__(problem, ρ, extras)
    @unpack res, grad = problem
    r, q, ∇g = res.main, res.dist, grad

    __evaluate_residuals__(problem, extras, true, true)
    __evaluate_gradient__(problem, ρ, extras)

    loss = dot(r, r) # 1/n * ∑ᵢ (zᵢ - X*β)²
    dist = dot(q, q) # ∑ⱼ (P(βₘ) - β)²
    gradsq = dot(∇g, ∇g)
    obj = 1//2 * (loss + ρ * dist)

    return IterationResult(loss, obj, dist, gradsq)
end

function __evaluate_reg_objective__(problem, λ, extras)
    @unpack coeff, res, grad = problem
    β, r, ∇g = coeff, res.main, grad
    T = floattype(problem)

    __evaluate_residuals__(problem, extras, true, false)
    __evaluate_reg_gradient__(problem, λ, extras)

    loss = dot(r, r) # 1/n * ∑ᵢ (zᵢ - X*β)²
    gradsq = dot(∇g, ∇g)
    penalty = dot(β, β)
    objective = 1//2 * (loss + λ * penalty)

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
