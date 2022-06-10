"""
Iterate via Steepest Descent (SD).
"""
struct SD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::SD, problem::BinarySVMProblem, ::Nothing)
    @unpack X, coeff = problem
    n, p, _ = probdims(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # worker arrays
    z = similar(X, n)

    return (; projection=L0Projection(nparams), z=z,)
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::SD, problem::BinarySVMProblem, extras)
    if :projection in keys(extras) && :Z in keys(extras) # TODO
        return extras
    else
        __mm_init__(SD(), problem, nothing)
    end
end

# Update data structures due to change in model subsets, k.
__mm_update_sparsity__(::SD, problem::BinarySVMProblem, ρ, k, extras) = nothing

# Update data structures due to changing ρ.
__mm_update_rho__(::SD, problem::BinarySVMProblem, ρ, k, extras) = nothing

# Update data structures due to changing λ.
__mm_update_lambda__(::SD, problem::BinarySVMProblem, λ, extras) = nothing

# Apply one update.
function __mm_iterate__(::SD, problem::BinarySVMProblem, ρ, k, extras)
    @unpack coeff, grad, res = problem
    @unpack projection = extras
    β, ∇g, X∇g = coeff, grad, res.main
    X = get_design_matrix(problem)
    n, _, _ = probdims(problem)
    T = floattype(problem)

    # Project and then evaluate gradient.
    apply_projection(projection, problem, k)
    __evaluate_residuals__(problem, extras, true, true)
    __evaluate_gradient__(problem, ρ, extras)

    # Find optimal step size
    a², b² = convert(T, 1/n), convert(T, ρ)
    mul!(X∇g, X, ∇g)
    C1 = dot(∇g, ∇g)
    C2 = dot(X∇g, X∇g)
    t = ifelse(iszero(C1) && iszero(C2), 0.0, C1 / (a²*C2 + b²*C1))

    # Move in the direction of steepest descent.
    axpy!(-t, ∇g, β)

    return nothing
end

# Apply one update in regularized problem.
function __reg_iterate__(::SD, problem::BinarySVMProblem, λ, extras)
    @unpack coeff, grad, res = problem
    β, ∇g, X∇g = coeff, grad, res.main
    X = get_design_matrix(problem)
    n, _, _ = probdims(problem)
    T = floattype(problem)

    # Evaluate the gradient using residuals.
    __evaluate_residuals__(problem, extras, true, false)
    __evaluate_reg_gradient__(problem, λ, extras)

    # Find optimal step size
    a², b² = convert(T, 1/n), convert(T, λ)
    mul!(X∇g, X, ∇g)
    C1 = dot(∇g, ∇g)
    C2 = dot(X∇g, X∇g)
    t = ifelse(iszero(C1) && iszero(C2), zero(T), C1 / (a²*C2 + b²*C1))

    # Move in the direction of steepest descent.
    axpy!(-t, ∇g, β)

    return nothing
end
