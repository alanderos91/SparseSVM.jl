"""
Iterate via Steepest Descent (SD).
"""
struct SD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::SD, problem::BinarySVMProblem, ::Nothing)
    @unpack n, p, kernel = problem
    A = get_design_matrix(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # constants
    Abar = vec(mean(A, dims=1))

    # worker arrays
    z = similar(A, n)

    return (; projection=L0Projection(nparams), z=z, Abar=Abar,)
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
__mm_update_sparsity__(::SD, problem::BinarySVMProblem, lambda, rho, k, extras) = nothing

# Update data structures due to changing rho.
__mm_update_rho__(::SD, problem::BinarySVMProblem, lambda, rho, k, extras) = nothing

# Update data structures due to changing lambda.
__mm_update_lambda__(::SD, problem::BinarySVMProblem, lambda, extras) = nothing

# Apply one update.
function __mm_iterate__(::SD, problem::BinarySVMProblem, lambda, rho, k, extras)
    n = problem.n
    T = floattype(problem)
    c1, c2, c3 = convert(T, 1/n), convert(T, rho), convert(T, lambda)

    apply_projection(extras.projection, problem, k)
    __evaluate_residuals__(problem, extras, true, true)
    __evaluate_gradient__(problem, lambda, rho, extras)
    __steepest_descent__(problem, extras, c1, c2+c3)

    return nothing
end

# Apply one update in regularized problem.
function __reg_iterate__(::SD, problem::BinarySVMProblem, lambda, extras)
    n = problem.n
    T = floattype(problem)
    c1, c3 = convert(T, 1/n), convert(T, lambda)

    __evaluate_residuals__(problem, extras, true, false)
    __evaluate_reg_gradient__(problem, lambda, extras)
    __steepest_descent__(problem, extras, c1, c3)

    return nothing
end

function __steepest_descent__(problem, extras, alpha, gamma)
    @unpack coeff, grad, res, intercept = problem
    @unpack Abar = extras
    β, ∇g = coeff, grad
    A = get_design_matrix(problem)
    A∇g_w = res.main
    T = floattype(problem)

    if intercept
        ∂g_b, ∇g_w = grad[1], view(∇g, 2:length(∇g))
        intercept_term = ∂g_b^2 + 2*∂g_b * dot(Abar, ∇g_w)
    else
        ∂g_b, ∇g_w = zero(T), view(∇g, 1:length(∇g))
        intercept_term = zero(T)
    end

    # Find optimal step size
    mul!(A∇g_w, A, ∇g_w)
    ∇gnorm2 = dot(∇g_w, ∇g_w)
    A∇gnorm2 = alpha * dot(A∇g_w, A∇g_w) + gamma * ∇gnorm2 + intercept_term
    indeterminate = iszero(∇gnorm2) && iszero(A∇gnorm2)
    t = ifelse(indeterminate, zero(T), ∇gnorm2 / A∇gnorm2)

    # Move in the direction of steepest descent.
    axpy!(-t, ∇g, β)
end
