"""
Iterate via Steepest Descent (SD).
"""
struct SD <: AbstractMMAlgorithm end

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

# Assume extras has the correct data structures.
__mm_init__(::SD, problem::BinarySVMProblem, extras) = extras

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

    # Find optimal step size
    ∂g_b, ∇g_w = __slope_and_coeff_views__(∇g, intercept)
    BLAS.gemv!('N', one(T), A, ∇g_w, zero(T), A∇g_w)
    ∇g_w_norm2 = BLAS.dot(∇g_w, ∇g_w)
    A∇g_w_norm2 = alpha * BLAS.dot(A∇g_w, A∇g_w) + gamma * ∇g_w_norm2
    if intercept
        numerator = ∇g_w_norm2 + ∂g_b^2
        denominator = A∇g_w_norm2 + ∂g_b^2 + 2*∂g_b * BLAS.dot(Abar, ∇g_w)
    else
        numerator = ∇g_w_norm2
        denominator = A∇g_w_norm2
    end
    indeterminate = iszero(numerator) && iszero(denominator)
    t = ifelse(indeterminate, zero(T), numerator / denominator)

    # Move in the direction of steepest descent.
    BLAS.axpy!(-t, ∇g, β)
end
