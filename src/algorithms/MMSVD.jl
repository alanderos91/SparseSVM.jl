"""
Iterate by solving linear systems with a thin singular value decomposition (SVD).
"""
struct MMSVD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::MMSVD, problem::BinarySVMProblem, ::Nothing)
    @unpack n, p, kernel = problem
    A = get_design_matrix(problem)
    nparams = ifelse(kernel isa Nothing, p, n)

    # thin SVD of A
    U, s, V = __svd_wrapper__(A)
    r = length(s) # number of nonzero singular values

    # constants
    Abar = vec(mean(A, dims=1))
    
    # worker arrays
    z = similar(A, n)
    buffer = similar(A, r)
    
    # diagonal matrices
    Ψ = Diagonal(similar(A, r))

    return (;
        projection=L0Projection(nparams),
        U=U, s=s, V=V,
        z=z, Ψ=Ψ, Abar=Abar,
        buffer=buffer,
    )
end

# Check for data structure allocations; otherwise initialize.
function __mm_init__(::MMSVD, problem::BinarySVMProblem, extras)
    if :projection in keys(extras) && :buffer in keys(extras) # TODO
        return extras
    else
        __mm_init__(MMSVD(), problem, nothing)
    end
end

# Update data structures due to change in model size, k.
__mm_update_sparsity__(::MMSVD, problem::BinarySVMProblem, lambda, rho, k, extras) = nothing

# Update data structures due to changing rho.
__mm_update_rho__(::MMSVD, problem::BinarySVMProblem, lambda, rho, k, extras) = update_diagonal(problem, lambda, rho, extras)
# __mm_update_rho__(::MMSVD, problem::BinarySVMProblem, rho, k, extras) = update_matrices(problem, rho, extras)

# Update data structures due to changing lambda. 
__mm_update_lambda__(::MMSVD, problem::BinarySVMProblem, lambda, extras) = update_diagonal(problem, lambda, zero(lambda), extras)

function update_diagonal(problem::BinarySVMProblem, lambda, rho, extras)
    @unpack s, Ψ = extras
    n, _, _ = probdims(problem)
    T = floattype(problem)
    a², b², c² = convert(T, 1/n), convert(T, rho), convert(T, lambda)

    # Update the diagonal matrix Ψ = (a² Σ²) / (a² Σ² + b² I).
    __update_diagonal__(Ψ.diag, s, a², b², c²)

    return nothing
end

function __update_diagonal__(diag, s, a², b², c²)
    for i in eachindex(diag)
        sᵢ² = s[i]^2
        diag[i] = a² * sᵢ² / (a² * sᵢ² + b² + c²)
    end
end

# solves (A'A + γ*I) x = b using thin SVD of A
# x = γ*[I - V * Ψ * Vᵀ]*b
function __apply_H_inverse__!(x, H, b, buffer, α::Real=zero(eltype(x)))
    γ, V, Ψ = H

    if iszero(α)        # x = H⁻¹ b
        copyto!(x, b)
        BLAS.scal!(1/γ, x)
        α = one(γ)
    else                # x = x + α * H⁻¹ b
        axpy!(α/γ, b, x)
    end

    # accumulate Ψ * Vᵀ * b
    mul!(buffer, V', b)
    lmul!(Ψ, buffer)

    # complete the product with a 5-arg mul!
    mul!(x, V, buffer, -α/γ, one(γ))

    return nothing
end

function __H_inverse_quadratic__(H, x, buffer)
    γ, V, Ψ = H

    mul!(buffer, V', x)
    @inbounds for i in eachindex(buffer)
        buffer[i] = sqrt(Ψ.diag[i]) * buffer[i]
    end

    return 1/γ * (dot(x, x) - dot(buffer, buffer))
end

# Apply one update.
function __mm_iterate__(::MMSVD, problem::BinarySVMProblem, lambda, rho, k, extras)
    n = problem.n
    T = floattype(problem)
    c1, c2, c3 = convert(T, 1/n), convert(T, rho), convert(T, lambda)
    
    f = function(problem, extras)
        A = get_design_matrix(problem)

        # LHS: H = A'A + λI; pass as (γ, V, Ψ) which computes H⁻¹ = γ[I - V Ψ Vᵀ]
        H = (c2+c3, extras.V, extras.Ψ)

        # RHS: u = 1/n * Aᵀzₘ + ρ * P(wₘ)
        _, u = get_params_proj(problem)
        mul!(u, A', extras.z, c1, c2)

        return H, u
    end

    apply_projection(extras.projection, problem, k)
    __evaluate_residuals__(problem, extras, true, false)
    __linear_solve_SVD__(f, problem, extras)

    return nothing
end

# Apply one update in reguarlized problem.
function __reg_iterate__(::MMSVD, problem::BinarySVMProblem, lambda, extras)
    n = problem.n
    T = floattype(problem)
    c1, c3 = convert(T, 1/n), convert(T, lambda)

    f = function(problem, extras)
        A = get_design_matrix(problem)

        # LHS: H = A'A + λI; pass as (γ, V, Ψ) which computes H⁻¹ = γ[I - V Ψ Vᵀ]
        H = (c3, extras.V, extras.Ψ)

        # RHS: u = 1/n * Aᵀzₘ
        _, u = get_params_proj(problem)
        fill!(u, 0)
        mul!(u, A', extras.z, c1, zero(c1))

        return H, u
    end

    __evaluate_residuals__(problem, extras, true, false)
    __linear_solve_SVD__(f, problem, extras)

    return nothing
end

#
#   NOTE: worker arrays must not be aliased with coefficients, w!!!
#
function __linear_solve_SVD__(compute_LHS_and_RHS::Function, problem::BinarySVMProblem, extras)
    @unpack n, intercept = problem
    @unpack y, coeff, proj = problem
    @unpack buffer = extras
    @unpack z, V, Ψ, Abar = extras

    _, w = get_params(problem)
    T = floattype(problem)    
    H, u = compute_LHS_and_RHS(problem, extras)

    # Apply the MM update to coefficients: H*w = RHS
    __apply_H_inverse__!(w, H, u, buffer, zero(T))

    # Apply Schur complement in H to compute intercept and shift coefficients.
    if intercept
        v = mean(z)
        t = 1 - __H_inverse_quadratic__(H, Abar, buffer)
        b = (v - dot(Abar, w)) / t
        __apply_H_inverse__!(w, H, Abar, buffer, -b)
        __set_intercept_component__!(coeff, b)
    end

    return nothing
end
