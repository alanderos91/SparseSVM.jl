"""
Iterate by solving linear systems with a thin singular value decomposition (SVD).
"""
struct MMSVD <: AbstractMMAlgorithm end

# Initialize data structures.
function __mm_init__(::MMSVD, problem::BinarySVMProblem, ::Nothing)
    @unpack n, p, kernel = problem
    A = get_design_matrix(problem)
    nparams = ifelse(kernel isa Nothing, p, n)
    T = floattype(problem)

    # thin SVD of A
    F = __svd_wrapper__(A)
    r = length(F.S) # number of nonzero singular values

    # constants
    Abar = vec(mean(A, dims=1))
    
    # worker arrays
    z = similar(A, n); fill!(z, zero(T))
    buffer = similar(A, r); fill!(buffer, zero(T))
    
    # diagonal matrices
    Ψ = Diagonal(similar(A, r)); fill!(Ψ.diag, zero(T))

    return (;
        projection=L0Projection(nparams),
        U=F.U, s=F.S, V=Matrix(F.V),
        z=z, Ψ=Ψ, Abar=Abar,
        buffer=buffer,
    )
end

function __mm_init__(::MMSVD, problem::MultiSVMProblem, ::Nothing)
    kernel, strategy = problem.kernel, problem.strategy
    if kernel isa Nothing && strategy isa OVR
        # only need 1 SVD since X and problem dimensions do not change between binary SVMs
        extras_1 = __mm_init__(MMSVD(), problem.svm[1], nothing)
        extras = [extras_1 for _ in problem.svm]
    else # kernel isa Kernel || strategy isa OVO
        # need a different SVD in each binary SVM
        extras = [__mm_init__(MMSVD(), svm, nothing) for svm in problem.svm]
    end
    return extras
end

# Assume extras has the correct data structures.
__mm_init__(::MMSVD, problem, extras) = extras

# Update data structures due to change in model size, k.
__mm_update_sparsity__(::MMSVD, problem::BinarySVMProblem, lambda, rho, k, extras) = nothing

# Update data structures due to changing rho.
__mm_update_rho__(::MMSVD, problem::BinarySVMProblem, lambda, rho, k, extras) = update_diagonal(problem, lambda, rho, extras)

# Update data structures due to changing lambda. 
__mm_update_lambda__(::MMSVD, problem::BinarySVMProblem, lambda, extras) = update_diagonal(problem, lambda, zero(lambda), extras)

function update_diagonal(problem::BinarySVMProblem, lambda, rho, extras)
    @unpack s, Ψ = extras
    n, _, _ = probdims(problem)
    T = floattype(problem)
    a, b, c = convert(T, 1/n), convert(T, rho), convert(T, lambda)

    # Update the diagonal matrix Ψ = (a² Σ²) / (a² Σ² + b² I).
    __update_diagonal__(Ψ.diag, s, a, b, c)

    return nothing
end

function __update_diagonal__(diag, s, a, b, c)
    @inbounds for i in eachindex(diag)
        s2_i = s[i]^2
        diag[i] = a * s2_i / (a * s2_i + b + c)
    end
end

# solves (A'A + γ*I) x = b using thin SVD of A
function __apply_H_inverse__!(x, H, b, buffer, α::Real=zero(eltype(x)))
    γ, V, Ψ = H

    if iszero(α)        # x = H⁻¹ b
        copyto!(x, b)
        BLAS.scal!(1/γ, x)
        α = one(γ)
    else                # x = x + α * H⁻¹ b
        BLAS.axpy!(α/γ, b, x)
    end

    # accumulate Ψ * Vᵀ * b
    BLAS.gemv!('T', one(γ), V, b, zero(γ), buffer)
    lmul!(Ψ, buffer)

    # complete the product with a 5-arg mul!
    BLAS.gemv!('N', -α/γ, V, buffer, one(γ), x)

    return nothing
end

function __H_inverse_quadratic__(H, x, buffer)
    γ, V, Ψ = H

    T = eltype(buffer)
    BLAS.gemv!('T', one(T), V, x, zero(T), buffer)
    @inbounds for i in eachindex(buffer)
        buffer[i] = sqrt(Ψ.diag[i]) * buffer[i]
    end

    return 1/γ * (BLAS.dot(x, x) - BLAS.dot(buffer, buffer))
end

# Apply one update.
function __mm_iterate__(::MMSVD, problem::BinarySVMProblem, lambda, rho, k, extras)
    n = problem.n
    T = floattype(problem)
    c1, c2, c3 = convert(T, 1/n), convert(T, rho), convert(T, lambda)
    
    f = let c1=c1, c2=c2, c3=c3
        function(problem, extras)
            A = get_design_matrix(problem)

            # LHS: H = A'A + λI; pass as (γ, V, Ψ) which computes H⁻¹ = γ⁻¹[I - V Ψ Vᵀ]
            H = (c2+c3, extras.V, extras.Ψ)

            # RHS: u = 1/n * Aᵀzₘ + ρ * P(wₘ)
            _, u = get_params_proj(problem)
            BLAS.gemv!('T', c1, A, extras.z, c2, u)

            return H, u
        end
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

    f = let c1=c1, c3=c3
        function(problem, extras)
            A = get_design_matrix(problem)

            # LHS: H = A'A + λI; pass as (γ, V, Ψ) which computes H⁻¹ = γ[I - V Ψ Vᵀ]
            H = (c3, extras.V, extras.Ψ)

            # RHS: u = 1/n * Aᵀzₘ
            _, u = get_params_proj(problem)
            fill!(u, 0)
            BLAS.gemv!('T', c1, A, extras.z, zero(c1), u)

            return H, u
        end
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
