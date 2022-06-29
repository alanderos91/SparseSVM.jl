"""
Iterate by solving linear systems with a thin singular value decomposition (SVD).
"""
struct MMSVD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::MMSVD, problem::BinarySVMProblem, ::Nothing)
    @unpack n, p, y, intercept, kernel = problem
    A = get_design_matrix(problem)
    nparams = ifelse(kernel isa Nothing, p, n)

    # thin SVD of A
    U, s, V = __svd_wrapper__(A)
    r = length(s) # number of nonzero singular values

    # constants
    Abar = vec(sum(A, dims=1))
    ldiv!(n, Abar)
    
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
    @unpack n, intercept = problem
    @unpack y, coeff, proj = problem
    @unpack buffer, projection = extras
    @unpack z, Ψ, U, s, V, Abar = extras

    A = get_design_matrix(problem)
    _, w = get_params(problem)
    T = floattype(problem)    

    # need to compute z via residuals...
    apply_projection(projection, problem, k)
    __evaluate_residuals__(problem, extras, true, false)
    c1, c2, c3 = convert(T, 1/n), convert(T, rho), convert(T, lambda)
    
    # compute RHS: 1/n * Aᵀzₘ + ρ * P(wₘ)
    idx = get_projection_indices(problem)
    u = view(proj, idx)
    mul!(u, A', z, c1, c2)

    # Apply the MM update to coefficients: H*w = RHS
    H = (c2+c3, V, Ψ)
    __apply_H_inverse__!(w, H, u, buffer, zero(T))

    # Apply Schur complement in H to compute intercept and shift coefficients.
    if intercept
        v = sum(z) / n
        t = 1 - __H_inverse_quadratic__(H, Abar, buffer)
        b = v / t - dot(Abar, w)
        __apply_H_inverse__!(w, H, Abar, buffer, (b-v)/t)
        coeff[1] = b
    end

    return nothing
end

# Apply one update in reguarlized problem.
function __reg_iterate__(::MMSVD, problem::BinarySVMProblem, lambda, extras)
    @unpack n, intercept = problem
    @unpack y, coeff, proj = problem
    @unpack buffer = extras
    @unpack z, V, Ψ, Abar = extras

    A = get_design_matrix(problem)
    _, w = get_params(problem)
    T = floattype(problem)    

    # need to compute z via residuals...
    __evaluate_residuals__(problem, extras, true, false)
    c1, c3 = convert(T, 1/n), convert(T, lambda)

    # compute RHS: 1/n * Aᵀzₘ
    _, u = get_params_proj(problem)
    fill!(u, 0)
    mul!(u, A', z, c1, zero(c1))

    # Apply the MM update to coefficients: H*w = RHS
    H = (c3, V, Ψ)
    __apply_H_inverse__!(w, H, u, buffer, zero(T))

    # Apply Schur complement in H to compute intercept and shift coefficients.
    if intercept
        v = sum(z) / n
        t = 1 - __H_inverse_quadratic__(H, Abar, buffer)
        b = (v - dot(Abar, w)) / t
        __apply_H_inverse__!(w, H, Abar, buffer, -b)
        coeff[1] = b
    end

    return nothing
end
