"""
Iterate by solving linear systems with a thin singular value decomposition (SVD).
"""
struct MMSVD <: AbstractMMAlg end

# Initialize data structures.
function __mm_init__(::MMSVD, problem::BinarySVMProblem, ::Nothing)
    @unpack coeff = problem
    X = get_design_matrix(problem)
    n, p, _ = probdims(problem)
    T = floattype(problem)
    nparams = ifelse(problem.kernel isa Nothing, p, n)

    # thin SVD of X
    U, s, V = __svd_wrapper__(X)
    r = length(s) # number of nonzero singular values

    # worker arrays
    z = similar(X, n)
    buffer = similar(X, r)
    
    # diagonal matrices
    Ψ = Diagonal(similar(X, r))
    # VΨVt = similar(X, length(coeff), length(coeff))

    return (;
        projection=L0Projection(nparams),
        U=U, s=s, V=V,
        z=z, Ψ=Ψ, #VΨVt=VΨVt,
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
__mm_update_sparsity__(::MMSVD, problem::BinarySVMProblem, ρ, k, extras) = nothing

# Update data structures due to changing ρ.
__mm_update_rho__(::MMSVD, problem::BinarySVMProblem, ρ, k, extras) = update_diagonal(problem, ρ, extras)
# __mm_update_rho__(::MMSVD, problem::BinarySVMProblem, ρ, k, extras) = update_matrices(problem, ρ, extras)

# Update data structures due to changing λ. 
__mm_update_lambda__(::MMSVD, problem::BinarySVMProblem, λ, extras) = update_diagonal(problem, λ, extras)

function update_diagonal(problem::BinarySVMProblem, λ, extras)
    @unpack s, Ψ = extras
    n, _, _ = probdims(problem)
    T = floattype(problem)
    a², b² = convert(T, 1/n), convert(T, λ)

    # Update the diagonal matrix Ψ = (a² Σ²) / (a² Σ² + b² I).
    __update_diagonal__(Ψ.diag, s, a², b²)

    return nothing
end

function __update_diagonal__(diag, s, a², b²)
    for i in eachindex(diag)
        sᵢ² = s[i]^2
        diag[i] = a² * sᵢ² / (a² * sᵢ² + b²)
    end
end

# function update_matrices(problem::BinarySVMProblem, λ, extras)
#     @unpack Ψ, V, VΨVt = extras

#     update_diagonal(problem, λ, extras)
#     VΨVt .= V * Ψ * V'

#     return nothing
# end

# Apply one update.
function __mm_iterate__(::MMSVD, problem::BinarySVMProblem, ρ, k, extras)
    @unpack intercept, coeff, proj = problem
    @unpack buffer, projection = extras
    @unpack z, Ψ, U, s, V = extras
    β, pₘ = coeff, proj
    Σ = Diagonal(s)
    T = floattype(problem)

    # need to compute z via residuals...
    apply_projection(projection, problem, k)
    __evaluate_residuals__(problem, extras, true, false)

    # Update parameters: β = P(βₘ) + V * Ψ * (Σ⁻¹Uᵀzₘ - VᵀP(βₘ)) 
    mul!(buffer, U', z)
    ldiv!(Σ, buffer)
    mul!(buffer, V', pₘ, -one(T), one(T))
    lmul!(Ψ, buffer)
    mul!(β, V, buffer)
    axpy!(one(T), pₘ, β)

    return nothing
end

# function __mm_iterate__(::MMSVD, problem::BinarySVMProblem, ρ, k, extras)
#     @unpack coeff, proj = problem
#     @unpack z, VΨVt, projection = extras
#     β, pₘ = coeff, proj
#     X = get_design_matrix(problem)
#     T = floattype(problem)
#     n, _, _ = probdims(problem)

#     # need to compute Z via residuals...
#     apply_projection(projection, problem, k)
#     __evaluate_residuals__(problem, extras, true, false)

#     # Update parameters: β = 1/ρ * (I - VΨVᵀ) * (1/n Xᵀ zₘ + ρ P(βₘ)) 
#     a, b = convert(T, 1/n), convert(T, ρ)
#     mul!(pₘ, X', z, a, ρ)
#     copyto!(β, pₘ)
#     mul!(β, VΨVt, pₘ, -1/b, 1/b)

#     return nothing
# end

# Apply one update in reguarlized problem.
function __reg_iterate__(::MMSVD, problem::BinarySVMProblem, λ, extras)
    @unpack intercept, coeff = problem
    @unpack buffer = extras
    @unpack z, Ψ, U, s, V = extras
    β = coeff
    Σ = Diagonal(s)

    # need to compute z via residuals...
    __evaluate_residuals__(problem, extras, true, false)

    # Update parameters: B = V * Ψ * Σ⁻¹ * Uᵀ * Z
    mul!(buffer, U', z)
    ldiv!(Σ, buffer)
    lmul!(Ψ, buffer)
    mul!(β, V, buffer)

    return nothing
end
