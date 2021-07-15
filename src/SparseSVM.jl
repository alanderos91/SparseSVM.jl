module SparseSVM
using DataFrames: copy, copyto!
using DataDeps, CSV, DataFrames, CodecZlib
using MLDataUtils
using KernelFunctions, LinearAlgebra, Random, Statistics

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

const DATA_DIR = joinpath(@__DIR__, "data")

"""
`list_datasets()`

List available datasets in SparseSVM.
"""
list_datasets() = map(x -> splitext(x)[1], readdir(DATA_DIR))

function __init__()
    for dataset in list_datasets()
        include(joinpath(DATA_DIR, dataset * ".jl"))
    end
end

"""
`dataset(str)`

Load a dataset named `str`, if available. Returns data as a `DataFrame` where
the first column contains labels/targets and the remaining columns correspond to
distinct features.
"""
function dataset(str)
    # Locate dataset file.
    dataset_path = @datadep_str str
    file = readdir(dataset_path)
    index = findfirst(x -> occursin("data.", x), file)
    if index isa Int
        dataset_file = joinpath(dataset_path, file[index])
    else # is this unreachable?
        error("Failed to locate a data.* file in $(dataset_path)")
    end
    
    # Read dataset file as a DataFrame.
    df = if splitext(dataset_file)[2] == ".csv"
        CSV.read(dataset_file, DataFrame)
    else # assume .csv.gz
        open(GzipDecompressorStream, dataset_file, "r") do stream
            CSV.read(stream, DataFrame)
        end
    end
    return df
end

function process_dataset(path::AbstractString; header=false, missingstrings="", kwargs...)
    input_df = CSV.read(path, DataFrame, header=header, missingstrings=missingstrings)
    process_dataset(input_df; kwargs...)
    rm(path)
end

function process_dataset(input_df::DataFrame;
    target_index=-1,
    feature_indices=1:0,
    ext=".csv")
    # Build output DataFrame.
    output_df = DataFrame()
    output_df.target = input_df[!, target_index]
    output_df = hcat(output_df, input_df[!, feature_indices], makeunique=true)
    output_cols = [ :target; [Symbol("x", n) for n in eachindex(feature_indices)] ]
    rename!(output_df, output_cols)
    dropmissing!(output_df)
    
    # Write to disk.
    output_path = "data" * ext
    if ext == ".csv"
        CSV.write(output_path, output_df, delim=',', writeheader=true)
    elseif ext == ".csv.gz"
        open(GzipCompressorStream, output_path, "w") do stream
            CSV.write(stream, output_df, delim=",", writeheader=true)
        end
    else
        error("Unknown file extension option '$(ext)'")
    end
end

##### OBJECTIVE #####
"""
`eval_objective(Ab::AbstractVector, y, b, p, rho, k, intercept)`

Evaluate ``\\frac{1}{2} \\left[ \\frac{1}{m} \\|y - A \\beta\\|^{2} + \\frac{\\rho}{n-k+1} \\dist(b, S_{k})^{2} \\right]``.
Assumes `Ab = A * b`. 
"""
function eval_objective(Ab::AbstractVector, y, b, p, rho, k, intercept)
    T = eltype(Ab)
    m, n = length(y), length(b)
    obj = zero(T)
    for i in eachindex(b) # rho * |P(b) - b|^2 contribution; can reduce to BLAS.axpby!
        obj = obj + (p[i] - b[i])^2
    end
    obj = rho / (n - k + 1) * obj
    for i in eachindex(y) # |z - A*b|^2 contribution
        obj = obj + max(one(T) - y[i] * Ab[i], zero(T))^2 / m
    end
    return obj / 2
end

"""
`eval_objective(A::AbstractMatrix, y, b, p, rho, k, intercept)`

Evaluate ``\\frac{1}{2} \\left[ \\frac{1}{m} \\|y - A \\beta\\|^{2} + \\frac{\\rho}{n-k+1} \\dist(b, S_{k})^{2} \\right]``.
Assumes `A` is the design matrix.
"""
eval_objective(A::AbstractMatrix, y, b, p, rho, k, intercept) = eval_objective(A*b, y, b, p, rho, k, intercept)
##### END OBJECTIVE #####

##### PROJECTIONS ######
"""
Project `x` onto sparsity set with `k` non-zero elements.
Assumes `idx` enters as a vector of indices into `x`.
"""
function project_sparsity_set!(x, idx, k)
    # do nothing if k > length(x)
    if k ≥ length(x) return x end
    
    # fill with zeros if k ≤ 0
    if k ≤ 0 return fill!(x, 0) end
    
    # find the spliting element
    pivot = search_partialsort!(idx, x, k)
    
    # apply the projection
    kcount = 0
    @inbounds for i in eachindex(x)
        if abs(x[i]) <= abs(pivot) || kcount ≥ k
            x[i] = 0
        else
            kcount += 1
        end
    end
    
    return x
end

"""
Search `x` for the pivot that splits the vector into the `k`-largest elements in magnitude.

The search preserves signs and returns `x[k]` after partially sorting `x`.
"""
function search_partialsort!(idx, x, k)
    #
    # Based on https://github.com/JuliaLang/julia/blob/788b2c77c10c2160f4794a4d4b6b81a95a90940c/base/sort.jl#L863
    # This eliminates a mysterious allocation of ~48 bytes per call for
    #   sortperm!(idx, x, alg=algorithm, lt=isless, by=abs, rev=true, initialized=false)
    # where algorithm = PartialQuickSort(lo:hi)
    # Savings are small in terms of performance but add up for CV code.
    #
    lo = k
    hi = k+1
    
    # Order arguments
    lt  = isless
    by  = abs
    rev = true
    o = Base.Order.Forward
    order = Base.Order.Perm(Base.Sort.ord(lt, by, rev, o), x)
    
    # Initialize the idx array; algorithm relies on idx[i] = i
    @inbounds for i in eachindex(idx)
        idx[i] = i
    end
    
    # sort!(idx, lo, hi, PartialQuickSort(k), order)
    Base.Sort.Float.fpsort!(idx, PartialQuickSort(lo:hi), order)
    
    return x[idx[k+1]]
end

"""
Returns a tuple `(pvec, idx)` where `pvec` is a slice into `p` when `intercept == true`
or `p` otherwise. The vector `idx` is a collection of indices into `pvec`.
"""
function get_model_coefficients(p, intercept)
    n = length(p)
    if intercept
        (pvec, idx) = (view(p, 1:n-1), collect(1:n-1))
    else
        (pvec, idx) = (p, collect(1:n))
    end
    return (pvec, idx)
end

##### END PROJECTIONS #####

##### MAIN DRIVER #####
function _init_weights_!(b, X, y, intercept)
    m, n = size(X)
    idx = ifelse(intercept, 1:n-1, 1:n)
    y_bar = mean(y)
    for j in idx
        @views x = X[:, j]
        x_bar = mean(x)
        A = zero(eltype(b))
        B = zero(eltype(b))
        for i in 1:m
            A += (x[i] - x_bar) * (y[i] - y_bar)
            B += (x[i] - x_bar) ^2
        end
        b[j] = B == 0 ? 1e-6*rand() : A / B
    end
    if intercept
        b[end] = y_bar
    end
    return b
end

"""
Solve the distance-penalized SVM using algorithm `f` by slowly annealing the
distance penalty.

**Note**: The function `f` must have signature `f(b, A, y, tol, k)` where `b`
represents the model's parameters.

### Arguments

- `f`: A function implementing an optimization algorithm.
- `A`: The design matrix.
- `y`: The class labels which must be `1` or `-1`.
- `tol`: Relative tolerance for objective.
- `k`: Desired number of nonzero features.
- `intercept`: A `Bool` indicating whether `A` contains a column of 1s for the intercept.

### Options

- `mult`: Multiplier to update rho; i.e. `rho = mult * rho`.
- `nouter`: Number of subproblems to solve.
- `rho_init`: Initial value for the penalty coefficient.
"""
function annealing(f, A, y, tol, k, intercept; kwargs...)
    T = eltype(A)
    b = randn(T, size(A, 2))
    annealing!(f, b, A, y, tol, k, intercept; kwargs...)
end

function annealing!(f, b, A, y, tol, k, intercept;
    init::Bool=true,
    mult::Real=1.5,
    ninner::Int=1000,
    nouter::Int=10,
    rho_init::Real=1.0,
    fullsvd::Union{Nothing,SVD}=nothing,
    verbose::Bool=false,
    )
    #
    init && _init_weights_!(b, A, y, intercept)
    rho = rho_init
    T = eltype(A) # should be more careful here to make sure BLAS works correctly
    (obj, old, iters) = (zero(T), zero(T), 0)
    m, n = size(A)
    
    # check if svd(A) is needed
    if f isa typeof(sparse_direct!)
        extras = alloc_svd_and_extras(A, fullsvd=fullsvd)
    else
        extras = nothing
    end
    
    for n in 1:nouter
        # solve problem for fixed rho
        if verbose
            print(n,"  ")
            _, cur_iters, obj = @time f(b, A, y, rho, tol, k, intercept, extras, ninner=ninner, verbose=true)
        else
            _, cur_iters, obj = f(b, A, y, rho, tol, k, intercept, extras, ninner=ninner, verbose=false)
        end
        
        # if abs(old - obj) < tol * (old + 1)
        #   break
        # else
        #   old = obj
        # end
        
        iters += cur_iters
        
        # update according to annealing schedule
        rho = mult * rho
    end
    
    if b isa Vector
        p = copy(b)
        pvec, idx = get_model_coefficients(p, intercept)
        project_sparsity_set!(pvec, idx, k)
        dist = norm(p - b) / sqrt(n - k + 1)
        copyto!(b, p)
    else
        @views p = copy(b[:,1])
        pvec, idx = get_model_coefficients(p, intercept)
        dist = zero(eltype(p))
        for j in axes(b, 2)
            @views b_j = b[:,j]
            copyto!(p, b_j)
            project_sparsity_set!(pvec, idx, k)
            dist = dist + norm(p - b_j) / sqrt(n - k + 1)
            copyto!(b_j, p)
        end
    end
    
    if verbose
        print("\niters = ", iters)
        print("\ndist  = ", dist)
        print("\nobj   = ", obj)
        print("\nTotal Time: ")
    end
    return iters, obj, dist
end

export annealing, annealing!
##### END MAIN DRIVER #####

##### MM ALGORITHMS #####
"""
Solve the distance-penalized SVM with fixed `rho` via steepest descent.

This version allocates the coefficient vector `b`.
"""
function sparse_steepest(A::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool; kwargs...) where T <: AbstractFloat
    b = randn(T, size(A, 2))
    sparse_steepest!(b, A, y, rho, tol, k, intercept, nothing; kwargs...)
end

"""
Solve the distance-penalized SVM with fixed `rho` via steepest descent.

This version updates the coefficient vector `b` and can be used with `annealing`.
"""
function sparse_steepest!(b::Vector{T}, A::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool, extras; ninner::Int=1000, verbose::Bool=false) where T <: AbstractFloat
    #
    (m, n) = size(A)
    (obj, old, iters) = (zero(T), zero(T), 0)
    (grad, increment) = (zeros(T, n), zeros(T, n))
    p = copy(b) # for projection
    (pvec, idx) = get_model_coefficients(p, intercept)
    z = zeros(T, m)
    Ab = A * b
    project_sparsity_set!(pvec, idx, k)
    old = eval_objective(Ab, y, b, p, rho, k, intercept)
    
    # initialize worker array for Nesterov acceleration
    b_old = copy(b)
    nest_iter = 1
    for iter = 1:ninner
        iters = iters + 1
        @. grad = (b - p) # penalty contribution
        for i = 1:m
            # z[i] = - y[i] * max(one(T) - y[i] * Ab[i], zero(T))
            c = y[i] * Ab[i]
            z[i] = ifelse(c ≤ 1, Ab[i] - y[i], 0.0)
        end
        mul!(grad, A', z, 1/m, rho / (n-k+1))
        s = dot(grad, grad) # optimal step size
        mul!(Ab, A, grad)
        t = dot(Ab, Ab)
        s = s / (1/m*t + rho/(n-k+1) * s + eps()) # add small bias to prevent NaN
        @. increment = -s * grad
        
        @. b = b + increment
        copyto!(p, b)
        project_sparsity_set!(pvec, idx, k)
        mul!(Ab, A, b)
        obj = eval_objective(Ab, y, b, p, rho, k, intercept)
        # counter = 0
        
        # Apply step halving.
        for _ in 0:3
            if obj < old
                break
            else
                # backtrack
                @. b = b - increment
                @. increment = 0.5 * increment

                # move forward
                @. b = b + increment
                copyto!(p, b)
                project_sparsity_set!(pvec, idx, k)
                mul!(Ab, A, b)
                obj = eval_objective(Ab, y, b, p, rho, k, intercept)
                # counter += 1
            end
        end
        
        has_converged = abs(old - obj) < tol * (old + one(T)) # norm(grad) < tol
        
        if  has_converged
            break
        else
            if obj > old
                copyto!(b_old, b)
                nest_iter = 1
            else            
                # Nesterov acceleration.
                γ = (nest_iter - 1) / (nest_iter + 2 - 1)
                @inbounds for i in eachindex(b)
                    xi = b[i]
                    yi = b_old[i]
                    zi = xi + γ * (xi - yi)
                    b_old[i] = xi
                    b[i] = zi
                end
                nest_iter += 1
            end
            old = obj
            copyto!(p, b)
            project_sparsity_set!(pvec, idx, k)
        end
        
    end
    verbose && print(iters,"  ",obj)
    return b, iters, obj
end

"""
Solve the distance-penalized SVM with fixed `rho` via normal equations.

This version allocates the coefficient vector `b`.
"""
function sparse_direct(A::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool; kwargs...) where T <: AbstractFloat
    b = randn(eltype(A), size(A, 2))
    extras = alloc_svd_and_extras(A)
    sparse_direct!(b, A, y, rho, tol, k, intercept, extras; kwargs...)
end

"""
Solve the distance-penalized SVM with fixed `rho` via normal equations.

This version updates the coefficient vector `b` and can be used with `annealing`.
"""
function sparse_direct!(b::Vector{T}, A::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool, extras; ninner::Int=1000, verbose::Bool=false) where T <: AbstractFloat
    (obj, old, iters) = (zero(T), zero(T), 0)
    p = copy(b) # for projection
    (pvec, idx) = get_model_coefficients(p, intercept)
    m, n = size(A)
    
    # unpack
    U, s, V = extras.U, extras.s, extras.V
    z, buffer = extras.z, extras.buffer
    b_old = extras.b_old
    
    # initialize
    Ab = buffer[1]; mul!(Ab, A, b)
    d2 = buffer[2]
    for i in eachindex(s)
        d2[i] = (s[i] / m) / (s[i]^2 / m + rho)
    end
    btmp = buffer[3]
    D1 = Diagonal(s)
    D2 = Diagonal(d2)
    
    # initialize projection
    project_sparsity_set!(pvec, idx, k)
    
    # initialize objective
    old = eval_objective(Ab, y, b, p, rho, k, intercept)
    
    # initialize worker array for Nesterov acceleration
    copyto!(b_old, b)
    
    nest_iter = 1
    for iter = 1:ninner
        iters = iters + 1
        
        # Update estimates
        compute_z!(z, Ab, y)
        update_beta!(b, btmp, U, V, D1, D2, z, p) # b == p after this update
        
        # update objective
        project_sparsity_set!(pvec, idx, k)
        mul!(Ab, A, b)
        obj = eval_objective(Ab, y, b, p, rho, k, intercept)
        
        # check convergence
        has_converged = abs(old - obj) < tol * (old + one(T)) # norm(grad) < tol
        
        if  has_converged
            break
        else
            if obj > old
                copyto!(b_old, b)
                nest_iter = 1
            else
                # Nesterov acceleration.
                γ = (nest_iter - 1) / (nest_iter + 2 - 1)
                @inbounds for i in eachindex(b)
                    xi = b[i]
                    yi = b_old[i]
                    zi = xi + γ * (xi - yi)
                    b_old[i] = xi
                    b[i] = zi
                end
                nest_iter += 1
            end
            old = obj
            copyto!(p, b)
            project_sparsity_set!(pvec, idx, k)
        end  
    end
    verbose && print(iters,"  ",obj)
    return b, iters, obj
end

function compute_z!(z, Ab, y)
    @inbounds for i in eachindex(z)
        c = y[i]*Ab[i]
        if c ≤ 1
            z[i] = y[i]
        else
            z[i] = Ab[i]
        end
    end
end

function update_beta!(b, btmp, U, V, D1, D2, z, p)
    r = size(D1, 1)
    U_block = view(U, :, 1:r)
    V_block = view(V, :, 1:r)

    # btmp = Σ V' p
    mul!(btmp, V_block', p)
    lmul!(D1, btmp)

    # btmp = U' z - btmp
    mul!(btmp, U_block', z, 1.0, -1.0)

    # btmp = [ (1/m Σ² + ρI)⁻¹ 1/m Σ ] btmp
    lmul!(D2, btmp)

    # b = p = p + V'b
    mul!(p, V_block, btmp, 1.0, 1.0)
    copyto!(b, p)
end

"""
Compute `svd(X)` and allocate additional arrays used by `sparse_direct!`.
"""
function alloc_svd_and_extras(A; fullsvd::Union{Nothing,SVD}=nothing)
    T = eltype(A)
    if fullsvd isa SVD
        Asvd = fullsvd
    else
        Asvd = svd(A)
    end
    U = Asvd.U # left singular vectors
    s = Asvd.S # singular values
    V = Asvd.V # right singular vectors
    z = zeros(T, size(A, 1)) # stores yᵢ * (Xb)ᵢ or yᵢ
    b_old = zeros(size(A, 2))

    buffer = Vector{Vector{T}}(undef, 3)
    buffer[1] = similar(A, axes(A, 1))  # for A * b
    buffer[2] = similar(s)              # for diagonal a^2 Σ / (a^2 Σ^2 + b^2 I)
    buffer[3] = similar(s)              # for updating β
    
    extras = (U=U, s=s, V=V, z=z, b_old=b_old, buffer=buffer)
end

export sparse_direct, sparse_direct!, sparse_steepest, sparse_steepest!
##### END MM ALGORITHMS #####

##### CLASSIFICATION #####
include("classifier.jl")

export MultiClassStrategy, OVO, OVR
export SVMBatch, BinaryClassifier, MultiClassifier, trainMM, trainMM!
export get_support_vecs, count_support_vecs
##### END CLASSIFICATION #####

end # end module