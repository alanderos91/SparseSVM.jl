module SparseSVM
using DataFrames: copy, copyto!
using DataDeps, CSV, DataFrames, CodecZlib
using MLDataUtils
using KernelFunctions, LinearAlgebra, Random

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

const DATA_DIR = joinpath(@__DIR__, "data")

"""
List available datasets in SparseSVM.
"""
list_datasets() = map(x -> splitext(x)[1], readdir(DATA_DIR))

function __init__()
  for dataset in list_datasets()
    include(joinpath(DATA_DIR, dataset * ".jl"))
  end
end

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
function eval_objective(Ab::AbstractVector, y, b, p, rho, k, intercept)
  T = eltype(Ab)
  m, n = length(y), length(b)
  obj = zero(T)
  for i in eachindex(b) # rho * |P(b) - b|^2 contribution; can reduce to BLAS.axpby!
    obj = obj + (p[i] - b[i])^2
  end
  obj = rho * inv(n -k + 1) * obj
  for i in eachindex(y) # |z - A*b|^2 contribution
    obj = obj + max(one(T) - y[i] * Ab[i], zero(T))^2 / m
  end
  return obj / 2
end

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
  y_bar = sum(y) / m
  for j in idx
    @views x = X[:, j]
    x_bar = sum(x) / m
    A = zero(eltype(b))
    B = zero(eltype(b))
    for i in 1:m
      A += (x[i] - x_bar) * (y[i] - y_bar)
      B += (x[i] - x_bar) ^2
    end
    b[j] = A / B
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
    b_old = zeros(size(A, 2))
    z = zeros(T, m)
    Ab = A * b
    project_sparsity_set!(pvec, idx, k)
    old = eval_objective(Ab, y, b, p, rho, k, intercept)
    
    # initialize worker array for Nesterov acceleration
    copyto!(b_old, b)
    
    for iter = 1:ninner
        iters = iters + 1
        @. grad = rho / (n-k+1) * (b - p) # penalty contribution
        for i = 1:m
            z[i] = - y[i] * max(one(T) - y[i] * Ab[i], zero(T))
        end
        BLAS.gemv!('T', 1/m, A, z, 1.0, grad)
        s = norm(grad)^2 # optimal step size
        mul!(Ab, A, grad)
        t = 1/m * norm(Ab)^2
        s = s / (t + rho/(n-k+1) * norm(grad)^2 + eps()) # add small bias to prevent NaN
        @. increment = - s * grad
        # counter = 0
        for step = 0:3 # step halving
            @. b = b + increment
            copyto!(p, b)
            project_sparsity_set!(pvec, idx, k)
            mul!(Ab, A, b)
            obj = eval_objective(Ab, y, b, p, rho, k, intercept)
            if obj < old
                break
            else
                @. b = b - increment
                @. increment = 0.5 * increment
                # counter += 1
            end
        end
        
        has_converged = abs(old - obj) < tol * (old + one(T)) # norm(grad) < tol
        
        if  has_converged
            break
        else
            # Update objective.
            old = obj
            
            # Nesterov acceleration.
            γ = (iter - 1) / (iter + 2 - 1)
            @inbounds for i in eachindex(b)
                xi = b[i]
                yi = b_old[i]
                zi = xi + γ * (xi - yi)
                b_old[i] = xi
                b[i] = zi
            end
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
    z = extras.z
    b_old = extras.b_old
    
    # initialize
    Ab = A * b
    
    # initialize projection
    project_sparsity_set!(pvec, idx, k)
    
    # initialize objective
    old = eval_objective(Ab, y, b, p, rho, k, intercept)
    
    # initialize worker array for Nesterov acceleration
    copyto!(b_old, b)
    
    for iter = 1:ninner
        iters = iters + 1
        
        # Update estimates
        compute_z!(z, Ab, y)
        update_beta!(b, U, s, V, z, p, rho / (n-k+1))
        
        # update objective
        copyto!(p, b)
        project_sparsity_set!(pvec, idx, k)
        mul!(Ab, A, b)
        obj = eval_objective(Ab, y, b, p, rho, k, intercept)
        
        # check convergence
        has_converged = abs(old - obj) < tol * (old + one(T)) # norm(grad) < tol
        
        if  has_converged
            break
        else
            old = obj
            
            # Nesterov acceleration
            γ = (iter - 1) / (iter + 2 - 1)
            @inbounds for i in eachindex(b)
                xi = b[i]
                yi = b_old[i]
                zi = xi + γ * (xi - yi)
                b_old[i] = xi
                b[i] = zi
            end
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

function accumulate_beta!(b, u, s, v, z, p, rho)
  m = length(b)
  uz = dot(u, z)
  vp = dot(v, p)
  
  c1 = s/m / (s^2/m + rho)
  c2 = s^2/m / (s^2/m + rho)
  c3 = c1 * uz - c2 * vp

  axpy!(c3, v, b)
end

function update_beta!(b, U, s, V, z, p, rho)
  copyto!(b, p)
  @views for i in eachindex(s)
      accumulate_beta!(b, U[:,i], s[i], V[:,i], z, p, rho)
  end
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

  extras = (U=U, s=s, V=V, z=z, b_old=b_old)
  return extras
end

export sparse_direct, sparse_direct!, sparse_steepest, sparse_steepest!
##### END MM ALGORITHMS #####

##### CLASSIFICATION #####
include("classifier.jl")

export MultiClassStrategy, OVO, OVR
export SVMBatch, BinaryClassifier, MultiClassifier, trainMM, trainMM!
##### END CLASSIFICATION #####

end # end module