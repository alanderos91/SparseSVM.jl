module SparseSVM
using MLDataUtils
using KernelFunctions, LinearAlgebra, Random

##### OBJECTIVE #####
function eval_objective(Ab::AbstractVector, y, b, p, rho)
  T = eltype(Ab)
  m, n = length(y), length(b)
  obj = zero(T)
  for i in 1:n # rho * |P(b) - b|^2 contribution; can reduce to BLAS.axpby!
    obj = obj + (p[i] - b[i])^2
  end
  obj = rho / n * obj
  for i in 1:m # |z - A*b|^2 contribution
    obj = obj + max(one(T) - y[i] * Ab[i], zero(T))^2 / m
  end
  return obj
end

eval_objective(A::AbstractMatrix, y, b, p, rho) = eval_objective(A*b, y, b, p, rho)
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
  )
  #
  init && randn!(b)
  rho = rho_init
  T = eltype(A) # should be more careful here to make sure BLAS works correctly
  (obj, old, iters) = (zero(T), zero(T), 0)

  # check if svd(A) is needed
  if f isa typeof(sparse_direct!)
    extras = alloc_svd_and_extras(A, fullsvd=fullsvd)
  else
    extras = nothing
  end

  for n in 1:nouter
    print(n,"  ")
    # solve problem for fixed rho
    _, cur_iters, obj = @time f(b, A, y, rho, tol, k, intercept, extras, ninner=ninner)

    # if abs(old - obj) < tol * (old + 1)
    #   break
    # else
    #   old = obj
    # end

    iters += cur_iters

    # update according to annealing schedule
    rho = mult * rho
  end
  p = copy(b)
  pvec, idx = get_model_coefficients(p, intercept)
  project_sparsity_set!(pvec, idx, k)
  dist = norm(p - b) / sqrt(length(b))
  print("\niters = ", iters)
  print("\ndist  = ", dist)
  print("\nobj   = ", obj)
  print("\nTotal Time")
  copyto!(b, p)
  return b, iters, obj, dist
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
function sparse_steepest!(b::Vector{T}, A::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool, extras; ninner::Int=1000) where T <: AbstractFloat
#
  (m, n) = size(A)
  (obj, old, iters) = (zero(T), zero(T), 0)
  (grad, increment) = (zeros(T, n), zeros(T, n))
  p = copy(b) # for projection
  (pvec, idx) = get_model_coefficients(p, intercept)
  z = zeros(T, m)
  Ab = A * b
  project_sparsity_set!(pvec, idx, k)
  old = eval_objective(Ab, y, b, p, rho)
  for iter = 1:ninner
    iters = iters + 1
    @. grad = rho / (k+intercept) * (b - p) # penalty contribution
    for i = 1:m
      z[i] = - y[i] * max(one(T) - y[i] * Ab[i], zero(T))
    end
    BLAS.gemv!('T', 1/m, A, z, 1.0, grad)
    s = norm(grad)^2 # optimal step size
    mul!(Ab, A, grad)
    t = 1/m * norm(Ab)^2
    s = s / (t + rho/(k+intercept) * norm(grad)^2 + eps()) # add small bias to prevent NaN
    @. increment = - s * grad
    # counter = 0
    for step = 0:3 # step halving
      @. b = b + increment
      copyto!(p, b)
      project_sparsity_set!(pvec, idx, k)
      mul!(Ab, A, b)
      obj = eval_objective(Ab, y, b, p, rho)
      if obj < old
        break
      else
        @. b = b - increment
        @. increment = 0.5 * increment
        # counter += 1
      end
    end
    # println(counter, " step halving iterations")
#     if iter <= 10 || mod(iter, 10) == 0 
#       println("iter = ",iter," obj = ",obj)
#     end
    if  abs(old - obj) < tol * (old + one(T)) # norm(grad) < tol
      break
    else
      old = obj
    end  
  end
  print(iters,"  ",obj)
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
function sparse_direct!(b::Vector{T}, A::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool, extras; ninner::Int=1000) where T <: AbstractFloat
  (m, n) = size(A)
  (obj, old, iters) = (zero(T), zero(T), 0)
  p = copy(b) # for projection
  (pvec, idx) = get_model_coefficients(p, intercept)

  # unpack
  U, s, V, S = extras.U, extras.s, extras.V, extras.S
  z, d1 = extras.z, extras.d1
  buf1, buf2 = extras.buf1, extras.buf2
  b_old = extras.b_old

  # Constants
  for i in eachindex(s)
    d1[i] = inv(s[i]^2/m + rho/(k+intercept))
  end

  Ab = A * b

  # initialize projection
  project_sparsity_set!(pvec, idx, k)

  # initialize objective
  old = eval_objective(Ab, y, b, p, rho)

  # initialize worker array for Nesterov acceleration
  copyto!(b_old, b)

  for iter = 1:ninner
    iters = iters + 1
    
    # update estimates
    @inbounds for i in eachindex(z)
      c = y[i]*Ab[i]
      if c ≤ 1
        z[i] = y[i]
      else
        z[i] = Ab[i]
      end
    end
    # b = V * D * S' * U' * z
    mul!(buf1, U', z)
    mul!(buf2, S', buf1)
    lmul!(Diagonal(d1), buf2)
    mul!(b, V, buf2)

    # b = b + rho * (V * D * V' * p)
    mul!(buf2, V', p)
    lmul!(Diagonal(d1), buf2)
    mul!(b, V, buf2, rho/(k+intercept), 1/m)

    # Nesterov acceleration
    γ = (iter - 1) / (iter + 2 - 1)
    @inbounds for i in eachindex(b)
      xi = b[i]
      yi = b_old[i]
      zi = xi + γ * (xi - yi)
      b_old[i] = xi
      b[i] = zi
    end

    # update objective
    copyto!(p, b)
    project_sparsity_set!(pvec, idx, k)
    mul!(Ab, A, b)
    obj = eval_objective(Ab, y, b, p, rho)
    # if iter <= 10 || mod(iter, 10) == 0 
    #   println("iter = ",iter," obj = ",obj)
    # end
    if  abs(old - obj) < tol * (old + one(T)) # norm(grad) < tol
      break
    else
      old = obj
    end  
  end
  print(iters,"  ",obj)
  return b, iters, obj
end

"""
Compute `svd(X)` and allocate additional arrays used by `sparse_direct!`.
"""
function alloc_svd_and_extras(A; fullsvd::Union{Nothing,SVD}=nothing)
  T = eltype(A)
  if fullsvd isa SVD
    Asvd = fullsvd
  else
    Asvd = svd(A, full=true)
  end
  U = Asvd.U # left singular vectors
  s = Asvd.S # singular values
  V = Asvd.V # right singular vectors

  (m, n) = size(A)

  if m ≥ n
    S = [Diagonal(s); zeros(m-n, n)]
  else
    S = [Diagonal(s) zeros(m, n-m)]
  end

  z = zeros(T, m)             # stores yᵢ * (Xb)ᵢ or yᵢ
  d1 = zeros(T, size(S, 2))   # for Diagonal matrix 1 / (sᵢ² + rho)
  buf1 = zeros(T, size(U, 1)) # for mat-vec multiply
  buf2 = zeros(T, size(V, 2)) # for mat-vec multiply

  extras = (U=U, s=s, V=V, S=S, z=z, d1=d1, buf1=buf1, buf2=buf2, b_old=zeros(n))
  return extras
end

export sparse_direct, sparse_direct!, sparse_steepest, sparse_steepest!
##### END MM ALGORITHMS #####

##### CLASSIFICATION #####
include("classifier.jl")

export SVMBatch, BinaryClassifier, MultiClassifier, trainMM, trainMM!
##### END CLASSIFICATION #####

end # end module