module SparseSVM
using LinearAlgebra, Random

##### OBJECTIVE #####
function eval_objective(Xb::AbstractVector, y, b, p, rho)
  T = eltype(Xb)
  m, n = length(y), length(b)
  obj = zero(T)
  for i in 1:n # rho * |P(b) - b|^2 contribution; can reduce to BLAS.axpby!
    obj = obj + (p[i] - b[i])^2
  end
  obj = rho / n * obj
  for i in 1:m # |z - X*b|^2 contribution
    obj = obj + max(one(T) - y[i] * Xb[i], zero(T))^2 / m
  end
  return obj
end

eval_objective(X::AbstractMatrix, y, b, p, rho) = eval_objective(X*b, y, b, p, rho)
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
##### END PROJECTIONS #####

##### MAIN DRIVER #####
"""
Solve the distance-penalized SVM using algorithm `f` by slowing annealing the
distance penalty.

**Note**: The function `f` must have signature `f(b, X, y, tol, k)` where `b`
represents the model parameters.

### Arguments

- `f`: A function implementing an optimization algorithm.
- `X`: The design matrix with the last column containing `1`s.
- `y`: The class labels which must be `1` or `-1`.
- `tol`: Relative tolerance for objective.
- `k`: Desired number of nonzero features.

### Options

- `mult`: Multiplier to update rho; i.e. `rho = mult * rho`.
- `nouter`: Number of subproblems to solve.
- `rho_init`: Initial value for the penalty coefficient.
"""
function annealing(f, X, y, tol, k; kwargs...)
  T = eltype(X)
  b = randn(T, size(X, 2))
  annealing!(f, b, X, y, tol, k; kwargs...)
end

function annealing!(f, b, X, y, tol, k; init::Bool=true, mult=1.5, nouter=10, rho_init=1.0)
  init && randn!(b)
  rho = rho_init
  T = eltype(X) # should be more careful here to make sure BLAS works correctly
  (obj, old, iters) = (zero(T), zero(T), 0)

  # check if svd(X) is needed
  if f isa typeof(sparse_direct!)
    extras = alloc_svd_and_extras(X)
  else
    extras = nothing
  end

  for n in 1:nouter
    print(n,"  ")
    # solve problem for fixed rho
    _, cur_iters, obj = @time f(b, X, y, rho, tol, k, extras)

    # if abs(old - obj) < tol * (old + 1)
    #   break
    # else
    #   old = obj
    # end

    iters += cur_iters

    # update according to annealing schedule
    rho = mult * rho
  end
  p = project_sparsity_set!(copy(b), collect(1:length(b)), k)
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
function sparse_steepest(X::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int) where T <: Float64
  b = randn(T, size(X, 2))
  sparse_steepest!(b, X, y, rho, tol, k, nothing)
end

"""
Solve the distance-penalized SVM with fixed `rho` via steepest descent.

This version updates the coefficient vector `b` and can be used with `annealing`.
"""
function sparse_steepest!(b::Vector{T}, X::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, extras) where T <: Float64
#
  (m, n) = size(X)
  (obj, old, iters) = (zero(T), zero(T), 0)
  (grad, increment) = (zeros(T, n), zeros(T, n))
  (p, idx) = (copy(b), collect(1:n)) # for projection
  a = zeros(T, m)
  Xb = X * b
  project_sparsity_set!(p, idx, k)
  old = eval_objective(Xb, y, b, p, rho)
  for iter = 1:1000
    iters = iters + 1
    @. grad = rho * (b - p) # penalty contribution
    for i = 1:m
      a[i] = - y[i] * max(one(T) - y[i] * Xb[i], zero(T))
    end
    BLAS.gemv!('T', 1.0, X, a, 1.0, grad)
    s = norm(grad)^2 # optimal step size
    mul!(Xb, X, grad)
    t = norm(Xb)^2
    s = s / (t + rho * norm(grad)^2 + eps()) # add small bias to prevent NaN
    @. increment = - s * grad
    # counter = 0
    for step = 0:3 # step halving
      @. b = b + increment
      copyto!(p, b)
      project_sparsity_set!(p, idx, k)
      mul!(Xb, X, b)
      obj = eval_objective(Xb, y, b, p, rho)
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
function sparse_direct(X, y, rho, tol, k)
  b = randn(T, size(X, 2))
  sparse_direct!(b, X, y, rho, tol, k, extras)
end

"""
Solve the distance-penalized SVM with fixed `rho` via normal equations.

This version updates the coefficient vector `b` and can be used with `annealing`.
"""
function sparse_direct!(b::Vector{T}, X::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, extras) where T <: Float64
  (m, n) = size(X)
  (obj, old, iters) = (zero(T), zero(T), 0)
  (p, idx) = (copy(b), collect(1:n)) # for projection

  # unpack
  U, s, V = extras.U, extras.s, extras.V
  z, d1, d2 = extras.z, extras.d1, extras.d2
  M1, M2, Mtmp = extras.M1, extras.M2, extras.Mtmp

  # Constants
  for i in eachindex(s)
    denominator = inv(s[i]^2 + rho)
    d1[i] = s[i] * denominator
    d2[i] = rho * denominator
  end
  # M1 = V * Diagonal(d1) * U'
  # M2 = V * Diagonal(d2) * V'
  mul!(Mtmp, V, Diagonal(d1))
  mul!(M1, Mtmp, U')
  mul!(Mtmp, V, Diagonal(d2))
  mul!(M2, Mtmp, V')

  Xb = X * b

  # initialize projection
  project_sparsity_set!(p, idx, k)

  ### initialize objective ###
  old = eval_objective(Xb, y, b, p, rho)

  for iter = 1:1000
    iters = iters + 1
    
    # update estimates
    @inbounds for i in eachindex(z)
      c = y[i]*Xb[i]
      if c < 1
        z[i] = y[i]
      else
        z[i] = c
      end
    end
    mul!(b, M2, p)
    mul!(b, M1, z, true, true)

    # update objective
    copyto!(p, b)
    project_sparsity_set!(p, idx, k)
    mul!(Xb, X, b)
    obj = eval_objective(Xb, y, b, p, rho)
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
Compute `svd(X)` and allocate additional arrays used by `sparse_direct!`.
"""
function alloc_svd_and_extras(X)
  T = eltype(X)
  Xsvd = svd(X)
  U = Xsvd.U # left singular vectors
  s = Xsvd.S # singular values
  V = Xsvd.V # right singular vectors

  (m, n) = size(X)
  z = zeros(T, m)          # stores yᵢ * (Xb)ᵢ or yᵢ
  d1 = zeros(T, length(s)) # for Diagonal matrix sᵢ / (sᵢ² + rho)
  d2 = zeros(T, length(s)) # for Diagonal matrix 1 / (sᵢ² + rho)
  M1 = zeros(T, n,  m)     # stores V * D₁ * U'
  M2 = zeros(T, n, n)      # stores V * D₂ * V'
  Mtmp = zeros(T, n, n)    # for computing M₁ and M₂

  extras = (U=U, s=s, V=V, z=z, d1=d1, d2=d2, M1=M1, M2=M2, Mtmp=Mtmp)
end

export sparse_direct, sparse_direct!, sparse_steepest, sparse_steepest!
##### END MM ALGORITHMS #####

##### CLASSIFICATION #####
include("classifier.jl")

export BinarySVMClassifier, MultiSVMClassifier, classify, trainMM
##### END CLASSIFICATION #####

end # end module