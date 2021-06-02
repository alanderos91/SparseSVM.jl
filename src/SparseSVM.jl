module SparseSVM
using KernelFunctions, LinearAlgebra, Random

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

function eval_kernel_objective(K, y, a, p, rho)
  T = eltype(K)
  n = size(K, 1)
  obj = zero(T)
  dist = zero(T)
  for i in 1:n
    s = zero(T)
    for j in 1:n
      s = s + K[i,j] * y[j] * a[j]
    end
    obj = obj + max(0, 1 - y[i] * s)
    dist = dist + (p[i] - a[i])^2
  end
  obj = obj / n + rho / n * dist
  return obj
end

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
function annealing(f, X, y, tol, k, intercept::Bool; kwargs...)
  T = eltype(X)
  b = randn(T, size(X, 2))
  annealing!(f, b, X, y, tol, k, intercept; kwargs...)
end

function annealing!(f, b, X, y, tol, k, intercept::Bool;
  init::Bool=true,
  mult::Real=1.5,
  ninner::Int=1000,
  nouter::Int=10,
  rho_init::Real=1.0,
  )
  #
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
    _, cur_iters, obj = @time f(b, X, y, rho, tol, k, intercept, extras, ninner=ninner)

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
  n = length(p)

  if intercept
    (pvec, idx) = (view(p, 1:n-1), collect(1:n-1))
  else
    (pvec, idx) = (p, collect(1:n))
  end

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
function sparse_steepest(X::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool; kwargs...) where T <: Float64
  b = randn(T, size(X, 2))
  sparse_steepest!(b, X, y, rho, tol, k, intercept, nothing; kwargs...)
end

"""
Solve the distance-penalized SVM with fixed `rho` via steepest descent.

This version updates the coefficient vector `b` and can be used with `annealing`.
"""
function sparse_steepest!(b::Vector{T}, X::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool, extras; ninner::Int=1000) where T <: Float64
#
  (m, n) = size(X)
  (obj, old, iters) = (zero(T), zero(T), 0)
  (grad, increment) = (zeros(T, n), zeros(T, n))
  p = copy(b) # for projection
  if intercept
    (pvec, idx) = (view(p, 1:n-1), collect(1:n-1))
  else
    (pvec, idx) = (p, collect(1:n))
  end
  a = zeros(T, m)
  Xb = X * b
  project_sparsity_set!(pvec, idx, k)
  old = eval_objective(Xb, y, b, p, rho)
  for iter = 1:ninner
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
      project_sparsity_set!(pvec, idx, k)
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
function sparse_direct(X, y, rho, tol, k, intercept::Bool; kwargs...)
  b = randn(eltype(X), size(X, 2))
  extras = alloc_svd_and_extras(X)
  sparse_direct!(b, X, y, rho, tol, k, intercept, extras; kwargs...)
end

"""
Solve the distance-penalized SVM with fixed `rho` via normal equations.

This version updates the coefficient vector `b` and can be used with `annealing`.
"""
function sparse_direct!(b::Vector{T}, X::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool, extras; ninner::Int=1000) where T <: Float64
  (m, n) = size(X)
  (obj, old, iters) = (zero(T), zero(T), 0)
  p = copy(b) # for projection
  if intercept
    (pvec, idx) = (view(p, 1:n-1), collect(1:n-1))
  else
    (pvec, idx) = (p, collect(1:n))
  end

  # unpack
  U, s, V, S = extras.U, extras.s, extras.V, extras.S
  z, d1 = extras.z, extras.d1
  buf1, buf2 = extras.buf1, extras.buf2
  b_old = extras.b_old

  # Constants
  for i in eachindex(s)
    d1[i] = inv(s[i]^2 + rho)
  end

  Xb = X * b

  # initialize projection
  project_sparsity_set!(pvec, idx, k)

  # initialize objective
  old = eval_objective(Xb, y, b, p, rho)

  # initialize worker array for Nesterov acceleration
  copyto!(b_old, b)

  for iter = 1:ninner
    iters = iters + 1
    
    # update estimates
    @inbounds for i in eachindex(z)
      c = y[i]*Xb[i]
      if c < 1
        z[i] = y[i]
      else
        z[i] = Xb[i]
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
    mul!(b, V, buf2, rho, true)

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
    mul!(Xb, X, b)
    obj = eval_objective(Xb, y, b, p, rho)
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
function alloc_svd_and_extras(X)
  T = eltype(X)
  Xsvd = svd(X, full=true)
  U = Xsvd.U # left singular vectors
  s = Xsvd.S # singular values
  V = Xsvd.V # right singular vectors

  (m, n) = size(X)

  if m ≥ n
    S = [Diagonal(s); zeros(m-n, n)]
  else
    S = [Diagonal(s) zeros(m, n-m)]
  end

  z = zeros(T, m)          # stores yᵢ * (Xb)ᵢ or yᵢ
  d1 = zeros(T, length(s)) # for Diagonal matrix 1 / (sᵢ² + rho)
  buf1 = zeros(m)          # for mat-vec multiply
  buf2 = zeros(n)          # for mat-vec multiply

  extras = (U=U, s=s, V=V, S=S, z=z, d1=d1, buf1=buf1, buf2=buf2, b_old=zeros(n))
end

function sparse_sdk!(a::Vector{T}, K::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept, extras; ninner::Int=1000) where T <: Float64
  #
  n = size(K, 1)
  (obj, old, iters) = (zero(T), zero(T), 0)
  (grad, increment) = (zeros(T, n), zeros(T, n))
  Y = Diagonal(y)
  KYa = K * Y * a
  buffer = zeros(T, n)
  (p, idx) = (copy(a), collect(1:n)) # for projection
  project_sparsity_set!(p, idx, k)
  old = eval_kernel_objective(K, y, a, p, rho)
  for iter = 1:ninner
    iters = iters + 1
    @. grad = a - p # penalty contribution
    lmul!(Y, grad)  # scale by Y to exploit Y*Y = I
    for i = 1:n
      buffer[i] = -y[i] * max(0, 1 - KYa[i] * y[i])
    end
    mul!(grad, K, buffer, true, rho) # = K*(KY*α - z) + ρ*Y(α - P(α))
    lmul!(Y, grad)                   # = YK*(KY*α - z) + ρ(α - P(α))
    s = norm(grad)^2 # optimal step size
    mul!(buffer, Y, grad) # = Y*∇g
    mul!(KYa, K, buffer)  # = K*Y*∇g
    t = norm(KYa)^2
    s = s / (t + rho * s + eps()) # add small bias to prevent NaN
    @. increment = - s * grad
    # counter = 0
    for step = 0:3 # step halving
      @. a = a + increment
      copyto!(p, a)
      project_sparsity_set!(p, idx, k)
      mul!(buffer, Y, a)
      mul!(KYa, K, buffer)
      obj = eval_kernel_objective(K, y, a, p, rho)
      if obj < old
        break
      else
        @. a = a - increment
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
  return a, iters, obj
end

function sparse_sdk(K::Matrix{T}, y::Vector{T}, rho::T, tol::T, k::Int, intercept::Bool; kwargs...) where T <: Float64
  a = randn(T, size(X, 2))
  sparse_sdk!(a, K, y, rho, tol, k, intercept, nothing; kwargs...)
end

export sparse_direct, sparse_direct!, sparse_steepest, sparse_steepest!, sparse_sdk, sparse_sdk!
##### END MM ALGORITHMS #####

##### CLASSIFICATION #####
include("classifier.jl")

export SVMBatch, BinaryClassifier, classify, trainMM
##### END CLASSIFICATION #####

end # end module