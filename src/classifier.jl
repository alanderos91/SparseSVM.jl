import Base: show

struct SVMBatch{T<:AbstractFloat,KF}
  X::Matrix{T}
  K::Union{Nothing,Matrix{T}}
  y::Vector{T}
  kernelf::KF
end

"""
```
SVMBatch(Xdata, ydata; ftype=nothing, kernelf=nothing)
```

Create a batch of samples `(ydata, Xdata)` for training a SVM.

### Keyword Arguments
- `ftype`: A floating point data type, which must have `AbstractFloat` as a supertype (default=`nothing`).
  By default, the floating point type is inferred from `Xdata`.
- `kernelf`: A kernel function from KernelFunctions.jl used to construct the kernel matrix `K` (default=`nothing`).
"""
function SVMBatch(Xdata, ydata; ftype::Union{Nothing,DataType}=nothing, kernelf::Union{Nothing,Kernel}=nothing)
  # Sanity checks.
  if size(Xdata, 1) != length(ydata)
    throw(DimensionMismatch("Number of rows in `Xdata` $(size(Xdata, 1)) must match length of `ydata` $(length(ydata))."))
  end

  if any(yi -> abs(yi) != 1, ydata)
    throw(DomainError(ydata, "Input labels must have entries -1 or 1."))
  end

  # Use same eltype of Xdata unless caller specifies a format.
  if ftype isa Nothing
    X = Xdata
    y = ydata
  elseif ftype <: AbstractFloat
    X = convert(Matrix{ftype}, Xdata)
    y = convert(Vector{ftype}, ydata)
  else
    error("Argument `ftype` must be a floating point type.")
  end

  # Assume linear case unless caller specifies a kernel function.
  if kernelf isa Nothing
    K = nothing
  else
    K = kernelmatrix(kernelf, X, obsdim=1)
  end

  return SVMBatch(X, K, y, kernelf)
end

function Base.show(io::IO, data::SVMBatch)
  println(io, "SVMBatch{", eltype(data.X), "}")
  println(io, "  ∘ samples:   ", nsamples(data))
  println(io, "  ∘ features:  ", nfeatures(data))
  print(io, "  ∘ kernel:    ", data.kernelf)
end

nsamples(data::SVMBatch) = size(data.X, 1)
nfeatures(data::SVMBatch) = size(data.X, 2)
kernelf(data::SVMBatch) = data.kernelf

function get_design_matrix(data::SVMBatch, intercept::Bool=false)
  # check for nonlinear case
  if kernelf(data) isa Nothing
    A_predictors = data.X
  else
    A_predictors = data.K
  end

  # check for intercept
  if intercept
    A = [A_predictors ones(size(A_predictors, 1))]
  else
    A = A_predictors
  end

  return A
end

abstract type Classifier end

struct BinaryClassifier{T<:AbstractFloat,KF} <: Classifier
  weights::Vector{T}
  intercept::Bool
  data::SVMBatch{T,KF}
end

"""
```
BinaryClassifier(data::SVMBatch; intercept::Bool=false)
```

Create a binary classifier from the given training `data`, given as a `SVMBatch` object.

### Keyword Arguments
- `intercept`: Specifies whether to include an intercept in the model (default=`false`).
"""
function BinaryClassifier(data::SVMBatch{T}; intercept::Bool=false) where T
  if kernelf(data) isa Nothing
    # linear case
    weights = Vector{T}(undef, nfeatures(data) + intercept)
  else
    # nonlinear case
    weights = Vector{T}(undef, nsamples(data) + intercept)
  end

  return BinaryClassifier(weights, intercept, data)
end

function classify(classifier::BinaryClassifier, x)
  weights = classifier.weights
  data = classifier.data
  κ = kernelf(data)
  X = data.X
  y = data.y

  if κ isa Nothing
    # linear case
    if classifier.intercept
      @views β, β₀ = weights[1:end-1], weights[end]
      fx = dot(x, β) + β₀
    else
      β = weights
      fx = dot(x, β)
    end
  else
    # nonlinear case
    fx = zero(T)
    if classifier.intercept
      @views α, α₀ = weights[1:end-1], weights[end]
      for (j, xj) in enumerate(eachrow(X))
        fx = fx + α[j] * y[j] * κ(x, xj)
      end
      fx = fx + α₀
    else
      α = weights
      for (j, xj) in enumerate(eachrow(X))
        fx = fx + α[j] * y[j] * κ(x, xj)
      end
    end
  end

  return sign(fx)
end

function trainMM(classifier::BinaryClassifier, f, tol, k; kwargs...)
  data = classifier.data
  intercept = classifier.intercept
  A = get_design_matrix(data, intercept)
  @time annealing!(f, classifier.weights, A, data.y, tol, k, intercept; kwargs...)
end

# # implements one-against-one approach
# struct MultiSVMClassifier <: Classifier
#   nclasses::Int
#   svm::Vector{BinarySVMClassifier}
#   svm_pair::Vector{Tuple{Int,Int}}
#   class::Vector{String}
#   vote::Vector{Int}
# end

# # linear case
# function MultiSVMClassifier(nfeatures::Int, class)
#   nclasses = length(class)
#   nsvms = binomial(nclasses, 2)
#   svm = Vector{BinarySVMClassifier}(undef, nsvms)
#   svm_pair = Vector{Tuple{Int,Int}}(undef, nsvms)
#   k = 1
#   for j in 1:nclasses, i in j+1:nclasses
#     svm[k] = BinarySVMClassifier(nfeatures, Dict(-1 => string(class[i]), 1 => string(class[j])))
#     svm_pair[k] = (i, j)
#     k += 1
#   end
#   return MultiSVMClassifier(nclasses, svm, svm_pair, string.(class), zeros(Int, nclasses))
# end

# # nonlinear case
# function MultiSVMClassifier(y::Vector, class)
#   nclasses = length(class)
#   nsvms = binomial(nclasses, 2)
#   svm = Vector{BinarySVMClassifier}(undef, nsvms)
#   svm_pair = Vector{Tuple{Int,Int}}(undef, nsvms)
#   k = 1
#   for j in 1:nclasses, i in j+1:nclasses
#     nweights = count(class -> class == i || class == j, y)
#     svm[k] = BinarySVMClassifier(nweights, Dict(-1 => string(class[i]), 1 => string(class[j])))
#     svm_pair[k] = (i, j)
#     k += 1
#   end
#   return MultiSVMClassifier(nclasses, svm, svm_pair, string.(class), zeros(Int, nclasses))
# end

# function classify(classifier::MultiSVMClassifier, x)
#   fill!(classifier.vote, 0)
#   for k in eachindex(classifier.svm)
#     (i, j) = classifier.svm_pair[k]
#     if classify(classifier.svm[k], x) < 0 # vote for class i
#       classifier.vote[i] += 1
#     else # vote for class j
#       classifier.vote[j] += 1
#     end
#   end
#   return argmax(classifier.vote)
# end

# function classify(classifier::MultiSVMClassifier, κ, X, y, x)
#   fill!(classifier.vote, 0)
#   for k in eachindex(classifier.svm)
#     (i, j) = classifier.svm_pair[k]
#     subset = findall(class -> class == i || class == j, y)
#     y_subset = [y[index] == i ? -1.0 : 1.0 for index in subset]
#     X_subset = X[subset, :]
#     if classify(classifier.svm[k], κ, X_subset, y_subset, x) < 0 # vote for class i
#       classifier.vote[i] += 1
#     else # vote for class j
#       classifier.vote[j] += 1
#     end
#   end
#   return argmax(classifier.vote)
# end

# function trainMM(classifier::MultiSVMClassifier, f, y, X, tol, k; kwargs...)
#   for ij in eachindex(classifier.svm)
#     (i, j) = classifier.svm_pair[ij]
#     println("\nTraining class $(i) against class $(j)\n")
#     subset = findall(class -> class == i || class == j, y)
#     y_subset = [y[index] == i ? -1.0 : 1.0 for index in subset]
#     if issymmetric(X) # nonlinear case
#       X_subset = X[subset, subset]
#     else # linear case
#       X_subset = X[subset, :]
#     end
#     trainMM(classifier.svm[ij], f, y_subset, X_subset, tol, k; kwargs...)
#   end
#   return classifier
# end
