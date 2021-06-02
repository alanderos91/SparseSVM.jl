import Base: show

struct SVMBatch{T,KF}
  X::Matrix{T}
  K::Union{Nothing,Matrix{T}}
  y::Vector{T}
  kernelf::KF
  intercept::Bool
end

"""
```
SVMBatch(Xdata, ydata; ftype=nothing, kernelf=nothing, intercept=false)
```

Create a batch of samples `(ydata, Xdata)` for training a SVM.

### Keyword Arguments
- `ftype`: A floating point data type, which must have `AbstractFloat` as a supertype (default=`nothing`).
  By default, the floating point type is inferred from `Xdata`.
- `kernelf`: A kernel function from KernelFunctions.jl used to construct the kernel matrix `K` (default=`nothing`).
- `intercept`: Specifies whether to include an intercept in the model (default=`false`).
  If `intercept == true` and `kernelf === nothing`, then `Xdata` is concatenated with a column of 1s
  to incorporate an intercept.
  If `intercept == true` and `kernelf` is a kernel function, then the intercept term is handled later by a classification algorithm.
"""
function SVMBatch(Xdata, ydata; ftype::Union{Nothing,DataType}=nothing, kernelf::Union{Nothing,Kernel}=nothing, intercept::Bool=false)
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

  # Assume no intercept unless caller asks for it.
  if intercept && kernelf isa Nothing
    X = [X ones(eltype(X), size(X, 1))]
  end

  return SVMBatch(X, K, y, kernelf, intercept)
end

function Base.show(io::IO, data::SVMBatch)
  println(io, "SVMBatch{", eltype(data.X), "}")
  println(io, "  ∘ samples:   ", size(data.X, 1))
  println(io, "  ∘ features:  ", size(data.X, 2))
  println(io, "  ∘ kernel:    ", data.kernelf)
  print(io, "  ∘ intercept? ", data.intercept)
end

abstract type Classifier end

struct BinaryClassifier{T,KF} <: Classifier
  weights::Vector{T}
  data::SVMBatch{T,KF}
end

# constructor
function BinaryClassifier(data::SVMBatch{T,KF}) where {T,KF}
  if KF <: Nothing
    # linear case
    weights = Vector{T}(undef, size(data.X, 2))
  else
    # nonlinear case
    weights = Vector{T}(undef, size(data.X, 1))
  end

  return BinaryClassifier(weights, data)
end

function classify(classifier::BinaryClassifier{T,KF}, x) where {T,KF}
  fx = zero(T)

  if KF <: Nothing
    # linear case
    β = classifier.weights
    fx = dot(x, β)
  else
    # nonlinear case
    α = classifier.weights
    κ = classifier.data.kernelf
    X = classifier.data.X
    y = classifier.data.y

    for (j, xj) in enumerate(eachrow(X))
      fx = fx + α[j] * y[j] * κ(x, xj)
    end
  end

  return sign(fx)
end

function trainMM(classifier::BinaryClassifier, f, tol, k; kwargs...)
  if classifier.data.kernelf isa Nothing
    @time annealing!(f, classifier.weights, classifier.data.X, classifier.data.y, tol, k, classifier.data.intercept; kwargs...)
  else
    @time annealing!(f, classifier.weights, classifier.data.K, classifier.data.y, tol, k, classifier.data.intercept; kwargs...)
  end
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
