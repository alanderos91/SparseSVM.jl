abstract type SVMClassifier end

struct BinarySVMClassifier <: SVMClassifier
  b::Vector{Float64}
  coding::Dict{Int,String}
end

# constructor
function BinarySVMClassifier(nfeatures::Integer, coding)
  return BinarySVMClassifier(zeros(nfeatures), coding)
end

function classify(classifier::BinarySVMClassifier, x)
  return sign(dot(classifier.b, x))
end

function trainMM(classifier::BinarySVMClassifier, f, y, X, tol, k; kwargs...)
  annealing!(f, classifier.b, X, y, tol, k; kwargs...)
end

# implements one-against-one approach
struct MultiSVMClassifier <: SVMClassifier
  nclasses::Int
  svm::Vector{BinarySVMClassifier}
  svm_pair::Vector{Tuple{Int,Int}}
  class::Vector{String}
  vote::Vector{Int}
end

function MultiSVMClassifier(nfeatures::Int, class)
  nclasses = length(class)
  nsvms = binomial(nclasses, 2)
  svm = Vector{BinarySVMClassifier}(undef, nsvms)
  svm_pair = Vector{Tuple{Int,Int}}(undef, nsvms)
  k = 1
  for j in 1:nclasses, i in j+1:nclasses
    svm[k] = BinarySVMClassifier(nfeatures, Dict(-1 => class[i], 1 => class[j]))
    svm_pair[k] = (i, j)
    k += 1
  end
  return MultiSVMClassifier(nclasses, svm, svm_pair, class, zeros(Int, nclasses))
end

function classify(classifier::MultiSVMClassifier, x)
  fill!(classifier.vote, 0)
  for k in eachindex(classifier.svm)
    (i, j) = classifier.svm_pair[k]
    if classify(classifier.svm[k], x) < 0 # vote for class i
      classifier.vote[i] += 1
    else # vote for class j
      classifier.vote[j] += 1
    end
  end
  return argmax(classifier.vote)
end

function trainMM(classifier::MultiSVMClassifier, f, y, X, tol, k; kwargs...)
  for ij in eachindex(classifier.svm)
    (i, j) = classifier.svm_pair[ij]
    println("\nTraining class $(i) against class $(j)\n")
    subset = findall(class -> class == i || class == j, y)
    y_subset = [y[index] == i ? -1.0 : 1.0 for index in subset]
    X_subset = X[subset, :]
    @time trainMM(classifier.svm[ij], f, y_subset, X_subset, tol, k; kwargs...)
  end
  return classifier
end
