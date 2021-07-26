import Base: show

struct SVMBatch{T<:AbstractFloat,S,KF}
    X::Matrix{T}
    K::Union{Nothing,Matrix{T}}
    y::Vector{T}
    kernelf::KF
    label2target::Dict{T,S}
end

"""
```
SVMBatch(X, y, label2target, [kernel])
```

Create a batch of samples `(ydata, Xdata)` for training a SVM.
The labels `ydata` must be coded as {-1,1}.

### Optional Arguments
- `kernelf`: A kernel function from KernelFunctions.jl used to construct the kernel matrix `K` (default=`nothing`).
"""
function SVMBatch(X, y, label2target, kernel::Union{Nothing,Kernel}=nothing)
    # Sanity checks.
    if size(X, 1) != length(y)
        throw(DimensionMismatch("Number of rows in `X` $(size(X, 1)) must match length of `y` $(length(y))."))
    end

    if any(yi -> abs(yi) != 1, y)
        throw(DomainError(y, "Input labels must have entries -1 or 1."))
    end

    # Assume linear case unless caller specifies a kernel function.
    if kernel isa Nothing
        K = nothing
    else
        K = kernelmatrix(kernel, X, obsdim=1)
    end

    return SVMBatch(X, K, y, kernel, label2target)
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
        A_predictors = data.K * Diagonal(data.y)
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

function (classifier::Classifier)(X::AbstractMatrix)
    classifier.(eachrow(X))
end

struct BinaryClassifier{T<:AbstractFloat,S,KF} <: Classifier
    weights::Vector{T}
    intercept::Bool
    data::SVMBatch{T,S,KF}
end

"""
```
BinaryClassifier{T}(Xdata, targets, refclass; [intercept], [kernel]) where T<:AbstractFloat
```

Create a binary classifier from the given training data given as a matrix `Xdata` and vector `targets`.
The type paramter `T` may be omitted in which case arrays default to `Float64`.
Elements in `targets` equal to `refclass` are assigned a positive value.

### Keyword Arguments
- `intercept`: Specifies whether to include an intercept in the model (default=`false`).
- `kernel`: A kernel function from KernelFunctions.jl used to construct the kernel matrix `K` (default=`nothing`).
"""
function BinaryClassifier{T}(Xdata, targets, refclass;
    intercept::Bool=false,
    kernel::Union{Nothing,Kernel}=nothing) where T<:AbstractFloat
    # sanity checks
    S = eltype(targets)
    if S != typeof(refclass)
        error("`targets` and `refclass` have different types.")
    end
    unique_labels = label(targets)
    if refclass ∉ unique_labels
        error("Reference class $(refclass) not found in `targets`.")
    end

    # Enforce an encoding using OneVsRest
    if nlabel(unique_labels) == 2
        # in this case, we can be explicit about the positive and negative labels
        if islabelenc(targets, LabelEnc.MarginBased)
            # labels already coded as {-1,1}
            enc = LabelEnc.OneVsRest(one(S), -one(S))
        else
            # otherwise need to check value for negative label
            other_class = unique_labels[findfirst(!=(refclass), unique_labels)]
            enc = LabelEnc.OneVsRest(refclass, other_class)
        end
    else
        # otherwise, there are multiple classes and we can only specify the reference class
        enc = LabelEnc.OneVsRest(refclass)
    end

    # handle mapping from target space to labels in {-1,1}
    y = convertlabel(LabelEnc.MarginBased(T), targets, enc)
    label2target = Dict(-one(T) => neglabel(enc), one(T) => poslabel(enc))

    # bundle training data into SVMBatch
    X = convert(Matrix{T}, Xdata)
    data = SVMBatch(X, y, label2target, kernel)

    # allocate vector that represents model, including intercept as needed
    if kernel isa Nothing
        # linear case
        weights = Vector{T}(undef, nfeatures(data) + intercept)
    else
        # nonlinear case
        weights = Vector{T}(undef, nsamples(data) + intercept)
    end

    return BinaryClassifier(weights, intercept, data)
end

"Create a `BinaryClassifier` assuming `Float64` arithmetic. See `BinaryClassifier{T}."
BinaryClassifier(Xdata, targets, refclass; kwargs...) = BinaryClassifier{Float64}(Xdata, targets, refclass; kwargs...)

function prediction(classifier::BinaryClassifier{T}, x) where T
    weights = classifier.weights
    data = classifier.data
    intercept = classifier.intercept
    κ = kernelf(data)
    X = data.X
    y = data.y

    if κ isa Nothing
        # linear case
        @views β, β₀ = weights[1:end-intercept], intercept*weights[end]
        fx = dot(x, β) + β₀
    else
        # nonlinear case
        fx = zero(T)
        @views α, α₀ = weights[1:end-intercept], intercept*weights[end]
        for (j, xj) in enumerate(eachrow(X))
            fx = fx + α[j] * y[j] * κ(x, xj)
        end
        fx = fx + α₀
    end

    return fx
end

function (classifier::BinaryClassifier{T})(x::AbstractVector) where T
    fx = prediction(classifier, x)
    label = classify(fx, LabelEnc.MarginBased(T))
    return classifier.data.label2target[label]
end

function trainMM(classifier::BinaryClassifier, f, tol, s; kwargs...)
    data = classifier.data
    intercept = classifier.intercept
    A = get_design_matrix(data, intercept)
    trainMM!(classifier, A, f, tol, s; kwargs...)
end

function trainMM!(classifier::BinaryClassifier, A, f, tol, s; verbose::Bool=false, kwargs...)
    data = classifier.data
    intercept = classifier.intercept
    k = sparsity_to_k(s, size(A, 2)-intercept)
    if verbose
        @time annealing!(f, classifier.weights, A, data.y, tol, k, intercept; verbose=true, kwargs...)
    else
        annealing!(f, classifier.weights, A, data.y, tol, k, intercept; verbose=false, kwargs...)
    end
end

abstract type MultiClassStrategy end

"One versus One: Create `k` choose `2` SVMs to classify data on `k` classes."
struct OVO <: MultiClassStrategy end

"One versus Rest: Create `k` SVMs to classify data on `k` classes."
struct OVR <: MultiClassStrategy end

struct MultiClassifier{T,S,MCS,KF} <: Classifier
    svm::Vector{BinaryClassifier{T,S,KF}}
    strategy::MCS
    votes::Vector{Int}
    target2ind::Dict{S,Int}
    label::Vector{S}
end

function MultiClassifier{T}(Xdata, targets;
    strategy::MultiClassStrategy=OVR(),
    intercept::Bool=false,
    kernel::Union{Nothing,Kernel}=nothing) where T<:AbstractFloat
    # setup labeled data and inferred encoding
    target2subset = labelmap(targets)
    labeled_data = (Xdata, targets)
    encoding = LabelEnc.NativeLabels(targets, label(targets))
    target2ind = encoding.invlabel
    K = nlabel(encoding)
    votes = Vector{Int}(undef, K)

    # sanity checks
    if K ≤ 2
        error("Only $(K) classes detected. Consider using `BinaryClassifier`.")
    end

    SVMType = BinaryClassifier{T,eltype(targets),typeof(kernel)}
    if strategy isa OVR # one versus rest
        svm = Vector{SVMType}(undef, K)
        for k in 1:K
            class_k = label(encoding)[k]
            svm[k] = BinaryClassifier{T}(Xdata, targets, class_k,
            intercept=intercept, kernel=kernel)
        end
    elseif strategy isa OVO # one versus one
        svm = Vector{SVMType}(undef, binomial(K, 2))
        k = 1
        for j in 1:K, i in j+1:K
            class_i = label(encoding)[i]
            class_j = label(encoding)[j]
            idx = union(target2subset[class_i], target2subset[class_j])
            X_subset, targets_subset = getobs(labeled_data, idx, obsdim=1)
            svm[k] = BinaryClassifier{T}(X_subset, targets_subset, class_i,
            intercept=intercept, kernel=kernel)
            k += 1
        end
    else
        error("Strategy $(strategy) is not yet implemented.")
    end

    return MultiClassifier(svm, strategy, votes, target2ind, label(encoding))
end

MultiClassifier(Xdata, targets; kwargs...) = MultiClassifier{Float64}(Xdata, targets; kwargs...)

function prediction(classifier::MultiClassifier{T}, x) where T
    fx = zeros(T, length(classifier.svm))
    for (i, f) in enumerate(classifier.svm)
        fx[i] = SparseSVM.prediction(f, x)
    end
    return fx
end

function (classifier::MultiClassifier{T})(x::AbstractVector) where T
    vote, target2ind = classifier.votes, classifier.target2ind
    fill!(vote, 0)

    if classifier.strategy isa OVR # one versus rest
        for f in classifier.svm
            preclass = f(x)
            refclass = f.data.label2target[one(T)]
            
            # only cast vote when 
            if preclass == refclass
                k = target2ind[preclass]
                vote[k] += 1
            end
        end
    elseif classifier.strategy isa OVO # one versus one
        for f in classifier.svm
            preclass = f(x)
            k = target2ind[preclass]
            vote[k] += 1
        end
    else
        error("Classification for strategy $(strategy) is not yet implemented.")
    end

    l = classify(vote, LabelEnc.OneOfK)
    return classifier.label[l]
end

function trainMM(classifier::MultiClassifier, f, tol, s;
    verbose::Bool=false,
    matrix=nothing,
    fullsvd=nothing,
    kwargs...
    )
    #
    if matrix isa Nothing
        Amat = ASVD = MaybeIndexable(nothing)
    else
        Amat = matrix
        if fullsvd isa Nothing
            ASVD = MaybeIndexable(nothing)
        else
            ASVD = [svd(A, full=true) for A in Amat]
        end
    end
    info = if verbose
        @time trainMM!(classifier, Amat, f, tol, s; verbose=true, fullsvd=ASVD, kwargs...)
    else
        trainMM!(classifier, Amat, f, tol, s; verbose=false, fullsvd=ASVD, kwargs...)
    end
    return info # total_iters, total_obj, total_dist
end

function trainMM!(classifier::MultiClassifier{T}, Amat, f, tol, s;
    verbose::Bool=false,
    fullsvd=MaybeIndexable(nothing),
    kwargs...) where T
    total_iters = 0
    total_obj = zero(T)
    total_dist = zero(T)
    M = length(classifier.svm)
    for (i, svm) in enumerate(classifier.svm)
        if verbose
            label2target = svm.data.label2target
            pos = label2target[one(T)]
            neg = label2target[-one(T)]
            println("Training $(pos) vs $(neg)")
            println()
        end
        # check if design matrix is already allocated
        if Amat[i] isa Nothing
            A = get_A(svm)
        else
            A = Amat[i]
        end
        iters, obj, dist = trainMM!(svm, A, f, tol, s; verbose=verbose, fullsvd=fullsvd[i], kwargs...)
        total_iters += iters
        total_obj += obj / M
        total_dist += dist / M
        verbose && println()
    end
    return total_iters, total_obj, total_dist
end

##### UTILS #####

get_A(c::BinaryClassifier) = get_design_matrix(c.data, c.intercept)

function get_A_and_SVD(c::BinaryClassifier)
    A = get_A(c)
    Asvd = svd(A)
    return A, Asvd
end

get_A(c::MultiClassifier) = get_A.(c.svm)

function get_A_and_SVD(c::MultiClassifier)
    A = get_A(c)
    Asvd = [svd(Ai) for Ai in A]
    return A, Asvd
end

import Base: getindex

struct MaybeIndexable{T,VT<:Union{Nothing,AbstractVector{T}}}
    data::VT
end

MaybeIndexable(::Nothing) = MaybeIndexable{Nothing,Nothing}(nothing)

Base.getindex(x::MaybeIndexable, i) = x.data[i]
Base.getindex(::MaybeIndexable{Nothing}, i) = nothing

function sparsity_to_k(s, n)
    abs(s) > 1 && error("Sparsity level must be between 0 and 1. Got: s = $(s) and n = $(n)")
    return max(1, round(Int, n * (1-s)))
end

##### checking support vectors #####

function get_support_vecs(classifier::BinaryClassifier)
    if classifier.data.kernelf isa Nothing
        # linear case: check violations
        Ab = map(x -> prediction(classifier, x), eachrow(classifier.data.X))
        idx = findall(classifier.data.y .* Ab .< 1)
    else
        # non-linear case: check for non-zero weights
        intercept = classifier.intercept
        idx = findall(!isequal(0), classifier.weights[1:end-intercept])
    end
    return sort!(idx)
end

function get_support_vecs(classifier::MultiClassifier)
    idx = Int[]
    for svm in classifier.svm
        union!(idx, get_support_vecs(svm))
    end
    return sort!(idx)
end

count_support_vecs(classifier::Classifier) = length(get_support_vecs(classifier))
