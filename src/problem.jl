abstract type MultiClassStrategy end

"One versus One: Create `c` choose `2` SVMs to classify data on `c` classes."
struct OVO <: MultiClassStrategy end

determine_number_svms(::OVO, c) = binomial(c, 2)

"One versus Rest: Create `c` SVMs to classify data on `c` classes."
struct OVR <: MultiClassStrategy end

determine_number_svms(::OVR, c) = c

has_inferrable_encoding(::Type{T}) where T <: Union{Number,AbstractString,Symbol} = true
has_inferrable_encoding(x) = false

function create_X_and_K(kernel::Kernel, data, intercept) where T<:Real
    K = kernelmatrix(kernel, data, obsdim=1)
    intercept && (K = [K ones(size(K, 1))])
    X = copy(data)
    return X, K
end

function create_X_and_K(::Nothing, data, intercept) where T<:Real
    X = intercept ? [data ones(size(data, 1))] : copy(data)
    return X, nothing
end

abstract type AbstractSVM end

struct BinarySVMProblem{T,S,YT,XT,KT,LT,encT,kernT,coeffT,resT,gradT} <: AbstractSVM
    n::Int  # number of samples
    p::Int  # number of predictors/features

    y::YT           # targets, n × 1
    X::XT           # features matrix, n × p
    K::KT           # kernel matrix, n × n
    intercept::Bool # indicates presence (true) or absence (false) of intercept term

    labels::LT          # original labels
    ovr_encoding::encT  # maps labels to a binary encoding

    kernel::kernT       # kernel used to construct kernel matrix in nonlinear classification
    coeff::coeffT       # estimates of coefficients/weights
    coeff_prev::coeffT  # previous estimates of coefficients/weights

    proj::coeffT        # projection of coefficients onto a particular sparsity set
    res::resT           # residuals in quadratic surrogate
    grad::gradT         # gradient with respect to coefficients

    function BinarySVMProblem{T}(n::Int, p::Int,
            y::YT, X::XT, K::KT, intercept::Bool,
            labels::AbstractVector{S}, ovr_encoding::encT,
            kernel::kernT, coeff::coeffT, coeff_prev::coeffT,
            proj::coeffT, res::resT, grad::gradT
        ) where {T,S,YT,XT,KT,encT,kernT,coeffT,resT,gradT}
        for arrayT in (YT, coeffT, resT, gradT)
            if eltype(arrayT) != T
                error("Inconsistent element types $(eltype(arrayT)) and $(T).")
            end
        end
        if kernel isa Nothing && eltype(XT) != T
            error("Inconsistent element types $(eltype(XT)) and $(T).")
        elseif !(kernel isa Nothing) && eltype(KT) != T
            error("Inconsistent element types $(eltype(KT)) and $(T).")
        end
        LT = typeof(labels)

        new{T,S,YT,XT,KT,LT,encT,kernT,coeffT,resT,gradT}(
            n, p, y, X, K, intercept,
            labels, ovr_encoding,
            kernel, coeff, coeff_prev,
            proj, res, grad
        )
    end
end

function BinarySVMProblem(labels, data, positive_label::S, coeff, coeff_prev, proj;
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing,
    negative_label::Union{Nothing,S}=nothing
    ) where S
    # Infer problem information.
    unique_labels = MLDataUtils.label(labels)
    n, p = size(data)
    T = eltype(data)

    # Sanity checks.
    if !has_inferrable_encoding(S) && negative_label isa Nothing
        error("""
        Cannot automatically infer an binary encoding for `$(S)`-valued labels without using heuristics.
        Please specify a negative label with the `negative_label` keyword argument.
        """)
    end
    if eltype(labels) != S
        error("`eltype(labels)` and `positive_label` have different types.")
    end
    if positive_label ∉ unique_labels
        error("Positive label $(positive_label) not found in `labels`.")
    end
    if length(labels) != n
        error("The labels ($(length(labels))) and data ($(n)) do not have the same number of observations.")
    end
    if nlabel(unique_labels) > 2 && negative_label ∈ unique_labels
        error("The negative label is found in the label set but labeled data contain more than two categories. Use a different negative label to avoid ambiguity.")
    end

    # Enforce an encoding using OneVsRest.
    if negative_label isa Nothing && nlabel(unique_labels) == 2
        nl = unique_labels[ findfirst(!isequal(positive_label), unique_labels) ]
        ovr_encoding = LabelEnc.OneVsRest(positive_label, nl)
    elseif negative_label isa Nothing && nlabel(unique_labels) != 2
        ovr_encoding = LabelEnc.OneVsRest(positive_label)
    else
        ovr_encoding = LabelEnc.OneVsRest(positive_label, negative_label)
    end
    margin_encoding = LabelEnc.MarginBased(T)
    y = convertlabel(margin_encoding, labels, ovr_encoding)

    # Create design matrices.
    X, K = create_X_and_K(kernel, data, intercept)

    # Allocate additional data structures used to fit a model.
    res = similar(data, n)
    grad = similar(coeff)

    return BinarySVMProblem{T}(n, p,
        y, X, K, intercept,
        labels, ovr_encoding,
        kernel, coeff, coeff_prev,
        proj, res, grad,
    )
end

function BinarySVMProblem(labels, data, positive_label;
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing,
    kwargs...)
    # Infer problem information.
    n, p = size(data)

    # Allocate arrays for coefficients.
    coeff = kernel isa Nothing ? similar(data, p+intercept) : similar(data, n+intercept)
    coeff_prev = similar(coeff)
    proj = similar(coeff)

    return BinarySVMProblem(labels, data, positive_label, coeff, coeff_prev, proj;
        intercept=intercept, kernel=kernel, kwargs...)
end

"""
Return the floating-point type used for model coefficients.
"""
floattype(::BinarySVMProblem{T}) where T = T

"""
Return the type used for data labels.
"""
labeltype(::BinarySVMProblem{T,S}) where {T,S} = S

"""
Return the design matrix used for fitting a classifier.

Uses `problem.X` when `problem.kernel isa Nothing` and `problem.K` when `problem.kernel isa Kernel` from KernelFunctions.jl.
"""
get_design_matrix(problem::BinarySVMProblem) = __get_design_matrix__(problem.kernel, problem) # dispatch
__get_design_matrix__(::Nothing, problem::BinarySVMProblem) = problem.X  # linear case
__get_design_matrix__(::Kernel, problem::BinarySVMProblem) = problem.K   # nonlinear case

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::BinarySVMProblem) = (problem.n, problem.p, 2)

"""
Return the label associated with the +1 target. 
"""
MLDataUtils.poslabel(problem::BinarySVMProblem) = poslabel(problem.ovr_encoding)

"""
Return the label associated with the -1 target. Note that this label may be arbitrary.
"""
MLDataUtils.neglabel(problem::BinarySVMProblem) = neglabel(problem.ovr_encoding)

function Base.show(io::IO, problem::BinarySVMProblem)
    n, p, _ = probdims(problem)
    T = floattype(problem)
    S = labeltype(problem)
    kernT = typeof(problem.kernel)
    pl, nl = poslabel(problem), neglabel(problem)
    classifier_kind = kernT <: Nothing ? "linear" : "nonlinear"
    print(io, "BinarySVMProblem{$(T),$(S),$(kernT)}")
    print(io, "\n  ∘ $(classifier_kind) classifier")
    print(io, "\n  ∘ $(n) sample(s)")
    print(io, "\n  ∘ $(p) feature(s)")
    print(io, "\n  ∘ positive label: $(pl)")
    print(io, "\n  ∘ negative label: $(nl)")
    print(io, "\n  ∘ intercept? $(problem.intercept)")
end

"""
    predict(problem::BinarySVMProblem, x)

When `x` is a vector, predict the target value of a sample/instance `x` based on the fitted model in `problem`.
Otherwise if `x` is a matrix then each sample is assumed to be aligned along rows (e.g. `x[i,:]` is sample `i`).

See also: [`classify`](@ref)
"""
predict(problem::BinarySVMProblem, x) = __predict__(problem.kernel, problem, x)

function __predict__(::Nothing, problem::BinarySVMProblem, x::AbstractVector)
    @unpack p, proj, intercept = problem
    β = view(proj.all, 1:p)
    β0 = proj.all[p+intercept]
    yhat = dot(view(x, 1:p), β)
    intercept && (yhat += β0)
    return yhat
end

function __predict__(::Nothing, problem::BinarySVMProblem, X::AbstractMatrix)
    @unpack p, proj, intercept = problem
    β = view(proj.all, 1:p)
    β0 = proj.all[p+intercept]
    yhat = view(X, :, 1:p) * β
    intercept && (yhat += β0)
    return yhat
end

function __predict__(::Kernel, problem::BinarySVMProblem, _x::AbstractVector)
    x = view(_x, 1:n)
    κ = problem.kernel
    α = view(problem.proj, 1:n)
    α0 = problem.proj[n+intercept]
    X = problem.X
    yhat = zero(floattype(problem))
    for (j, xⱼ) in enumerate(eachrow(X))
        yhat += α[j] * y[j] * κ(x, xⱼ)
    end
    intercept && (yhat += α0)
    return yhat
end

function __predict__(::Kernel, problem::BinarySVMProblem, X::AbstractMatrix)
    n = size(X, 1)
    yhat = Vector{floattype(problem)}(undef, n)
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @batch per=core for i in 1:n
        yhat[i] = predict(problem, view(X, i, :))
    end
    BLAS.set_num_threads(nthreads)
    return yhat
end

"""
    classify(problem::BinarySVMProblem, x)

Classify the samples/instances in `x` based on the model in `problem`.

If `x` is a vector then it is treated as an instance.
Otherwise if `x` is a matrix then each sample is assumed to be aligned along rows (e.g. `x[i,:]` is sample `i`).
See also: [`predict`](@ref)
"""
MLDataUtils.classify(problem::BinarySVMProblem, x) = __classify__(problem, predict(problem, x))

function __classify__(problem::BinarySVMProblem, y::Number)
    @unpack ovr_encoding = problem
    margin_encoding = LabelEnc.MarginBased(floattype(problem))
    predicted_label = classify(y, margin_encoding)
    return convertlabel(ovr_encoding, predicted_label, margin_encoding)
end

function __classify__(problem::BinarySVMProblem, y::AbstractVector)
    n = size(Y, 1)
    label = Vector{labeltype(problem)}(undef, n)
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @batch per=core for i in eachindex(label)
        label[i] = __classify__(problem, y[i])
    end
    BLAS.set_num_threads(nthreads)
    return label
end

struct MultiSVMProblem{T,S,BSVMT,dataT,kernT,stratT,encT,coeffT} <: AbstractSVM
    n::Int # number of samples
    p::Int # number of predictors/features
    c::Int # number of classes
    
    svm::Vector{BSVMT}  # collection of BinarySVMProblem objects
    labels::Vector{S}   # original data labels
    data::dataT         # original data matrix
    subset::Vector{Vector{Int}} # indicates indices along data rows used to fit binary SVMs
    intercept::Bool     # indicates presence (true) or absence (false) of intercept term

    kernel::kernT       # kernel used to construct kernel matrix in nonlinear classification
    strategy::stratT    # strategy for breaking up multiclass problem into binary problems
    encoding::encT
    vote::Vector{Int}   # vector used to classify a feature vector, c × 1

    coeff::coeffT       # estimates of coefficients/weights
    coeff_prev::coeffT  # previous estimates of coefficients/weights
    proj::coeffT        # projection of coefficients onto a particular sparsity set

    function MultiSVMProblem{T}(n::Int, p::Int, c::Int,
            svm::Vector{BSVMT}, labels::Vector{S}, data::dataT, subset::Vector{Vector{Int}}, intercept::Bool,
            kernel::kernT, strategy::stratT, encoding::encT, vote::Vector{Int},
            coeff::coeffT, coeff_prev::coeffT, proj::coeffT
        ) where {T,S,BSVMT,dataT,kernT,stratT,encT,coeffT}
        new{T,S,BSVMT,dataT,kernT,stratT,encT,coeffT}(
            n, p, c,
            svm, labels, data, subset, intercept,
            kernel, strategy, encoding, vote,
            coeff, coeff_prev, proj,
        )
    end
end

function __construct_svms__(::OVO, nsvms, labels, data, coeff, coeff_prev, proj, intercept, kernel, _ignore_)
    lm = labelmap(labels)
    ul = sort!(collect(keys(lm)))
    labeled_data = (labels, data)

    # helper function to create an SVM using OVO
    instantiate_svm = function(i, j, k)
        pl, nl = ul[i], ul[j]
        idx = sort!(union(lm[pl], lm[nl]))
        ls, ds = datasubset(labeled_data, idx, obsdim=1)
        cv, cpv, pv = view(coeff, :, k), view(coeff_prev, :, k), view(proj, :, k)
        idx, BinarySVMProblem(ls, ds, pl, cv, cpv, pv; kernel=kernel, intercept=intercept, negative_label=nl)
    end

    k = 1
    subset_1, svm_1 = instantiate_svm(2, 1, k)
    subset, svm = [subset_1], [svm_1]
    for j in 1:length(ul), i in j+1:length(ul)
        if i == 2 && j == 1 continue end
        k += 1
        subset_k, svm_k = instantiate_svm(i, j, k)
        push!(subset, subset_k)
        push!(svm, svm_k)
    end

    return svm, subset
end

function __construct_svms__(::OVR, nsvms, labels, data, coeff, coeff_prev, proj, intercept, kernel, nl)
    lm = labelmap(labels)
    ul = sort!(collect(keys(lm)))
    idx = collect(eachindex(labels))
    ls, ds = labels, data

    # helper function to create an SVM usingn OVR
    instantiate_svm = function(i)
        pl = ul[i]
        cv, cpv, pv = view(coeff, :, i), view(coeff_prev, :, i), view(proj, :, i)
        BinarySVMProblem(ls, ds, pl, cv, cpv, pv; kernel=kernel, intercept=intercept, negative_label=nl)
    end

    subset = [idx for _ in 1:nsvms]
    svm = [instantiate_svm(i) for i in 1:nsvms]

    return svm, subset
end

function MultiSVMProblem(labels, data, coeff, coeff_prev, proj;
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing,
    strategy::MultiClassStrategy=OVO(),
    negative_label::Union{Nothing,S}=nothing,
    ) where S
    # Infer problem information.
    unique_labels = MLDataUtils.label(labels)
    n, p = size(data)
    c = nlabel(unique_labels)
    T = eltype(data)
    encoding = LabelEnc.NativeLabels(unique_labels)

    # Construct binary classifiers to handle multiclass problem.
    nsvms = determine_number_svms(strategy, c)
    svm, subset = __construct_svms__(strategy, nsvms, labels, data, coeff, coeff_prev, proj, intercept, kernel, negative_label)

    # Allocate array to handle voting by binary classifiers.
    vote = zeros(Int, c)

    return MultiSVMProblem{T}(
        n, p, c,
        svm, labels, data, subset, intercept,
        kernel, strategy, encoding, vote,
        coeff, coeff_prev, proj,
    )
end

function MultiSVMProblem(labels, data;
    intercept::Bool=true,
    kernel::Union{Nothing,Kernel}=nothing,
    strategy::MultiClassStrategy=OVO(),
    kwargs...
    )
    # Infer problem information.
    unique_labels = MLDataUtils.label(labels)
    n, p = size(data)
    c = nlabel(unique_labels)

    # Allocate arrays for coefficients.
    nsvms = determine_number_svms(strategy, c)
    coeff = kernel isa Nothing ? similar(data, p+intercept, nsvms) : similar(data, n+intercept, nsvms)
    coeff_prev = similar(coeff)
    proj = similar(coeff)

    return MultiSVMProblem(labels, data, coeff, coeff_prev, proj;
        intercept=intercept, kernel=kernel, strategy=strategy, kwargs...)
end

"""
Return the floating-point type used for model coefficients.
"""
floattype(::MultiSVMProblem{T}) where T = T

"""
Return the type used for data labels.
"""
labeltype(::MultiSVMProblem{T,S}) where {T,S} = S

"""
Returns the number of samples, number of features, and number of categories, respectively.
"""
probdims(problem::MultiSVMProblem) = (problem.n, problem.p, problem.c)

function Base.show(io::IO, problem::MultiSVMProblem)
    n, p, c = probdims(problem)
    T = floattype(problem)
    S = labeltype(problem)
    kernT = typeof(problem.kernel)
    classifier_kind = kernT <: Nothing ? "linear" : "nonlinear"
    stratT = typeof(problem.strategy)
    strategy_kind = stratT <: OVO ? "One versus One" : "One versus Rest"
    nsvms = length(problem.svm)
    print(io, "MultiSVMProblem{$(T),$(S),$(kernT)}")
    print(io, "\n  ∘ $(classifier_kind) classifier")
    print(io, "\n  ∘ $(strategy_kind) ($(nsvms) SVMs)")
    print(io, "\n  ∘ $(n) sample(s)")
    print(io, "\n  ∘ $(p) feature(s)")
    print(io, "\n  ∘ $(c) categorie(s)")
    print(io, "\n  ∘ intercept? $(problem.intercept)")
end

cast_votes!(problem, x) = __cast_votes__!(problem.strategy, problem.svm, problem.encoding, problem.vote, x)

function __cast_votes__!(::OVO, svms, encoding, vote, x)
    fill!(problem.vote, 0)
    for svm in svms
        predicted_label = classify(svm, x)
        j = label2ind(predicted_label, encoding)
        vote[j] += 1
    end
end

function __cast_votes__!(::OVR, svms, encoding, vote, x)
    fill!(problem.vote, 0)
    for svm in svms
        predicted_label = classify(svm, x)
        positive_label = poslabel(svm)
        if predicted_label == positive_label
            j = label2ind(predicted_label, encoding)
            vote[j] += 1
        end
    end
end

determine_winner(problem::MultiSVMProblem) = ind2label(argmax(problem.vote), problem.encoding)

function MLDataUtils.classify(problem::MultiSVMProblem, x::AbstractVector)
    cast_votes!(problem, x)
    predicted_label = determine_winner(problem)
    return predicted_label
end

function MLDataUtils.classify(problem::MultiSVMProblem, X::AbstractMatrix)
    predicted_label = Vector{labeltype(problem)}(undef, size(X, 1))
    for i in eachindex(predicted_label)
        predicted_label[i] = classify(problem, view(X, i, :))
    end
    return predicted_label
end