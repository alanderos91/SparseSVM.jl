abstract type MultiClassStrategy end

"One versus One: Create `c` choose `2` SVMs to classify data on `c` classes."
struct OVO <: MultiClassStrategy end

determine_number_svms(::OVO, c) = binomial(c, 2)

"One versus Rest: Create `c` SVMs to classify data on `c` classes."
struct OVR <: MultiClassStrategy end

determine_number_svms(::OVR, c) = c

has_inferrable_encoding(::Type{T}) where T <: Union{Number,AbstractString,Symbol} = true
has_inferrable_encoding(x) = false

create_X_and_K(kernel::Kernel, y, data) = (copy(data), rmul!( kernelmatrix(kernel, data, obsdim=1), Diagonal(y) ))

create_X_and_K(::Nothing, y, data) = (copy(data), nothing)

function alloc_targets!(y, l, margin_enc, ovr_enc)
    for i in eachindex(y)
        y[i] = MLDataUtils.convertlabel(margin_enc, l[i], ovr_enc)
    end
    return y
end

alloc_coefficients(::Nothing, data, n, p, intercept) = similar(data, intercept+p)
alloc_coefficients(::Kernel, data, n, p, intercept) = similar(data, intercept+n)
alloc_coefficients(::Nothing, data, n, p, nsvms, intercept) = similar(data, intercept+p, nsvms)
alloc_coefficients(::Kernel, data, n, p, nsvms, intercept) = similar(data, intercept+n, nsvms)

get_coefficients_view(::Union{Nothing,Kernel}, data, arr::Matrix, n, k, intercept) = view(arr, :, k)

function get_coefficients_view(::Kernel, data, arr::VectorOfVectors, n, k, intercept)
    push!(arr, similar(data, intercept+n))
    return arr[k]
end

abstract type AbstractSVM end

struct BinarySVMProblem{T,S,YT,XT,KT,LT,encT,kernT,coeffT,resT,gradT} <: AbstractSVM
    n::Int  # number of samples
    p::Int  # number of predictors/features

    y::YT           # targets, n × 1
    X::XT           # features matrix, n × p
    KY::KT           # kernel matrix, n × n
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
            y::YT, X::XT, KY::KT, intercept::Bool,
            labels::AbstractVector{S}, ovr_encoding::encT,
            kernel::kernT, coeff::coeffT, coeff_prev::coeffT,
            proj::coeffT, res::resT, grad::gradT
        ) where {T,S,YT,XT,KT,encT,kernT,coeffT,resT,gradT}
        for arrayT in (YT, coeffT, gradT) # need a way to check nested arrays? like res
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

        fill!(coeff, zero(T))
        fill!(coeff_prev, zero(T))
        fill!(proj, zero(T))
        fill!(res.main, zero(T))
        fill!(res.dist, zero(T))
        fill!(grad, 0)

        new{T,S,YT,XT,KT,LT,encT,kernT,coeffT,resT,gradT}(
            n, p, y, X, KY, intercept,
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
    y = similar(data, n)
    alloc_targets!(y, labels, margin_encoding, ovr_encoding)

    # Create design matrices.
    X, KY = create_X_and_K(kernel, y, data)

    # Allocate additional data structures used to fit a model.
    res = (; main=similar(data, n), dist=similar(coeff))
    grad = similar(coeff)

    return BinarySVMProblem{T}(n, p,
        y, X, KY, intercept,
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
    coeff = alloc_coefficients(kernel, data, n, p, intercept)
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

Uses `problem.X` when `problem.kernel isa Nothing` and `problem.KY` when `problem.kernel isa Kernel` from KernelFunctions.jl.
"""
get_design_matrix(problem::BinarySVMProblem) = __get_design_matrix__(problem.kernel, problem) # dispatch
__get_design_matrix__(::Nothing, problem::BinarySVMProblem) = problem.X  # linear case
__get_design_matrix__(::Kernel, problem::BinarySVMProblem) = problem.KY   # nonlinear case

"""
Return the data matrix used to fit the SVM.
"""
get_problem_data(problem::BinarySVMProblem) = problem.X

"""
Access model parameters (intercept + coefficients).
"""
get_params(problem::BinarySVMProblem) = __slope_and_coeff_views__(problem.coeff, problem.intercept)

"""
Access previous model parameters (intercept + coefficients).
"""
get_params_prev(problem::BinarySVMProblem) = __slope_and_coeff_views__(problem.coeff_prev, problem.intercept)

"""
Access projected model parameters (intercept + coefficients).
"""
get_params_proj(problem::BinarySVMProblem) = __slope_and_coeff_views__(problem.proj, problem.intercept)

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

"""
    change_data(problem::BinarySVMProblem, labels::AbstractVector, data)

Create a new `BinarySVMProblem` instance from the labeled dataset `(label, data)` using the vertex encoding from the reference `problem`.
"""
function change_data(problem::BinarySVMProblem, labels, data)
    @unpack ovr_encoding = problem
    @unpack intercept, kernel = problem

    T = floattype(problem)
    has_intercept = all(isequal(1), data[:,end])
    n, p = length(labels), has_intercept ? size(data, 2)-1 : size(data, 2)

    # Enforce an encoding using OneVsRest.
    margin_encoding = LabelEnc.MarginBased(T)
    y = similar(data, n)
    alloc_targets!(y, labels, margin_encoding, ovr_encoding)

    # Create design matrices.
    X, KY = create_X_and_K(kernel, y, data)

    # Allocate additional data structures used to fit a model.
    coeff = alloc_coefficients(kernel, data, n, p, intercept)
    coeff_prev = similar(coeff)
    proj = similar(coeff)
    res = (; main=similar(data, n), dist=similar(coeff))
    grad = similar(coeff)

    return BinarySVMProblem{T}(n, p,
        y, X, KY, intercept,
        labels, ovr_encoding,
        kernel, coeff, coeff_prev,
        proj, res, grad,
    )
end

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
    @unpack p = problem
    b, w = get_params_proj(problem)
    yhat = b + dot(view(x, 1:p), w)
    return yhat
end

function __predict__(::Nothing, problem::BinarySVMProblem, _X::AbstractMatrix)
    T, p = floattype(problem), problem.p
    b, w = get_params_proj(problem)
    X = view(_X, :, 1:p)
    yhat = similar(X, size(X, 1))
    fill!(yhat, b)
    BLAS.gemv!('N', one(T), X, w, one(T), yhat)
    return yhat
end

function __predict__(kernel::Kernel, problem::BinarySVMProblem, _x::AbstractVector)
    @unpack p, y = problem
    b, w = get_params_proj(problem)
    x = view(_x, 1:p)
    KY = kernelmatrix(kernel, problem.X, x', obsdim=1)
    lmul!(Diagonal(y), KY)
    yhat = BLAS.dot(KY, w) + b
    return yhat
end

function __predict__(kernel::Kernel, problem::BinarySVMProblem, _X::AbstractMatrix)
    @unpack p, y = problem
    T = floattype(problem)
    b, w = get_params_proj(problem)
    X = view(_X, :, 1:p)
    yhat = Vector{floattype(problem)}(undef, size(X, 1))
    fill!(yhat, b)
    KY = kernelmatrix(kernel, X, problem.X, obsdim=1)
    rmul!(KY, Diagonal(y))
    BLAS.gemv!('N', one(T), KY, w, one(T), yhat)
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
    n = length(y)
    predicted_label = Vector{labeltype(problem)}(undef, n)
    nthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    @batch per=core for i in eachindex(predicted_label)
        @inbounds predicted_label[i] = __classify__(problem, y[i])
    end
    BLAS.set_num_threads(nthreads)
    return predicted_label
end

"""
Retrieve indices for support vectors of the binary SVM.

- In the linear case, a point ``x_{i}`` is a support vector if ``max\\{0, 1 - y_{i} f(x_{i})\\} > 0``.
- In the nonlinear case, a point ``x_{i}`` is a support vector if its coefficient ``\\alpha_{i} > 0``.  
"""
support_vectors(problem::BinarySVMProblem) = sort!(support_vectors(problem.kernel, problem))

function support_vectors(::Nothing, problem::BinarySVMProblem)
    y = problem.y
    yhat = SparseSVM.predict(problem, problem.X)

    # determine scale of values
    scale_min, scale_max = Inf, -Inf
    @inbounds for i in eachindex(y)
        z = y[i] * yhat[i]
        absz = abs(z)
        if absz > 0
            scale_min = min(scale_min, absz)
            scale_max = max(scale_max, absz)
        end
    end
    tol = sqrt(eps() / min(1, scale_min))

    margin_sv = Int[]
    nonmargin_sv = Int[]
    @inbounds for i in eachindex(y)
        z = y[i]*yhat[i]
        on_margin = abs(1-z) / scale_max <= tol
        slack = max(0, 1-z)
        positive_slack = slack > 0
        if positive_slack # non-margin support vectors
            push!(nonmargin_sv, i)
        elseif on_margin # margin support vectors
            push!(margin_sv, i)
        end
    end
    return union!(margin_sv, nonmargin_sv)
end

function support_vectors(::Kernel, problem::BinarySVMProblem)
    _, projected_coefficients = get_params_proj(problem)
    return findall(!iszero, projected_coefficients)
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

function __construct_svms__(::OVO, lm, nsvms, labels, data, coeff, coeff_prev, proj, intercept, kernel, _ignore_)
    ul = sort!(collect(keys(lm)))
    labeled_data = (labels, data)

    # helper function to create an SVM using OVO
    instantiate_svm = function(i, j, k)
        pl, nl = ul[i], ul[j]
        idx = sort!(union(lm[pl], lm[nl]))
        ls, ds = datasubset(labeled_data, idx, obsdim=1)
        n = length(ls)

        # Assign view of coefficients from parent array.
        cv = get_coefficients_view(kernel, ds, coeff, n, k, intercept)
        cpv = get_coefficients_view(kernel, ds, coeff_prev, n, k, intercept)
        pv = get_coefficients_view(kernel, ds, proj, n, k, intercept)

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

function __construct_svms__(::OVR, lm, nsvms, labels, data, coeff, coeff_prev, proj, intercept, kernel, nl)
    ul = sort!(collect(keys(lm)))
    idx = collect(eachindex(labels))
    ls, ds = labels, data

    # helper function to create an SVM usingn OVR
    instantiate_svm = function(i)
        pl = ul[i]
        n = length(ls)

        # Assign view of coefficients from parent array.
        cv = get_coefficients_view(kernel, ds, coeff, n, i, intercept)
        cpv = get_coefficients_view(kernel, ds, coeff_prev, n, i, intercept)
        pv = get_coefficients_view(kernel, ds, proj, n, i, intercept)
        
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
    lm = labelmap(labels)

    # Construct binary classifiers to handle multiclass problem.
    nsvms = determine_number_svms(strategy, c)
    svm, subset = __construct_svms__(strategy, lm, nsvms, labels, data, coeff, coeff_prev, proj, intercept, kernel, negative_label)

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
    if kernel isa Nothing || strategy isa OVR
        coeff = alloc_coefficients(kernel, data, n, p, nsvms, intercept)
        coeff_prev = similar(coeff)
        proj = similar(coeff)
    else # nonlinear w/ OVO strategy
        coeff = VectorOfVectors{eltype(data)}()
        coeff_prev = VectorOfVectors{eltype(data)}()
        proj = VectorOfVectors{eltype(data)}()
    end

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

"""
Return the data matrix used to fit the SVM.
"""
get_problem_data(problem::MultiSVMProblem) = problem.data

"""
Access model parameters (intercept + coefficients).
"""
get_params(problem::MultiSVMProblem) = __slope_and_coeff_views__(problem.coeff, problem.intercept)

"""
Access previous model parameters (intercept + coefficients).
"""
get_params_prev(problem::MultiSVMProblem) = __slope_and_coeff_views__(problem.coeff_prev, problem.intercept)

"""
Access projected model parameters (intercept + coefficients).
"""
get_params_proj(problem::MultiSVMProblem) = __slope_and_coeff_views__(problem.proj, problem.intercept)

"""
Retrieve indices for support vectors of the multiclass SVM.
    
This is taken as the union of support indices for every binary SVM in the multiclass reduction. 
"""
support_vectors(problem::MultiSVMProblem) = sort!(support_vectors(problem.strategy, problem))

function support_vectors(::OVR, problem::MultiSVMProblem)
    support_idx = Int[]
    for binary_svm in problem.svm
        union!(support_idx, support_vectors(binary_svm))
    end
    return support_idx
end

function support_vectors(::OVO, problem::MultiSVMProblem)
    support_idx = Int[]
    for (binary_svm, subset) in zip(problem.svm, problem.subset)
        svm_idx = support_vectors(binary_svm)
        idx = view(subset, svm_idx)
        union!(support_idx, idx)
    end
    return support_idx
end

"""
    change_data(problem::BinarySVMProblem, labels::AbstractVector, data)

Create a new `BinarySVMProblem` instance from the labeled dataset `(label, data)` using the vertex encoding from the reference `problem`.
"""
function change_data(problem::MultiSVMProblem, labels, data)
    # Inherit properties from original problem
    @unpack c, kernel, intercept, strategy, encoding = problem
    nsvms = length(problem.svm)
    T = floattype(problem)
    lm = labelmap(labels)

    # Make sure label map accounts for every category in original problem
    unique_labels = label(problem.labels)
    for l in unique_labels
        if !haskey(lm, l)
            lm[l] = Int[]
        end
    end

    # Infer problem information from data.    
    has_intercept = all(isequal(1), data[:,end])
    n, p = length(labels), has_intercept ? size(data, 2)-1 : size(data, 2)

    # Allocate additional data structures used to fit a model.
    if kernel isa Nothing || strategy isa OVR
        coeff = alloc_coefficients(kernel, data, n, p, nsvms, intercept)
        coeff_prev = similar(coeff)
        proj = similar(coeff)
    else # OVO
        coeff = VectorOfVectors{eltype(data)}()
        coeff_prev = VectorOfVectors{eltype(data)}()
        proj = VectorOfVectors{eltype(data)}()
    end

    # Construct binary classifiers to handle multiclass problem.
    svm, subset = __change_svm_data__(strategy, lm, problem, labels, data, coeff, coeff_prev, proj)

    # Allocate array to handle voting by binary classifiers.
    vote = zeros(Int, c)

    return MultiSVMProblem{T}(
        n, p, c,
        svm, labels, data, subset, intercept,
        kernel, strategy, encoding, vote,
        coeff, coeff_prev, proj,
    )
end

function __change_svm_data__(::OVO, lm, problem, labels, data, coeff, coeff_prev, proj)
    has_intercept = all(isequal(1), data[:,end])
    labeled_data = (labels, data)

    # helper function to create an SVM using OVO
    instantiate_svm = function(old, i)
        @unpack ovr_encoding = old
        @unpack intercept, kernel = old
        T = floattype(old)

        pl, nl = MLDataUtils.poslabel(old), MLDataUtils.neglabel(old)
        idx = sort!(union(lm[pl], lm[nl]))
        ls, ds = datasubset(labeled_data, idx, obsdim=1)
        n, p = length(ls), has_intercept ? size(ds, 2)-1 : size(ds, 2)

        # Enforce an encoding using OneVsRest.
        margin_encoding = LabelEnc.MarginBased(T)
        y = similar(ds, n)
        alloc_targets!(y, ls, margin_encoding, ovr_encoding)

        # Create design matrices.
        X, KY = create_X_and_K(kernel, y, ds)

        # Assign view of coefficients from parent array.
        cv = get_coefficients_view(kernel, ds, coeff, n, i, intercept)
        cpv = get_coefficients_view(kernel, ds, coeff_prev, n, i, intercept)
        pv = get_coefficients_view(kernel, ds, proj, n, i, intercept)

        # Allocate additional data structures.
        res = (; main=similar(ds, n), dist=similar(cv))
        grad = similar(cv)

        svm = BinarySVMProblem{T}(n, p,
            y, X, KY, intercept,
            ls, ovr_encoding,
            kernel, cv, cpv, pv, res, grad,
        )

        return idx, svm
    end

    subset_1, svm_1 = instantiate_svm(problem.svm[1], 1)
    subset, svms = [subset_1], [svm_1]
    for i in 2:length(problem.svm)
        old = problem.svm[i]
        subset_i, svm_i = instantiate_svm(old, i)
        push!(subset, subset_i)
        push!(svms, svm_i)
    end

    return svms, subset
end

function __change_svm_data__(::OVR, lm, problem, labels, data, coeff, coeff_prev, proj)
    has_intercept = all(isequal(1), data[:,end])
    ls, ds = (labels, data)

    # helper function to create an SVM usingn OVR
    instantiate_svm = function(old, i)
        @unpack ovr_encoding = old
        @unpack intercept, kernel = old
    
        T = floattype(old)
        n, p = length(ls), has_intercept ? size(ds, 2)-1 : size(ds, 2)
        
        # Enforce an encoding using OneVsRest.
        margin_encoding = LabelEnc.MarginBased(T)
        y = similar(ds, n)
        alloc_targets!(y, ls, margin_encoding, ovr_encoding)

        # Create design matrices.
        X, KY = create_X_and_K(kernel, y, ds)

        # Assign view of coefficients from parent array.
        cv = get_coefficients_view(kernel, ds, coeff, n, i, intercept)
        cpv = get_coefficients_view(kernel, ds, coeff_prev, n, i, intercept)
        pv = get_coefficients_view(kernel, ds, proj, n, i, intercept)
        
        # Allocate additional data structures.
        res = (; main=similar(ds, n), dist=similar(cv))
        grad = similar(cv)

        return BinarySVMProblem{T}(n, p,
            y, X, KY, intercept,
            ls, ovr_encoding,
            kernel, cv, cpv, pv, res, grad,
        )
    end

    idx = collect(eachindex(labels))
    subset = [idx for _ in eachindex(problem.svm)]
    svm = [instantiate_svm(svm, i) for (i, svm) in enumerate(problem.svm)]

    return svm, subset
end

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

predict(problem::MultiSVMProblem, x) = __predict__(problem.kernel, problem, x)

function __predict__(::Nothing, problem::MultiSVMProblem, x::AbstractVector)
    yhat = Vector{Float64}(undef, length(problem.svm))
    for (j, svm) in enumerate(problem.svm)
        @inbounds yhat[j] = __predict__(nothing, svm, x)
    end
    return yhat
end

function __predict__(::Nothing, problem::MultiSVMProblem, X::AbstractMatrix)
    yhat = Matrix{Float64}(undef, size(X, 1), length(problem.svm))
    for (j, svm) in enumerate(problem.svm)
        @inbounds yhat[:,j] .= __predict__(nothing, svm, X)
    end
    return yhat
end

function __predict__(kernel::Kernel, problem::MultiSVMProblem, x::AbstractVector)
    p = problem.p
    yhat = Vector{Float64}(undef, length(problem.svm))
    @inbounds for (k, svm) in enumerate(problem.svm)
        b, w = get_params_proj(svm)
        KY = kernelmatrix(kernel, svm.X, view(x, 1:p)', obsdim=1)
        lmul!(Diagonal(svm.y), KY)
        yhat[k] = BLAS.dot(KY, w) + b
    end
    return yhat
end

function __predict__(kernel::Kernel, problem::MultiSVMProblem, X::AbstractMatrix)
    p = problem.p
    yhat = Matrix{Float64}(undef, size(X, 1), length(problem.svm))
    idx = Int[]
    for subset in problem.subset
        sort!(union!(idx, subset))
    end
    K = kernelmatrix(kernel, view(X, :, 1:p), problem.data, obsdim=1)
    @inbounds for k in axes(yhat, 2)
        b, w = get_params_proj(problem.svm[k])
        c = sparsevec(problem.subset[k], w .* problem.svm[k].y, size(K, 2))
        yhat[:,k] .= K*c .+ b
    end
    return yhat
end

function MLDataUtils.classify(problem::MultiSVMProblem, x)
    y = predict(problem, x)
    __classify__(problem.strategy, y, problem.svm, problem.encoding)
end

function __classify__(strategy::MultiClassStrategy, Y::AbstractMatrix, svm, encoding)
    S = MLDataUtils.labeltype(encoding)
    predicted_label = Vector{S}(undef, size(Y, 1))
    num_julia_threads = Threads.nthreads()

    if num_julia_threads == 1
        for i in axes(Y, 1)
            y = view(Y, i, :)
            @inbounds predicted_label[i] = __classify__(strategy, y, svm, encoding)
        end
    else # num_julia_threads > 1
        workers = [(similar(Y, size(Y, 2)), similar(Y, Int, size(Y, 2))) for _ in 1:num_julia_threads]
        num_BLAS_threads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        @batch per=core for i in axes(Y, 1)
            y = view(Y, i, :)
            confidence, vote = workers[Threads.threadid()]
            @inbounds predicted_label[i] = __classify__(strategy, y, svm, encoding, confidence, vote)
        end
        BLAS.set_num_threads(num_BLAS_threads)
    end
    
    return predicted_label
end

function __classify__(strategy::MultiClassStrategy, y::AbstractVector, svm, encoding)
    __classify__(strategy, y, svm, encoding, similar(y), similar(y, Int))
end

function __classify__(::OVO, y::AbstractVector, svm, encoding, confidence, vote)
    fill!(confidence, 0)
    fill!(vote, 0)
    # Array 'confidence' accumulates values y_k predicted by each classifier.
    # These values are scaled to (-1/3, 1/3) to nudge votes in case of ties.
    # Crucially, this should not change results when there is a difference of one vote.
    #
    # This is the approach used by SciKitLearn (3/12/22):
    # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/multiclass.py#L474
    @inbounds for (y_k, svm_k) in zip(y, svm)
        label_k = __classify__(svm_k, y_k)
        pl, nl = poslabel(svm_k), neglabel(svm_k)
        i, j = label2ind(pl, encoding), label2ind(nl, encoding)
        confidence[i] += y_k
        confidence[j] -= y_k
        vote[i] += label_k == pl
        vote[j] += label_k == nl
    end
    for (i, x) in enumerate(confidence)
        @inbounds confidence[i] = x / (3 * abs(x) + 1) + vote[i]
    end
    predicted_label = ind2label(argmax(confidence), encoding)
    return predicted_label
end

function __classify__(::OVR, y::AbstractVector, svm, encoding, confidence, vote)
    fill!(confidence, 0)
    @inbounds for (y_k, svm_k) in zip(y, svm)
        pl = poslabel(svm_k)
        i = label2ind(pl, encoding)
        confidence[i] += y_k
    end
    predicted_label = ind2label(argmax(confidence), encoding)
    return predicted_label
end

get_labeled_data(problem::BinarySVMProblem) = (problem.labels, problem.X)
get_labeled_data(problem::MultiSVMProblem) = (problem.labels, problem.data)
