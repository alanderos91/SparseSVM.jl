using SparseSVM, KernelFunctions, Statistics, LinearAlgebra, StatsBase, MLDataUtils
using CSV, DataFrames, Random, StableRNGs
using Logging
# using MKL

using SparseSVM: CVStatisticsCallback, RepeatedCVCallback, extract_cv_data

function create_classifier(::Type{BinarySVMProblem}, labeled_data, svm_kwargs)
    L, X = labeled_data
    positive_label = unique(L) |> sort |> first |> string
    problem = BinarySVMProblem(string.(L), X, positive_label; svm_kwargs...)
    foreach(arr -> fill!(arr, 0), (problem.coeff, problem.coeff_prev, problem.proj))
    return problem
end

function create_classifier(::Type{MultiSVMProblem}, labeled_data, svm_kwargs)
    L, X = labeled_data
    problem = MultiSVMProblem(string.(L), X; svm_kwargs...)
    for svm in problem.svm
        foreach(arr -> fill!(arr, 0), (svm.coeff, svm.coeff_prev, svm.proj))
    end
    return problem
end

# See: https://github.com/JuliaLang/julia/issues/27574#issuecomment-397838647
function dropnames(namedtuple::NamedTuple, names::Tuple{Vararg{Symbol}}) 
    keepnames = Base.diff_names(Base._nt_names(namedtuple), names)
   return NamedTuple{keepnames}(namedtuple)
end

##### Make sure we set up BLAS threads correctly #####
BLAS.set_num_threads(10)

##### performance metrics #####

prediction_accuracy(problem, L, X) = mean(SparseSVM.classify(problem, X) .== L)

mse(x, y) = mean( (x - y) .^ 2 )

function discovery_metrics(x, y)
    TP = FP = TN = FN = 0
    for (xi, yi) in zip(x, y)
        TP += (xi != 0) && (yi != 0)
        FP += (xi != 0) && (yi == 0)
        TN += (xi == 0) && (yi == 0)
        FN += (xi == 0) && (yi != 0)
    end
    return (TP, FP, TN, FN)
end

##### misc #####
nonzero_coeff_indices(arr::AbstractVector) = findall(!iszero, arr) |> sort!

function nonzero_coeff_indices(arr::AbstractMatrix)
    idx = Int[]
    foreach(coeff -> union!(idx, nonzero_coeff_indices(coeff)), eachcol(arr))
    sort!(idx)
end

function nonzero_coeff_indices(arr::Union{SparseSVM.VectorOfVectors,AbstractVector{<:AbstractVector}})
    idx = Int[]
    foreach(coeff -> union!(idx, nonzero_coeff_indices(coeff)), arr)
    sort!(idx)
end

extract_reduced_subset(problem) = extract_reduced_subset(problem.kernel, problem)

function extract_reduced_subset(::Nothing, problem)
    _, coeffs = SparseSVM.get_params_proj(problem)
    return nonzero_coeff_indices(coeffs)
end

function extract_reduced_subset(::Kernel, problem)
    return SparseSVM.support_vectors(problem)
end

match_problem_dimensions(problem, data, idx, training_data) = match_problem_dimensions(problem.kernel, data, idx, training_data)

function match_problem_dimensions(::Nothing, data, idx, training_data)
    L, X = data
    return (L, X[:, idx])
end

function match_problem_dimensions(::Kernel, data, idx, training_data)
    L, X = data
    if training_data
        return L[idx], X[idx, :]
    else
        L, X
    end
end

function extract_iters_and_stats(result::Tuple)
    result[1], result[2]
end

function extract_iters_and_stats(result::Vector)
    a = mean(first, result)
    b = (;
        risk=mean(x -> last(x).risk, result),
        loss=mean(x -> last(x).loss, result),
        objective=mean(x -> last(x).objective, result),
        distance=mean(x -> last(x).distance, result),
        gradient=mean(x -> last(x).gradient, result),
        norm=mean(x -> last(x).norm, result),
    )

    return a, b
end
