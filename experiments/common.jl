using SparseSVM, KernelFunctions, Statistics, LinearAlgebra, StatsBase, MLDataUtils
using CSV, DataFrames, Random, StableRNGs
using Logging
using MKL

using SparseSVM: CVStatisticsCallback, RepeatedCVCallback, extract_cv_data

using LaTeXStrings, TexTables

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

# checks causal vs noncausal assuming y is the ground truth
function confusion_matrix_coefficients(x, y)
    TP = FP = TN = FN = 0
    for (xi, yi) in zip(x, y)
        TP += (xi != 0) && (yi != 0)
        FP += (xi != 0) && (yi == 0)
        TN += (xi == 0) && (yi == 0)
        FN += (xi == 0) && (yi != 0)
    end
    return (TP, FP, TN, FN)
end

# checks correct vs incorrect in binary classification
function confusion_matrix_predictions(x, y, pl)
    TP = FP = TN = FN = 0
    for (xi, yi) in zip(x, y)
        TP += (xi == yi) && (yi == pl) # match true label & yi has positive label (TP)
        FP += (xi != yi) && (yi != pl) # different labels & yi has negative label (FP)
        TN += (xi == yi) && (yi != pl) # match true label & yi has negative label (TN)
        FN += (xi != yi) && (yi == pl) # different labels & yi has positive label (FN)
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

##### table utils #####

function summarize_over_folds(df)
    gdf = groupby(df, [:algorithm, :replicate, :lambda, :sparsity])
    replicate_df = combine(gdf,
        :nvars => first => :variables,
        :iters => mean => :iterations,
        :iters => sem => :iterations_se,
        :time => mean => :time,
        :time => sem => :time_se,
        :risk => mean => :risk,
        :risk => sem => :risk_se,
        :gradient => mean => :gradient,
        :gradient => sem => :gradient_se,
        :distance => mean => :distance,
        :distance => sem => :distance_se,
        :norm => (x -> mean(1 ./ x)) => :margin,
        :norm => (x -> sem(1 ./ x)) => :margin_se,
        :nnz => mean => :nnz,
        :nnz => sem => :nnz_se,
        :anz => mean => :anz,
        :anz => sem => :anz_se,
        :nsv => mean => :nsv,
        :nsv => sem => :nsv_se,
        :train => (x -> 100*mean(x)) => :train,
        :train => (x -> 100*sem(x)) => :train_se,
        :validation => (x -> 100*mean(x)) => :validation,
        :validation => (x -> 100*sem(x)) => :validation_se,
        :test => (x -> 100*mean(x)) => :test,
        :test => (x -> 100*sem(x)) => :test_se,
    )
    replicate_df
end

function change_formatting!(col, format::AbstractString)
    for x in col.data.vals
        x.format = format
    end
end

function change_formatting!(col, format::Tuple)
    for x in col.data.vals
        x.format = format[1]
        x.format_se = format[2]
    end
end

function add_column!(cols, header, keys, df, col_index; format="{:.2f}")
    col = TableCol(header, keys, df[!,col_index] |> Vector)
    change_formatting!(col, format)
    push!(cols, col)
end

function add_column_with_se!(cols, header, keys, df, col_index; format="{:.2f}")
    col_index_se = Symbol(col_index, :_se)
    col = TableCol(header, keys, df[!,col_index] |> Vector, df[!,col_index_se] |> Vector)
    change_formatting!(col, format)
    push!(cols, col)
end

function add_grouped_column!(cols, header, subheaders, keys, dfs, col_index; kwargs...)
    tmp = []
    for (subheader, df) in zip(subheaders, dfs)
        add_column!(tmp, subheader, keys, df, col_index; kwargs...)
    end
    push!(cols, join_table(header => hcat(tmp...)))
end

function add_grouped_column_with_se!(cols, header, subheaders, keys, dfs, col_index; kwargs...)
    tmp = []
    for (subheader, df) in zip(subheaders, dfs)
        add_column_with_se!(tmp, subheader, keys, df, col_index; kwargs...)
    end
    push!(cols, join_table(header => hcat(tmp...)))
end
