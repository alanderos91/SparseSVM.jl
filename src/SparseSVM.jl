module SparseSVM
using DataFrames: copy, copyto!
using DataDeps, CSV, DataFrames, CodecZlib
using MLDataUtils
using KernelFunctions, LinearAlgebra, Random, Statistics, StatsBase
using Polyester, Parameters
using Printf

import Base: show
import MLDataUtils: poslabel, neglabel, classify

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

const DATA_DIR = joinpath(@__DIR__, "data")

"""
`list_datasets()`

List available datasets in SparseSVM.
"""
list_datasets() = map(x -> splitext(x)[1], readdir(DATA_DIR))

function __init__()
    for dataset in list_datasets()
        include(joinpath(DATA_DIR, dataset * ".jl"))
    end
end

"""
`dataset(str)`

Load a dataset named `str`, if available. Returns data as a `DataFrame` where
the first column contains labels/targets and the remaining columns correspond to
distinct features.
"""
function dataset(str)
    # Locate dataset file.
    dataset_path = @datadep_str str
    file = readdir(dataset_path)
    index = findfirst(x -> occursin("data.", x), file)
    if index isa Int
        dataset_file = joinpath(dataset_path, file[index])
    else # is this unreachable?
        error("Failed to locate a data.* file in $(dataset_path)")
    end
    
    # Read dataset file as a DataFrame.
    df = if splitext(dataset_file)[2] == ".csv"
        CSV.read(dataset_file, DataFrame)
    else # assume .csv.gz
        open(GzipDecompressorStream, dataset_file, "r") do stream
            CSV.read(stream, DataFrame)
        end
    end
    return df
end

function process_dataset(path::AbstractString; header=false, missingstrings="", kwargs...)
    input_df = CSV.read(path, DataFrame, header=header, missingstrings=missingstrings)
    process_dataset(input_df; kwargs...)
    rm(path)
end

function process_dataset(input_df::DataFrame;
    target_index=-1,
    feature_indices=1:0,
    ext=".csv")
    # Build output DataFrame.
    output_df = DataFrame()
    output_df.target = input_df[!, target_index]
    output_df = hcat(output_df, input_df[!, feature_indices], makeunique=true)
    output_cols = [ :target; [Symbol("x", n) for n in eachindex(feature_indices)] ]
    rename!(output_df, output_cols)
    dropmissing!(output_df)
    
    # Write to disk.
    output_path = "data" * ext
    if ext == ".csv"
        CSV.write(output_path, output_df, delim=',', writeheader=true)
    elseif ext == ".csv.gz"
        open(GzipCompressorStream, output_path, "w") do stream
            CSV.write(stream, output_df, delim=",", writeheader=true)
        end
    else
        error("Unknown file extension option '$(ext)'")
    end
end

include("problem.jl")
include("utilities.jl")
include("projections.jl")

abstract type AbstractMMAlg end

include(joinpath("algorithms", "SD.jl"))
include(joinpath("algorithms", "MMSVD.jl"))

const DEFAULT_ANNEALING = geometric_progression
const DEFAULT_CALLBACK = __do_nothing_callback__
# const DEFAULT_SCORE_FUNCTION = prediction_error

const DEFAULT_GTOL = 1e-3
const DEFAULT_DTOL = 1e-3
const DEFAULT_RTOL = 1e-6

"""
    fit(algorithm, problem, s; kwargs...)

Solve optimization problem at sparsity level `s`.

The solution is obtained via a proximal distance `algorithm` that gradually anneals parameter estimates
toward the target sparsity set.
"""
function fit(algorithm::AbstractMMAlg, problem::BinarySVMProblem, s::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing) # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, s, extras, (true,false,); kwargs...)
end

function fit(algorithm::AbstractMMAlg, problem::MultiSVMProblem, s::Real; kwargs...)
    extras = [__mm_init__(algorithm, svm, nothing) for svm in problem.svm] # initialize extra data structures
    SparseSVM.fit!(algorithm, problem, s, extras, (true,false,); kwargs...)
end

"""
    fit!(algorithm, problem, s, [extras], [update_extras]; kwargs...)

Same as `fit(algorithm, problem, s)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `ρ` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `dist < dtol || abs(dist - old) < rtol * (1 + old)`, where `dist` is the distance and `dtol` and `rtol` are tolerance parameters.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `nouter`: The number of outer iterations; i.e. the maximum number of `ρ` values to use in annealing (default=`100`).
- `dtol`: An absolute tolerance parameter for the squared distance (default=`1e-6`).
- `rtol`: A relative tolerance parameter for the squared distance (default=`1e-6`).
- `rho_init`: The initial value for `ρ` (default=1.0).
- `rho_max`: The maximum value for `ρ` (default=1e8).
- `rhof`: A function `rhof(ρ, iter, rho_max)` used to determine the next value for `ρ` in the annealing sequence. The default multiplies `ρ` by `1.2`.
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.

See also: [`SparseSVM.anneal!`](@ref) for additional keyword arguments applied at the annealing step.
"""
function fit!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, s::Real,
    extras=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    nouter::Int=100,
    dtol::Real=DEFAULT_DTOL,
    rtol::Real=DEFAULT_RTOL,
    rho_init::Real=1.0,
    rho_max::Real=1e8,
    rhof::Function=DEFAULT_ANNEALING,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    @unpack projection = extras
    
    # Fix model size(s).
    k = sparsity_to_k(problem, s)

    # Initialize ρ and iteration count.
    ρ, iters = rho_init, 0

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, ρ, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, ρ, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    init_result = __evaluate_objective__(problem, ρ, extras)
    result = SubproblemResult(0, init_result)
    cb(0, problem, ρ, k, result)
    old = sqrt(result.distance)

    for iter in 1:nouter
        # Solve minimization problem for fixed rho.
        verbose && print("\n",iter,"  ρ = ",ρ)
        result = SparseSVM.anneal!(algorithm, problem, ρ, s, extras, (false,true,); verbose=verbose, cb=cb, kwargs...)

        # Update total iteration count.
        iters += result.iters

        cb(iter, problem, ρ, k, result)

        # Check for convergence to constrained solution.
        dist = sqrt(result.distance)
        if dist < dtol || abs(dist - old) < rtol * (1 + old)
            break
        else
          old = dist
        end
                
        # Update according to annealing schedule.
        ρ = ifelse(iter < nouter, rhof(ρ, iter, rho_max), ρ)
    end
    
    # Project solution to the constraint set.
    apply_projection(projection, problem, k)
    loss, obj, dist, gradsq = __evaluate_objective__(problem, ρ, extras)

    if verbose
        print("\n\niters = ", iters)
        print("\n1/n ∑ᵢ max{0, 1 - yᵢ xᵢᵀ β}² = ", loss)
        print("\nobjective  = ", obj)
        print("\ndistance   = ", sqrt(dist))
        println("\n|gradient| = ", sqrt(gradsq))
    end

    return SubproblemResult(iters, loss, obj, dist, gradsq)
end

function fit!(algorithm::AbstractMMAlg, problem::MultiSVMProblem, s::Real,
    extras=nothing,
    update_extras::NTuple{2,Bool}=(true,false,);
    kwargs...)
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end
    
    n = length(problem.svm)
    total_iter, total_loss, total_objv, total_dist, total_grad = 0, 0.0, 0.0, 0.0, 0.0

    # Create closure to fit a particular SVM.
    function __fit__!(k)
        subproblem = problem.svm[k]
        r = SparseSVM.fit!(algorithm, subproblem, s, extras[k], update_extras; kwargs...)
        i, l, o, d, g = r
        total_iter += i
        total_loss += l
        total_objv += o
        total_dist += d
        total_grad += g
        return r
    end

    # Fit each SVM to build the classifier.
    result = __fit__!(1)
    results = [result]
    for k in 2:n
        result = __fit__!(k)
        push!(results, result)
    end
    total = SubproblemResult(total_iter, IterationResult(total_loss, total_objv, total_dist, total_grad))

    return (; total=total, result=results)
end

"""
    anneal(algorithm, problem, ρ, s; kwargs...)

Solve the `ρ`-penalized optimization problem at sparsity level `s`.
"""
function anneal(algorithm::AbstractMMAlg, problem::BinarySVMProblem, ρ::Real, s::Real; kwargs...)
    extras = __mm_init__(algorithm, problem, nothing)
    SparseSVM.anneal!(algorithm, problem, ρ, s, extras, (true,true,); kwargs...)
end

"""
    anneal!(algorithm, problem, ρ, s, [extras], [update_extras]; kwargs...)

Same as `anneal(algorithm, problem, ρ, s)`, but with preallocated data structures in `extras`.

!!! Note
    The caller should specify whether to update data structures depending on `s` and `ρ` using `update_extras[1]` and `update_extras[2]`, respectively.

    Convergence is determined based on the rule `grad < gtol`, where `grad` is the Euclidean norm of the gradient and `gtol` is a tolerance parameter.

!!! Tip
    The `extras` argument can be constructed using `extras = __mm_init__(algorithm, problem, nothing)`.

# Keyword Arguments

- `ninner`: The maximum number of iterations (default=`10^4`).
- `gtol`: An absoluate tolerance parameter on the squared Euclidean norm of the gradient (default=`1e-6`).
- `nesterov_threshold`: The number of early iterations before applying Nesterov acceleration (default=`10`).
- `verbose`: Print convergence information (default=`false`).
- `cb`: A callback function for extending functionality.
"""
function anneal!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, ρ::Real, s::Real,
    extras=nothing,
    update_extras::NTuple{2,Bool}=(true,true);
    ninner::Int=10^4,
    gtol::Real=DEFAULT_GTOL,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    cb::Function=DEFAULT_CALLBACK,
    kwargs...
    )
    # Check for missing data structures.
    if extras isa Nothing
        error("Detected missing data structures for algorithm $(algorithm).")
    end

    # Get problem info and extra data structures.
    @unpack intercept, coeff, coeff_prev, proj = problem
    @unpack projection = extras

    # Fix model size(s).
    k = sparsity_to_k(problem, s)

    # Update data structures due to hyperparameters.
    update_extras[1] && __mm_update_sparsity__(algorithm, problem, ρ, k, extras)
    update_extras[2] && __mm_update_rho__(algorithm, problem, ρ, k, extras)

    # Check initial values for loss, objective, distance, and norm of gradient.
    apply_projection(projection, problem, k)
    result = __evaluate_objective__(problem, ρ, extras)
    cb(0, problem, ρ, k, result)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Use previous estimates in case of warm start.
    copyto!(coeff, coeff_prev)

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "distance", "|gradient|")
    for iter in 1:ninner
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __mm_iterate__(algorithm, problem, ρ, k, extras)

        # Update loss, objective, distance, and gradient.
        apply_projection(projection, problem, k)
        result = __evaluate_objective__(problem, ρ, extras)

        cb(iter, problem, ρ, k, result)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, sqrt(result.distance), sqrt(result.gradient))
        end

        # Assess convergence.
        obj = result.objective
        if sqrt(result.gradient) < gtol
            break
        elseif iter < ninner
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end
    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev, coeff)

    return SubproblemResult(iters, result)
end

"""
```init!(algorithm, problem, ϵ, λ, [_extras_]; [maxiter=10^3], [gtol=1e-6], [nesterov_threshold=10], [verbose=false])```

Initialize a `problem` with its `λ`-regularized solution.
"""
function init!(algorithm::AbstractMMAlg, problem::BinarySVMProblem, λ, _extras_=nothing;
    maxiter::Int=10^3,
    gtol::Real=DEFAULT_GTOL,
    nesterov_threshold::Int=10,
    verbose::Bool=false,
    )
    # Check for missing data structures.
    extras = __mm_init__(algorithm, problem, _extras_)

    # Get problem info and extra data structures.
    @unpack coeff, coeff_prev, proj = problem

    # Update data structures due to hyperparameters.
    __mm_update_lambda__(algorithm, problem, λ, extras)

    # Initialize coefficients.
    randn!(coeff)
    copyto!(coeff_prev, coeff)

    # Check initial values for loss, objective, distance, and norm of gradient.
    result = __evaluate_reg_objective__(problem, λ, extras)
    old = result.objective

    if sqrt(result.gradient) < gtol
        return SubproblemResult(0, result)
    end

    # Initialize iteration counts.
    iters = 0
    nesterov_iter = 1
    verbose && @printf("\n%-5s\t%-8s\t%-8s\t%-8s", "iter.", "loss", "objective", "|gradient|")
    for iter in 1:maxiter
        iters += 1

        # Apply the algorithm map to minimize the quadratic surrogate.
        __reg_iterate__(algorithm, problem, λ, extras)

        # Update loss, objective, and gradient.
        result = __evaluate_reg_objective__(problem, λ, extras)

        if verbose
            @printf("\n%4d\t%4.3e\t%4.3e\t%4.3e", iter, result.loss, result.objective, sqrt(result.gradient))
        end

        # Assess convergence.
        obj = result.objective
        if sqrt(result.gradient) < gtol
            break
        elseif iter < maxiter
            needs_reset = iter < nesterov_threshold || obj > old
            nesterov_iter = __apply_nesterov__!(coeff, coeff_prev, nesterov_iter, needs_reset)
            old = obj
        end
    end
    if verbose print("\n\n") end

    # Save parameter estimates in case of warm start.
    copyto!(coeff_prev, coeff)
    copyto!(proj, coeff)

    return SubproblemResult(iters, result)
end

function init!(algorithm::AbstractMMAlg, problem::MultiSVMProblem, λ, extras=nothing; kwargs...)
    n = length(problem.svm)
    total_iter, total_loss, total_objv, total_dist, total_grad = 0, 0.0, 0.0, 0.0, 0.0

    # Create closure to fit a particular SVM.
    function __init__!(k)
        subproblem = problem.svm[k]
        r = SparseSVM.init!(algorithm, subproblem, λ, extras; kwargs...)
        i, l, o, d, g = r
        total_iter += i
        total_loss += l
        total_objv += o
        total_dist += d
        total_grad += g
        return r
    end

    # Fit each SVM to build the classifier.
    result = __init__!(1)
    results = [result]
    for k in 2:n
        result = __init__!(k)
        push!(results, result)
    end
    total = SubproblemResult(total_iter, IterationResult(total_loss, total_objv, total_dist, total_grad))

    return (; total=total, result=results)
end

export MultiClassStrategy, OVO, OVR
export BinarySVMProblem, MultiSVMProblem
export MMSVD, SD

end # end module