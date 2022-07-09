module SparseSVM
using DataDeps, CSV, DataFrames, CodecZlib
using MLDataUtils
using KernelFunctions, LinearAlgebra
using Random, Statistics, StatsBase, StableRNGs
using Polyester, Parameters
using Printf, ProgressMeter

using DataFrames: copy, copyto!
using ArraysOfArrays: VectorOfVectors

import Base: show, getproperty
import MLDataUtils: poslabel, neglabel, classify
import StatsBase: fit, transform!, reconstruct!

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

const DATA_DIR = joinpath(@__DIR__, "data")

include("simulation.jl")

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

function __mm_init__(algorithm, problem::MultiSVMProblem, ::Nothing)
    return [__mm_init__(algorithm, svm, nothing) for svm in problem.svm]
end

function __mm_init__(algorithm, problem::MultiSVMProblem, extras)
    for (i, svm) in enumerate(problem.svm)
        extras[i] = __mm_init__(algorithm, svm, extras[i])
    end
    return extras
end

const INTERCEPT_INDEX = Val(:last)

const DEFAULT_ANNEALING = geometric_progression
const DEFAULT_CALLBACK = __do_nothing_callback__
const DEFAULT_SCORE_FUNCTION = prediction_errors

const DEFAULT_GTOL = 1e-3
const DEFAULT_DTOL = 1e-3
const DEFAULT_RTOL = 1e-6

include("fit.jl")
include("cv.jl")
include("transform.jl")

export MultiClassStrategy, OVO, OVR
export BinarySVMProblem, MultiSVMProblem
export MMSVD, SD
export ZScoreTransform, NormalizationTransform

end # end module