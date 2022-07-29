module SparseSVM
using DataDeps, CSV, DataFrames, CodecZlib
using MLDataUtils
using KernelFunctions, LinearAlgebra, SparseArrays
using Random, Statistics, StatsBase, StableRNGs
using Polyester, Parameters
using Printf, ProgressMeter

using DataFrames: copy, copyto!
using ArraysOfArrays: VectorOfVectors

import Base: show, getproperty
import LIBSVM
import MLDataUtils: poslabel, neglabel, classify
import StatsBase: fit, transform!, reconstruct!

##### DATA #####

#=
Uses DataDeps to download data as needed.
Inspired by UCIData.jl: https://github.com/JackDunnNZ/UCIData.jl
=#

##### DATADEP REGISTRATION #####

const DATADEPNAME = "SparseSVM"
const DATA_DIR = joinpath(@__DIR__, "data")
const MESSAGES = Ref(String[])
const REMOTE_PATHS = Ref([])
const CHECKSUMS = Ref([])
const FETCH_METHODS = Ref([])
const POST_FETCH_METHODS = Ref([])
const DATASETS = Ref(String[])

include("simulation.jl")
include("datadeps.jl")

function __init__()
    # Delete README.md in data/.
    readme = joinpath(DATA_DIR, "README.md")
    if ispath(readme)
        rm(readme)
    end
    
    # Add arguments to Refs.
    for dataset_jl in readdir(DATA_DIR)
        include(joinpath(DATA_DIR, dataset_jl))
    end

    # Compile a help message from each dataset.
    # Save the output of the help message in DATA_DIR.
    readme_content = """
    # SparseSVM Examples
    
    You can load an example by invoking `SparseSVM.dataset(name)`.
    The list of available datasets is accessible via `SparseSVM.list_datasets()`.

    Please note that the descriptions here are *very* brief summaries. Follow the links for additional information.

    $(join(MESSAGES[], '\n')) 
    """

    open(readme, "w") do io
        write(io, readme_content)
    end

    # Register the DataDep as SparseSVM.
    register(DataDep(
        DATADEPNAME,
        """
        Welcome to the SparseSVM installation.
    
        This program will now attempt to

            (1) download a few datasets from the UCI Machine Learning Repository, and
            (2) simulate additional synthetic datasets.
        
        Please see $(readme) for a preview of each example.
        """,
        REMOTE_PATHS[],
        CHECKSUMS[];
        fetch_method=FETCH_METHODS[],
        post_fetch_method=POST_FETCH_METHODS[],
    ))

    # Trigger the download process.
    @datadep_str(DATADEPNAME)
end

##### END DATADEP REGISTRATION #####

include("problem.jl")
include("transform.jl")
include("utilities.jl")
include("projections.jl")
include("callbacks.jl")

abstract type AbstractAlgorithm end
abstract type AbstractMMAlgorithm <: AbstractAlgorithm end

include(joinpath("algorithms", "SD.jl"))
include(joinpath("algorithms", "MMSVD.jl"))
include(joinpath("algorithms", "libsvm.jl"))

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

const DEFAULT_ANNEALING = geometric_progression(1.2) # rho_0 * 1.2^t
const DEFAULT_CALLBACK = __do_nothing_callback__
const DEFAULT_SCORE_FUNCTION = prediction_errors

const DEFAULT_GTOL = 1e-3
const DEFAULT_DTOL = 1e-3
const DEFAULT_RTOL = 1e-6

include("fit.jl")
include("cv.jl")

export MultiClassStrategy, OVO, OVR
export BinarySVMProblem, MultiSVMProblem
export MMSVD, SD, L2SVM, L1SVM
export ZScoreTransform, NormalizationTransform, NoTransformation

end # end module