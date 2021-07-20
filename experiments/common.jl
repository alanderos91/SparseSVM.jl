using SparseSVM, MLDataUtils, KernelFunctions, LinearAlgebra
using CSV, DataFrames, Random, StableRNGs, Statistics
using Dates, DelimitedFiles, ProgressMeter

##### Make sure we set up BLAS threads correctly #####
BLAS.set_num_threads(8)

##### handle filenames #####
function generate_filename(experiment, algorithm)
    return string(experiment,
        "-", Dates.format(now(), dateformat"yyyymmdd-HHMMSS"),
        "-algorithm=", string(algorithm),
    )
end

##### Define special type for selecting different algorithms #####
@enum AlgOption SD=1 MM=2

function get_algorithm_func(opt::AlgOption)
    if opt == SD
        return sparse_steepest!
    elseif opt == MM
        return sparse_direct!
    else
        error("Unknown option $(opt)")
    end
end

##### wrappers for creating a classifier object #####
make_classifier(::Type{C}, X, targets, refclass; kwargs...) where C<:MultiClassifier = C(X, targets; kwargs...)
make_classifier(::Type{C}, X, targets, refclass; strategy::MultiClassStrategy=OVR(), kwargs...) where C<:BinaryClassifier = C(X, targets, refclass; kwargs...)

##### initializing model parameters #####
function initialize_weights!(classifier::BinaryClassifier, A::AbstractMatrix)
    y = classifier.data.y
    weights = classifier.weights
    intercept = classifier.intercept
    SparseSVM._init_weights_!(weights, A, y, intercept)

    # sanity checks
    if any(isnan, weights)
        error("Detected NaNs in computing univariate initial guess.")
    end
    
    return nothing
end

function initialize_weights!(classifier::MultiClassifier, A::Vector)
    svm = classifier.svm
    for i in eachindex(svm)
        initialize_weights!(svm[i], A[i])
    end
    return nothing
end

function randomize_weights!(classifier::BinaryClassifier, rng)
    Random.randn!(rng, classifier.weights)
    return nothing
end

function randomize_weights!(classifier::MultiClassifier, rng)
    foreach(svm -> randomize_weights!(svm, rng), classifier.svm)
end

##### performance metrics #####

function accuracy_score(classifier, X, targets)
    predictions = classifier.(eachrow(X))
    n = length(predictions)
    return sum(predictions .== targets) / n
end

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

##### standardization #####

function _rescale_!(::Val{:none}, X)
    X
end

function _rescale_!(::Val{:zscore}, X) # [-1, 1]
    rescale!(X, obsdim=1)
end

function _rescale_!(::Val{:minmax}, X) # [0, 1]
    xmin = minimum(X, dims=2)
    xmax = maximum(X, dims=2)
    @. X = (X - xmin) / (xmax - xmin)
end