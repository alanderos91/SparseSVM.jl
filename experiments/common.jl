# Algorithm Option Definitions
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

make_classifier(::Type{C}, X, targets, refclass; kwargs...) where C<:MultiClassifier = C(X, targets; kwargs...)
make_classifier(::Type{C}, X, targets, refclass; kwargs...) where C<:BinaryClassifier = C(X, targets, refclass; kwargs...)

initialize_weights!(local_rng, classifier::BinaryClassifier) = Random.randn!(local_rng, classifier.weights)
initialize_weights!(local_rng, classifier::MultiClassifier) = foreach(svm -> initialize_weights!(local_rng, svm), classifier.svm)

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

function support_vector_idx(classifier::BinaryClassifier)
    α = classifier.weights
    return findall(x -> x != 0, α)
end

function support_vector_idx(classifier::MultiClassifier)
    subset = Int[]
    for svm in classifier.svm
        s = support_vector_idx(svm)
        union!(subset, s)
    end
    return sort!(subset)
end

