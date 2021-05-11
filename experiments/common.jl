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

make_classifier(::Type{MultiSVMClassifier}, nfeatures, classes) = MultiSVMClassifier(nfeatures, classes)
make_classifier(::Type{BinarySVMClassifier}, nfeatures, classes) = BinarySVMClassifier(nfeatures, Dict(i=>string(c) for (i, c) in enumerate(classes)))

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
