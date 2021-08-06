using LIBSVM
import SparseSVM: Classifier, get_support_vecs, count_support_vecs
import LIBSVM: set_params!, fit!

# Binary case: % zeros in coefficients
measure_model_sparsity(weights::AbstractVector) = 100 * count(isequal(0), weights) / length(weights)

function measure_model_sparsity(clf::BinaryClassifier)
    weights = clf.intercept ? clf.weights[1:end-1] : clf.weights
    measure_model_sparsity(weights)
end

# Multi case: average % zeros in coefficients
measure_model_sparsity(weights::AbstractMatrix) = mean(measure_model_sparsity(w) for w in eachcol(weights))

function measure_model_sparsity(clf::MultiClassifier)
    x = zeros(length(clf.svm))
    for (i, svm) in enumerate(clf.svm)
        x[i] = measure_model_sparsity(svm)
    end
    return mean(x)
end

struct LIBSVMClassifier{M} <: SparseSVM.Classifier
    model::M
end

function (classifier::LIBSVMClassifier)(X::AbstractMatrix)
    LIBSVM.predict(classifier.model, X)
end

function get_support_vecs(classifier::LIBSVMClassifier{LinearSVC}, y, X)
    _, d = LIBSVM.LIBLINEAR.linear_predict(classifier.model.fit, X')
    idx = findall(y .* d' .< 1)
    return sort!(idx)
end

measure_model_sparsity(clf::LIBSVMClassifier{LinearSVC}) = measure_model_sparsity(clf.model.fit.w)
measure_model_sparsity(clf::LIBSVMClassifier{SVC}) = measure_model_sparsity(clf.model.fit.coefs)

LIBSVM.set_params!(classifier::LIBSVMClassifier; new_params...) = LIBSVM.set_params!(classifier.model; new_params...)
LIBSVM.fit!(classifier, X, y) = LIBSVM.fit!(classifier.model, X, y)

function LIBSVM_L2(;C::Real=1.0, tol::Real=1e-4, intercept::Bool=false, kwargs...)
    clf = LIBSVMClassifier(LinearSVC(;
        solver=LIBSVM.Linearsolver.L2R_L2LOSS_SVC,
        cost=C,
        tolerance=tol,
        bias= intercept ? 1.0 : 0.0,
        kwargs...
    ))
    return clf
end

function LIBSVM_L1(;C::Real=1.0, tol::Real=1e-4, intercept::Bool=false, kwargs...)
    clf = LIBSVMClassifier(LinearSVC(;
        solver=LIBSVM.Linearsolver.L1R_L2LOSS_SVC,
        cost=C,
        tolerance=tol,
        bias= intercept ? 1.0 : 0.0,
        kwargs...
    ))
    return clf
end

function LIBSVM_RB(;C::Real=1.0, tol::Real=1e-4, intercept::Bool=false, kwargs...)
    clf = LIBSVMClassifier(SVC(;
        kernel=LIBSVM.Kernel.RadialBasis,
        cost=C,
        tolerance=tol,
        kwargs...
    ))
    return clf
end
