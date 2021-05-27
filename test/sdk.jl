using SparseSVM, CSV, Statistics, Random, DataFrames, KernelFunctions
using Plots
gr(dpi=300)
include("experiments/load_data.jl")
include("experiments/common.jl")

println("[TEST: synthetic] ======================================\n")

df, X, classes = load_data("synthetic", seed=1234, intercept=false)
class_mapping = Dict(c => i for (i, c) in enumerate(classes))
y = Vector{Float64}(df.Class)
nsamples, nfeatures = size(X)

# Process options
f = sparse_sdk!
k = round(Int, 0.1 * nsamples)

# Create the train and test data sets.
ntrain = round(Int, 0.8 * nsamples)
train_set = 1:ntrain
test_set = ntrain+1:nsamples
Xtrain, Xtest = X[train_set, :], X[test_set, :]
ytrain, ytest = y[train_set], y[test_set]

κ = RBFKernel()
Ktrain = kernelmatrix(κ, Xtrain, obsdim=1)

# Training
classifier = make_classifier(BinarySVMClassifier, length(ytrain), classes)
@timed trainMM(classifier, f, ytrain, Ktrain, 1e-6, k, nouter=20, ninner=10^4, mult=1.5, has_intercept=false)

# Testing
v = map(x -> classify(classifier, κ, Xtrain, ytrain, x), eachrow(Xtest))
score = sum(v .== ytest)
str = "Score = $(score) / $(length(ytest)) ($(100 * score / length(ytest)) %)"
println()
println("$str\n")

# visualize
svs = support_vector_idx(classifier)
f1 = 1
f2 = 2
plot(xlabel="feature $f1", ylabel="feature $f2", title=str, legend=:outerright)
for class in unique(y)
    subset = findall(isequal(class), y)
    scatter!(X[subset, f1], X[subset, f2], label="class = $class")
end
scatter!(X[svs, f1], X[svs, f2], label="support vectors")
savefig("experiments/test_sdk_synthetic.png")

println("[TEST: spiral300] ======================================\n")

df, X, classes = load_data("spiral300", seed=1234, intercept=false)
class_mapping = Dict(c => i for (i, c) in enumerate(classes))
y = Vector{Float64}(df.Class)
nsamples, nfeatures = size(X)

# Process options
f = sparse_sdk!
k = 40

# Create the train and test data sets.
ntrain = round(Int, 0.8 * nsamples)
train_set = 1:ntrain
test_set = ntrain+1:nsamples
Xtrain, Xtest = X[train_set, :], X[test_set, :]
ytrain, ytest = y[train_set], y[test_set]

κ = RBFKernel()
Ktrain = kernelmatrix(κ, Xtrain, obsdim=1)

# Training
classifier = make_classifier(BinarySVMClassifier, length(ytrain), classes)
@timed trainMM(classifier, f, ytrain, Ktrain, 1e-6, k, nouter=50, ninner=10^4, mult=1.5, has_intercept=false)

# Testing
v = map(x -> classify(classifier, κ, Xtrain, ytrain, x), eachrow(Xtest))
score = sum(v .== ytest)
str = "Score = $(score) / $(length(ytest)) ($(100 * score / length(ytest)) %)"
println()
println("$str\n")

# visualize
svs = support_vector_idx(classifier)
f1 = 1
f2 = 2
plot(xlabel="feature $f1", ylabel="feature $f2", title=str, legend=:outerright)
for class in unique(y)
    subset = findall(isequal(class), y)
    scatter!(X[subset, f1], X[subset, f2], label="class = $class")
end
scatter!(X[svs, f1], X[svs, f2], label="support vectors")
savefig("experiments/test_sdk_spiral300.png")

println("[TEST: iris] ======================================")

df, X, classes = load_data("iris", seed=1234, intercept=false)
class_mapping = Dict(c => i for (i, c) in enumerate(classes))
y = [class_mapping[c] for c in df.Class]
nsamples, nfeatures = size(X)

# Process options
f = sparse_sdk!
k = 10

# Create the train and test data sets.
ntrain = round(Int, 0.8 * nsamples)
train_set = 1:ntrain
test_set = ntrain+1:nsamples
Xtrain, Xtest = X[train_set, :], X[test_set, :]
ytrain, ytest = y[train_set], y[test_set]

κ = RBFKernel()
Ktrain = kernelmatrix(κ, Xtrain, obsdim=1)

# Training
classifier = make_classifier(MultiSVMClassifier, ytrain, classes)
@timed trainMM(classifier, f, ytrain, Ktrain, 1e-6, k, nouter=50, ninner=10^4, mult=1.5, has_intercept=false)

# Testing
v = map(x -> classify(classifier, κ, Xtrain, ytrain, x), eachrow(Xtest))
score = sum(v .== ytest)
str = "Score = $(score) / $(length(ytest)) ($(100 * score / length(ytest)) %)"
println()
println("$str\n")

# visualize
svs = support_vector_idx(classifier)
f1 = 1
f2 = 2
plot(xlabel="feature $f1", ylabel="feature $f2", title=str, legend=:outerright)
for class in unique(y)
    subset = findall(isequal(class), y)
    scatter!(X[subset, f1], X[subset, f2], label="class = $class")
end
scatter!(X[svs, f1], X[svs, f2], label="support vectors")
savefig("experiments/test_sdk_iris.png")
