const OPTIONS = Dict(
    ##### Example 1: synthetic #####
    "synthetic" => (
        BinaryClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        scale=:zscore,
        proportion_train=0.8,
        ),
    ),
    ##### Example 2: iris #####
    "iris" => (
        MultiClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        strategy=OVO(),
        scale=:zscore,
        proportion_train=0.8,
        ),
    ),
    ##### Example 3: spiral #####
    "spiral" => (
        MultiClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=RBFKernel(),
        strategy=OVO(),
        scale=:zscore,
        proportion_train=0.8,
        ),
    ),
    ##### Example 4: letter-recognition #####
    "letter-recognition" => (
        MultiClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        strategy=OVO(),
        scale=:minmax,
        proportion_train=0.8,
        ),
    ),
    ##### Example 5: breast-cancer-wisconsin #####
    "breast-cancer-wisconsin" => (
        BinaryClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        strategy=OVO(),
        scale=:none,
        proportion_train=0.8,
        ),
    ),
    ##### Example 6: splice #####
    "splice" => (
        MultiClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        strategy=OVO(),
        scale=:minmax,
        proportion_train=0.8,
        ),
    ),
    ##### Example 7: TCGA-PANCAN-HiSeq #####
    "TCGA-PANCAN-HiSeq" => (
        MultiClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        strategy=OVO(),
        scale=:minmax,
        proportion_train=0.8,
        ),
    ),
    ##### Example 8: optdigits #####
    "optdigits" => (
        MultiClassifier{Float64},
        (tol=1e-6,
        ninner=10^4,
        nouter=100,
        mult=1.1,
        intercept=true,
        kernel=nothing,
        strategy=OVO(),
        scale=:none,
        proportion_train=0.68,
        ),
    ),
)

const EXAMPLES = collect(keys(OPTIONS))

const SPARSITY_GRID = Dict(
    "synthetic" => [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ],
    "iris" => [0.0, 0.25, 0.5, 0.75],
    "spiral" => [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ],
    "letter-recognition" => [i/16 for i in 0:15],
    "breast-cancer-wisconsin" => [i/10 for i in 0:9],
    "splice" => [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ],
    "TCGA-PANCAN-HiSeq" => [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ],
    "optdigits" => [
        range(0.0, 0.9; length=5);
        range(0.91, 0.99; length=5);
        range(0.991, 0.999; length=5);
    ],
)

const MARGIN_GRID = Dict(
    "synthetic" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ],
    "iris" => [1.0, 1e-1, 1e-2, 1e-3],
    "spiral" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ],
    "letter-recognition" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
        1e-45],
    "breast-cancer-wisconsin" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
    ],
    "splice" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ],
    "TCGA-PANCAN-HiSeq" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ],
    "optdigits" => [
        range(1.0, 0.1, length=5);
        range(0.09, 0.01, length=5);
        range(0.009, 0.001, length=5);
    ],
)
