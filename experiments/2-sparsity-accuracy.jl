include("common.jl")
using SparseArrays

function simulate_data(rng,
        m, n, k,
        effect_size,
        default_sigma_ij, default_sigma_j,
        causal_sigma_ij, causal_sigma_j,
    )
    # Set covariance matrix for predictors.
    Σ = sparse(1.0 * I, n, n)
    # Σ = Matrix{Float64}(default_sigma_j * I, n, n)
    # for j in 1:n, i in j+1:n
    #     Σ[j,i] = default_sigma_ij * randn(rng)
    # end

    # # Modify covariance matrix entries for causal predictors.
    # for j in 1:k
    #     Σ[j,j] = causal_sigma_j
    #     for i in j+1:k
    #         sigma_ij_sign = rand(rng, (-1, 1))
    #         Σ[j,i] = sigma_ij_sign * causal_sigma_ij
    #     end
    # end

    # Simulate predictors, X. No intercept.
    C = cholesky(Symmetric(Σ))
    L = sparse(C.L)
    X = Matrix{Float64}(undef, m, n)
    z = zeros(n)
    for i in axes(X, 1)
        Random.randn!(rng, z)
        @views mul!(X[i, :], L, z)
    end
    _rescale_!(Val(:zscore), X)

    # Set causal coefficients.
    beta = zeros(n)
    for j in 1:k
        beta_sign = rand(rng, (-1, 1))
        beta_effect_size = effect_size[1] + (effect_size[2] - effect_size[1]) * rand(rng)
        beta_effect_size = clamp(beta_effect_size, effect_size[1], effect_size[2])
        beta[j] = beta_sign * beta_effect_size
    end

    # Shuffle predictors
    perm = randperm(rng, n)
    causal_idx = perm[1:k]
    @. X[:, perm] = X[:, perm]
    @. beta[perm] = beta[perm]

    # Simulate the targets; one of two classes.
    y = sign.(X*beta)
    target = map(yi -> yi > 0 ? 'A' : 'B', y)

    return target, X, beta, causal_idx, Σ
end

function run_experiment(fname, algorithm::AlgOption, grid;
        m::Int=100,
        n::Int=50,
        k0::Int=1,
        effect_size::Vector=[1.0, 1.0],
        default_sigma_ij::Real=1e-4,
        default_sigma_j::Real=1.0,
        causal_sigma_ij::Real=default_sigma_ij,
        causal_sigma_j::Real=default_sigma_j,
        proportion_train::Real=0.8,
        tol::Real=1e-6,
        nouter::Int=20,
        ninner::Int=1000,
        mult::Real=1.2,
    )
    # Put options into a list.
    k0 = clamp(k0, 1, n)
    options = (
        :m => m,
        :n => n,
        :k0 => k0,
        :effect_size_min => effect_size[1],
        :effect_size_max => effect_size[2],
        :default_sigma_ij => default_sigma_ij,
        :default_sigma_j => default_sigma_j,
        :causal_sigma_ij => causal_sigma_ij,
        :causal_sigma_j => causal_sigma_j,
        :fname => fname,
        :algorithm => algorithm,
        :proportion_train => proportion_train,
        :tol => tol,
        :nouter => nouter,
        :ninner => ninner,
        :mult => mult,
    )

    # Simulate data.
    rng = StableRNG(2000)
    y, X, beta0, causal_idx, Σ = simulate_data(rng, m, n, k0,
        effect_size,
        default_sigma_ij,
        default_sigma_j,
        causal_sigma_ij,
        causal_sigma_j
    )
    labeled_data = (X, y)

    # Create the train and test data sets.
    (train_X, train_targets), (test_X, test_targets) = splitobs(labeled_data, at=proportion_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)
    gridvals = sort!(unique(grid), rev=false) # iterate from least sparse to most sparse models
    nvals = length(gridvals)

    # Initialize classifier.
    classifier = BinaryClassifier{Float64}(train_X, train_targets, first(train_targets), intercept=false)
    A, Asvd = SparseSVM.get_A_and_SVD(classifier)
    initialize_weights!(classifier, A)
    beta_init = copy(classifier.weights)

    # Open file to write results on disk. Save settings on a separate log.
    dir = joinpath("results", "experiment2")
    open(joinpath(dir, "$(fname).log"), "w+") do io
        for (key, val) in options
            writedlm(io, [key val], '=')
        end
    end
    results = open(joinpath(dir, "$(fname).out"), "w+")
    writedlm(results, ["m" "n" "k0" "sparsity" "sv" "time" "iter" "obj" "dist" "MSE" "TP" "FP" "TN" "FN" "train_acc" "test_acc"])

    function write_result(classifier, s, result)
        # Get timing and convergence data.
        t = result.time
        iters, obj, dist = result.value

        # Check support vectors.
        sv = count_support_vecs(classifier)

        # Test the classifier.
        train_acc = round(accuracy_score(classifier, train_X, train_targets)*100, sigdigits=4)
        test_acc = round(accuracy_score(classifier, test_X, test_targets)*100, sigdigits=4)
        sparsity = round(s*100, sigdigits=4)

        # Check quality of solution against ground truth.
        # How well are the effect sizes estimated?
        beta = classifier.weights
        MSE = mse(beta, beta0)

        # How well are the causal coefficient determined?
        (TP, FP, TN, FN) = discovery_metrics(beta, beta0)

        # Write results to file.
        writedlm(results, Any[
            m n k0 sparsity sv t iters obj dist MSE TP FP TN FN train_acc test_acc 
        ])
        flush(results)
    end

    # Run the algorithm with different sparsity values.
    @showprogress 1 "Testing $(algorithm) on $(m) × $(n) problem... " for s in gridvals
        copyto!(classifier.weights, beta_init)
        result = @timed trainMM!(classifier, A, f, tol, s, fullsvd=Asvd,
            nouter=nouter, ninner=ninner, mult=mult, init=false, verbose=false,
        )
        write_result(classifier, s, result)
    end

    # Close file.
    close(results)
end

# Options
algorithms = (MM, SD)
fnames = map(opt -> generate_filename(2, opt), algorithms)

default_size = 500
p = floor(Int, log10(default_size)) + 1
size_grid = (ceil(Int, 10^(p+u)) for u in range(0, 2, step=0.25))
k0 = SparseSVM.sparsity_to_k(0.9, default_size)
effect_size = [2.0, 10.0]
default_sigma_j = 1.0
default_sigma_ij = 0.0
causal_sigma_j = 1.0
causal_sigma_ij=causal_sigma_ij = 0.0

tol = 1e-6
ninner = 10^4
nouter = 100
mult = 1.2

# Pre-compile
sparsity_grid = [0.0, 0.5]
for opt in algorithms
    m = 100
    n = 100    
    fname = generate_filename(2, opt)
    run_experiment(fname, opt, sparsity_grid,
        m=100,
        n=100,
        k0=k0,
        effect_size=effect_size,
        default_sigma_j=default_sigma_j,
        default_sigma_ij=default_sigma_ij,
        causal_sigma_j=causal_sigma_j,
        causal_sigma_ij=causal_sigma_ij,
        tol=tol,
        ninner=2,
        nouter=2,
        mult=mult,
    )
    rm(joinpath("results", "experiment2", "$(fname).out"))
    rm(joinpath("results", "experiment2", "$(fname).log"))
end

sparsity_grid = [
    range(0.0, 0.9, step=1e-1);
    range(0.91, 0.99, step=2e-2);
    range(0.991, 0.999, step=2e-3);
    range(0.9991, 0.9999, step=2e-4)
]

# Case m > n: More samples than predictors.
for (opt, fname) in zip(algorithms, fnames), m in size_grid
    n = default_size
    run_experiment(fname, opt, sparsity_grid,
        m=m,
        n=n,
        k0=k0,
        effect_size=effect_size,
        default_sigma_j=default_sigma_j,
        default_sigma_ij=default_sigma_ij,
        causal_sigma_j=causal_sigma_j,
        causal_sigma_ij=causal_sigma_ij,
        tol=tol,
        ninner=ninner,
        nouter=nouter,
        mult=mult,
    )
end

# Case: m < n: More predictors than samples.
for (opt, fname) in zip(algorithms, fnames), n in size_grid
    m = default_size
    run_experiment(fname, opt, sparsity_grid,
        m=m,
        n=n,
        k0=k0,
        effect_size=effect_size,
        default_sigma_j=default_sigma_j,
        default_sigma_ij=default_sigma_ij,
        causal_sigma_j=causal_sigma_j,
        causal_sigma_ij=causal_sigma_ij,
        tol=tol,
        ninner=ninner,
        nouter=nouter,
        mult=mult,
    )
end
