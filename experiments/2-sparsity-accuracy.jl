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

function run_experiment(fname, algorithm::AlgOption, y, X, beta0, grid;
        proportion_train::Real=0.8,
        tol::Real=1e-6,
        nouter::Int=20,
        ninner::Int=1000,
        mult::Real=1.2,
    )
    # Create the train and test data sets.
    labeled_data = (X, y)
    (train_X, train_targets), (test_X, test_targets) = splitobs(labeled_data, at=proportion_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)
    gridvals = sort!(unique(grid), rev=false) # iterate from least sparse to most sparse models
    nvals = length(gridvals)

    # Initialize classifier.
    init_cost = @timed begin
        classifier = BinaryClassifier{Float64}(train_X, train_targets, first(train_targets), intercept=false)
        A, Asvd = SparseSVM.get_A_and_SVD(classifier)
        initialize_weights!(classifier, A)
        beta_init = copy(classifier.weights)
    end

    # Open file to write results on disk.
    results = open(joinpath(dir, "$(fname).out"), "a+")
    function write_result(classifier, s, result)
        # Get timing and convergence data.
        t = result.time + init_cost.time
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
    m, n = size(X)
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
proportion_train = 0.8

default_size = 500
p = floor(Int, log10(default_size)) + 1
size_grid = (ceil(Int, 10^(p+u)) for u in range(0, 2, step=0.25))
k0 = SparseSVM.sparsity_to_k(0.9, default_size)
effect_size = [2.0, 10.0]
default_sigma_j = 1.0
default_sigma_ij = 0.0
causal_sigma_j = 1.0
causal_sigma_ij=causal_sigma_ij = 0.0

seed = 2000
simulate(m, n) = simulate_data(StableRNG(seed), m, n, k0,
    effect_size,
    default_sigma_ij,
    default_sigma_j,
    causal_sigma_ij,
    causal_sigma_j
)

tol = 1e-6
ninner = 10^4
nouter = 100
mult = 1.2

# Initialize directories.
dir = joinpath("results", "experiment2")
!isdir("results") && mkdir("results")
!isdir(dir) && mkdir(dir)

# Pre-compile
sparsity_grid = [0.0, 0.5]
for opt in algorithms
    m = 100
    n = 100    
    fname = generate_filename(2, opt)
    y, X, beta0, _, _ = simulate(m, n)
    run_experiment(fname, opt, y, X, beta0, sparsity_grid,
        tol=tol,
        ninner=2,
        nouter=2,
        mult=mult,
        proportion_train=proportion_train,
    )
    rm(joinpath(dir, "$(fname).out"))
end

# Set the search grid.
sparsity_grid = [ [0.0, 0.5, 0.9]; [1 - 50 / n for n in size_grid] ]

# Initialize results and log files.
for (opt, fname) in zip(algorithms, fnames)
    options = (
        :seed => seed,
        :default_size => default_size,
        :k0 => k0,
        :effect_size_min => effect_size[1],
        :effect_size_max => effect_size[2],
        :default_sigma_ij => default_sigma_ij,
        :default_sigma_j => default_sigma_j,
        :causal_sigma_ij => causal_sigma_ij,
        :causal_sigma_j => causal_sigma_j,
        :algorithm => opt,
        :proportion_train => proportion_train,
        :tol => tol,
        :nouter => nouter,
        :ninner => ninner,
        :mult => mult,
    )

    # results file
    open(joinpath("results", "experiment2", "$(fname).out"), "a+") do results
        writedlm(results, ["m" "n" "k0" "sparsity" "sv" "time" "iter" "obj" "dist" "MSE" "TP" "FP" "TN" "FN" "train_acc" "test_acc"])
    end
    
    # log file
    open(joinpath(dir, "$(fname).log"), "a+") do io
        for (key, val) in options
            writedlm(io, [key val], '=')
        end
        println(io)
    end
end

# Case m > n: More samples than predictors.
for m in size_grid
    n = default_size

    # Simulate data.
    y, X, beta0, _, _ = simulate(m, n)

    for (opt, fname) in zip(algorithms, fnames)
        run_experiment(fname, opt, y, X, beta0, sparsity_grid,
            tol=tol,
            ninner=ninner,
            nouter=nouter,
            mult=mult,
            proportion_train=proportion_train,
        )
    end
end

# Case: m < n: More predictors than samples.
for n in size_grid
    m = default_size

    # Simulate data.
    y, X, beta0, _, _ = simulate(m, n)

    for (opt, fname) in zip(algorithms, fnames)
        run_experiment(fname, opt, y, X, beta0, sparsity_grid,
            tol=tol,
            ninner=ninner,
            nouter=nouter,
            mult=mult,
            proportion_train=proportion_train,
        )
    end
end
