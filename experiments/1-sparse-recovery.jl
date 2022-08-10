include("common.jl")
include("examples.jl")

using ProgressMeter

const HEADER = [
    "algorithm", "nsamples", "nvars", "ncausal",
    "lambda", "sparsity", "k",
    "time", "iters", "nSV", "risk", "loss", "objective", "distance", "gradient", "margin",
    "MSE", "wTP", "wFP", "wTN", "wFN",
    "trainTP", "trainFP", "trainTN", "trainFN", "train",
    "testTP", "testFP", "testTN", "testFN", "test", 
]

# Open file to write results on disk.
function write_result(filepath, algorithm, problem, lambda, sparsity, result, gt, (train_L, train_X), (test_L, test_X))
    open(filepath, "a") do results
        row = Vector{String}(undef, 31)
        # algorith + problem dimensions
        row[1] = algorithm |> typeof |> string
        row[2] = problem.n |> string
        row[3] = problem.p |> string
        row[4] = length(gt.causal) |> string

        # hyperparameter settings
        row[5] = lambda |> string
        row[6] = 100*sparsity |> string
        row[7] = SparseSVM.sparsity_to_k(problem, sparsity) |> string

        # convergence data
        (t, iters, stats) = result.time, result.value[1], result.value[2]
        row[8] = t |> string
        row[9] = iters |> string
        row[10] = SparseSVM.support_vectors(problem) |> length |> string
        row[11] = stats.risk |> string
        row[12] = stats.loss |> string
        row[13] = stats.objective |> string
        row[14] = stats.distance |> string
        row[15] = stats.gradient |> string
        row[16] = 1 / stats.norm |> string

        # performance metrics
        _, w = SparseSVM.get_params_proj(problem)
        row[17] = mse(w, gt.coeff) |> string

        (TP, FP, TN, FN) = confusion_matrix_coefficients(w, gt.coeff)
        row[18] = TP |> string
        row[19] = FP |> string
        row[20] = TN |> string
        row[21] = FN |> string
        
        (TP, FP, TN, FN) = confusion_matrix_predictions(SparseSVM.classify(problem, train_X), train_L, "A")
        row[22] = TP |> string # number correct in class A
        row[23] = FP |> string
        row[24] = TN |> string # number correct in class B
        row[25] = FN |> string
        row[26] = TP + TN |> string

        (TP, FP, TN, FN) = confusion_matrix_predictions(SparseSVM.classify(problem, test_X), test_L, "A")
        row[27] = TP |> string # number correct in class A
        row[28] = FP |> string
        row[29] = TN |> string # number correct in class B
        row[30] = FN |> string
        row[31] = TP + TN |> string

        # Write results to file.
        join(results, row, ','); write(results, '\n')
        flush(results)
    end
end

function run_experiment(filepath, algorithm, gt, lambda, grid; write_output::Bool=false, proportion_train::Real=0.8, kwargs...)
    # Create the train and test data sets.
    labeled_data = shuffleobs((gt.labels, gt.samples), obsdim=1, rng=gt.rng)
    cv_set, test_set = splitobs(labeled_data, at=proportion_train, obsdim=1)
    (train_L, train_X) = getobs(cv_set, obsdim=1)
    (test_L, test_X) = getobs(test_set, obsdim=1)

    # Run the algorithm with different sparsity values.
    F = StatsBase.fit(ZScoreTransform, train_X, dims=1)
    SparseSVM.__adjust_transform__(F)
    StatsBase.transform!(F, train_X)
    StatsBase.transform!(F, test_X)

    problem = BinarySVMProblem(train_L, train_X, "A", intercept=false)
    extras = SparseSVM.__mm_init__(algorithm, problem, nothing)
    m, n, _ = SparseSVM.probdims(problem)
    @showprogress 1 "Testing $(algorithm) on $(m) × $(n) problem... " for sparsity in grid
        fill!(problem.coeff_prev, 0)
        result = @timed SparseSVM.fit!(algorithm, problem, lambda, sparsity, extras; kwargs...)
        write_output && write_result(filepath, algorithm, problem, lambda, sparsity, result, gt, (train_L, train_X), (test_L, test_X))
    end

    return nothing
end


function main()
    # Options
    ALGORITHMS = (MMSVD(), SD())
    PROPORTION_TRAIN = 0.8
    DEFAULT_SIZE = 500
    K = round(Int, 0.1*DEFAULT_SIZE)
    W_RANGE = [2.0, 10.0]
    RNG = StableRNG(1903)
    GTOL = 1e-4
    DTOL = 1e-3
    NINNER = 10^6
    NOUTER = 100
    RHO_INIT = 1.0
    RHOF = SparseSVM.geometric_progression(1.2)
    NESTEROV_DELAY = 10
    RHO_MAX = 1e8
    LAMBDA = 1.0
    DIR = joinpath("results", "experiment2")

    # Initialize directories.
    !isdir("results") && mkdir("results")
    !isdir(DIR) && mkdir(DIR)

    p = floor(Int, log10(DEFAULT_SIZE)) + 1
    size_grid = (ceil(Int, 10^(p+u)) for u in range(0, 2, step=0.25))

    simulate(m, n) = let k=K, w_range=W_RANGE, rng=RNG
        SparseSVM.simulate_ground_truth(m, n, k, w_range; rng=rng)
    end

    kwargs = (;
        gtol=GTOL,
        dtol=DTOL,
        ninner=2,
        nouter=2,
        rhof=RHOF,
        rho_init=RHO_INIT,
        rho_max=RHO_MAX,
        nesterov_threshold=NESTEROV_DELAY,
        proportion_train=PROPORTION_TRAIN,
    )

    # Pre-compile.
    sparsity_grid = [0.0, 0.5]
    for algorithm in ALGORITHMS
        m, n = 100, 100
        fname = "not_used.txt"
        gt = simulate(m, n)
        run_experiment(fname, algorithm, gt, LAMBDA, sparsity_grid; write_output=false, kwargs...)
    end

    # Initialize output files with header.
    for algorithm in ALGORITHMS
        fname = joinpath(DIR, "$(string(typeof(algorithm))).out")
        open(fname, "w") do results
            global HEADER
            join(results, HEADER, ','); write(results, '\n')
            flush(results)
        end
    end

    kwargs = (;
        gtol=GTOL,
        dtol=DTOL,
        ninner=NINNER,
        nouter=NOUTER,
        rhof=RHOF,
        rho_init=RHO_INIT,
        rho_max=RHO_MAX,
        nesterov_threshold=NESTEROV_DELAY,
        proportion_train=PROPORTION_TRAIN,
    )

    sparsity_grid = make_sparsity_grid(DEFAULT_SIZE, 2)
    sparsity_grid = filter!(<(0.9), sparsity_grid)
    sparsity_grid = [ sparsity_grid; [1 - K / DEFAULT_SIZE]; [1 - K / n for n in size_grid] ]
    unique!(sort!(sparsity_grid))

    # Case m > n: More samples than predictors.
    for _m in size_grid
        m = round(Int, _m / PROPORTION_TRAIN)
        n = DEFAULT_SIZE
        gt = simulate(m, n)
        for algorithm in ALGORITHMS
            fname = joinpath(DIR, "$(string(typeof(algorithm))).out")
            run_experiment(fname, algorithm, gt, LAMBDA, sparsity_grid; write_output=true, kwargs...)
        end
    end

    # Case: m < n: More predictors than samples.
    for n in size_grid
        m = round(Int, DEFAULT_SIZE / PROPORTION_TRAIN) 
        gt = simulate(m, n)
        for algorithm in ALGORITHMS
            fname = joinpath(DIR, "$(string(typeof(algorithm))).out")
            run_experiment(fname, algorithm, gt, LAMBDA, sparsity_grid; write_output=true, kwargs...)
        end
    end
end

main()
