const MAXITER_PER_RHO = 10^5
const MAXITER_ANNEAL = 10^2
const RHO_INIT = 1.0
const RHO_MAX = 1e8
const NESTEROV_DELAY = 10
const RHOF = SparseSVM.geometric_progression(1.2)

const OPTIONS = Dict(
    ##### Example 1A: synthetic #####
    "synthetic" => (
        "synthetic",
        BinarySVMProblem,
        (;
            intercept=false,
            kernel=nothing,
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.8,
        ),
        false,
    ),
    ##### Example 1B: synthetic w/ errors #####
    "synthetic-hard" => (
        "synthetic-hard",
        BinarySVMProblem,
        (;
            intercept=false,
            kernel=nothing,
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.8,
        ),
        false,
    ),
    ##### Example 2A: iris #####
    "iris" => (
        "iris",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=nothing,
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-6,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=3,
            at=0.8,
        ),
        true,
    ),
    "iris-ovr" => (
        "iris",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=nothing,
            strategy=OVR(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-6,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=3,
            at=0.8,
        ),
        true,
    ),
    ##### Example 3: spiral #####
    "spiral" => (
        "spiral",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=RBFKernel(),
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.8,
        ),
        false,
    ),
    "spiral-hard" => (
        "spiral-hard",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=RBFKernel(),
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.8,
        ),
        false,
    ),
    ##### Example 4A: letter-recognition w/ linear classifier #####
    "letters-linear" => (
        "letters",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=nothing,
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=100,
        ),
        (;
            data_transform=NoTransformation,
            rng=StableRNG(1903),
            nreplicates=1,
            nfolds=5,
            at=0.8,
        ),
        false,
    ),
    ##### Example 4B: letter-recognition w/ nonlinear classifier #####
    "letters-nonlinear" => (
        "letters",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=RBFKernel(),
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=100,
        ),
        (;
            data_transform=NoTransformation,
            rng=StableRNG(1903),
            nreplicates=1,
            nfolds=5,
            at=0.8,
        ),
        false,
    ),
    ##### Example 5: breast-cancer-wisconsin #####
    "bcw" => (
        "bcw",
        BinarySVMProblem,
        (;
            intercept=true,
            kernel=nothing,
        ),
        (;
            dtol=1e-3,
            gtol=1e-6,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=NoTransformation,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.8,
        ),
        true,
    ),
    ##### Example 6: splice #####
    "splice" => (
        "splice",
        MultiSVMProblem,
        (;
            intercept=false,
            kernel=nothing,
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=NoTransformation,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.8,
        ),
        true,
    ),
    ##### Example 7: TCGA-HiSeq #####
    "TCGA-HiSeq" => (
        "TCGA-HiSeq",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=nothing,
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=2e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=ZScoreTransform,
            rng=StableRNG(1903),
            nreplicates=1,
            nfolds=3,
            at=0.8,
        ),
        false,
    ),
    ##### Example 8A: optdigits w/ linear classifier #####
    "optdigits-linear" => (
        "optdigits",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=nothing,
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=NoTransformation,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.68,
        ),
        false,
    ),
    ##### Example 8B: optdigits w/ nonlinear classifier #####
    "optdigits-nonlinear" => (
        "optdigits",
        MultiSVMProblem,
        (;
            intercept=true,
            kernel=RBFKernel(),
            strategy=OVO(),
        ),
        (;
            dtol=1e-3,
            gtol=1e-4,
            rtol=1e-6,
            maxiter=MAXITER_PER_RHO,
            ninner=MAXITER_PER_RHO,
            nouter=MAXITER_ANNEAL,
            rho_init=RHO_INIT,
            rho_max=RHO_MAX,
            rhof=RHOF,
            nesterov_threshold=NESTEROV_DELAY,
        ),
        (;
            data_transform=NoTransformation,
            rng=StableRNG(1903),
            nreplicates=10,
            nfolds=5,
            at=0.68,
        ),
        false,
    ),
)

const EXAMPLES = collect(keys(OPTIONS))

function make_sparsity_grid(n, m)
    xs = Float64[]

    if n > 100
        y = div(n, m)
        for i in 1:m
            if i == 1
                r = range(log10(y), 0.0, length=11)
            else
                r = range(log10(y), 0.0, length=10)
            end
            for ri in r
                x = 1 - (10.0 ^ ri + (m-i)*y) / n
                push!(xs, round(Int, n*x) / n)
            end
        end
    else
        for i in 0:n-1
            push!(xs, i/n)
        end
    end

    return unique!(xs)
end

function make_lambda_grid(a, b, m)
    xs = Float64[]
    
    for c in a:b-1
        r = range(c, c+1, length=m)
        for ri in r
            push!(xs, 10.0 ^ ri)
        end
    end

    return unique!(xs)
end

const SPARSITY_GRID = Dict(
    "synthetic" => make_sparsity_grid(500, 4),
    "synthetic-hard" => make_sparsity_grid(500, 4),
    "iris" => make_sparsity_grid(4, 1),
    "iris-ovr" => make_sparsity_grid(4, 1),
    "spiral" => make_sparsity_grid(800, 4),
    "spiral-hard" => make_sparsity_grid(800, 4),
    "letters-linear" => make_sparsity_grid(16, 1),
    "letters-nonlinear" => make_sparsity_grid(16000, 4),
    "bcw" => make_sparsity_grid(9, 1),
    "splice" => make_sparsity_grid(180, 5),
    "TCGA-HiSeq" => make_sparsity_grid(20266, 4),
    "optdigits-linear" => make_sparsity_grid(16, 1), 
    "optdigits-nonlinear" => make_sparsity_grid(3821, 4),
)

const DEFAULT_GRID = [1e-1, 1e0, 1e1]

const LAMBDA_GRID = Dict(
    "synthetic" => DEFAULT_GRID,
    "synthetic-hard" => DEFAULT_GRID,
    "iris" => make_lambda_grid(-4, 4, 5),
    "iris-ovr" => make_lambda_grid(-4, 4, 5),
    "spiral" => [1e0],
    "spiral-hard" => [1e0],
    "letters-linear" => make_lambda_grid(-1, 1, 5),
    "letters-nonlinear" => [1e-2],
    "bcw" => make_lambda_grid(-3, 3, 5),
    "splice" => DEFAULT_GRID,
    "TCGA-HiSeq" => DEFAULT_GRID,
    "optdigits-linear" => DEFAULT_GRID,
    "optdigits-nonlinear" => [1e0],
)
