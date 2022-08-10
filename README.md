# SparseSVM

This package implements sparse SVM algorithms for variable selection.
It provides proximal distance algorithms to minimize the objective
$$
    h_\rho(w,b)
    =
    \frac{1}{2n} \sum_{i=1}^{n} \max\{0, 1-y_{i} x_{i}^{\top} w \}^{2}
    + \frac{\lambda}{2}\|w\|_{2}^{2}
    + \frac{\rho}{2} \mathrm{dist}(w, S_{k})^{2},
$$
which is a combination of the L2-regularized / L2-SVM model and a distance-to-sparsity penalty.
Specifically, $\mathrm{dist}(w, S_{k})$ quantifies how close the weights $w$ are to a sparse representation with at most $k$ nonzero components.

It also includes scripts to reproduce results in "Algorithms for Sparse Support Vector
Machines".

Project code should exist as a separate folder, say `SparseSVM`, in a user's filesystem, and should *not* be installed as a package in the main Julia environment.

## Installation

These instructions assume one has installed Julia 1.7 (or higher), which is available at [https://julialang.org/downloads/](https://julialang.org/downloads/).

The files `Project.toml` and `Manifest.toml` are used to download and install packages used in developing the project's code.
This setup need only run once.
Please follow the instructions below to install SparseSVM correctly:

1. Start a new Julia session by either double-clicking the Julia application shipped from the official downloads page (e.g. on Windows and MacOS systems), or invoking the Julia command (e.g. `julia` by default if configured).

2. At the Julia prompt (`julia>`), type `]` to enter *package mode*. The prompt should now read
```julia-repl
(@v1.7) pkg>
```

3. **In package mode**, `activate` the environment specified by `SparseSVM`. For example, if the project code is contained in a folder named `SparseSVM`, the command prompt should resemble
```julia-repl
(@v1.7) pkg> activate /path/to/SparseSVM
```
Confirm the path to the project folder and then hit `Enter` to execute.
The prompt should now read
```julia-repl
(SparseSVM) pkg>
```

4. **In package mode**, `instantiate` all project dependencies; e.g.
```julia-repl
(SparseSVM) pkg> instantiate
```
then hit `Enter` to execute.

5. **In package mode**, `test` that the installation was successful by executing
```julia-repl
(SparseSVM) pkg> test
```

## Basic Usage

Code from this project should be run with the project's environment.
Make sure to run
```julia-repl
(@v1.7) pkg> activate /path/to/SparseSVM
```
before attempting to use the package code or run any scripts.

If executing a script, one can run
```julia-repl
julia -t 4 --project=/path/to/SparseSVM /path/to/script.jl arg1 arg2 ...
```
where

- `-t 4` specifies that we want 4 threads available to Julia,
- `--project=/path/to/SparseSVM` forces the Julia session to use the SparseSVM environment. One can also use `--project=.` if the current directory is `SparseSVM`.
- `/path/to/script.jl` is the script to execute; relative paths are valid.
- `arg1 arg2 ...` is a list of arguments passed to the Julia script.

---

To load the project code, simply run

```julia
using SparseSVM
```

The first time you load `SparseSVM` you are prompted to download example datasets. Files are typically stored in your home directory, e.g. `/home/username/.julia/datadeps/`.

Datasets are loaded with the `SparseSVM.dataset` command; e.g.

```julia
df = SparseSVM.dataset("synthetic")
```

where the object `df` is a `DataFrame` from the DataFrames.jl package.
The first time a particular dataset is loaded you will be prompted to proceed with the setup/download.

* Labels/targets always appear in the first column; that is, `df[!,1]` or `df.class`.
* Features are stored in the remaining columns; that is, `df[!,2:end]` or `df.x1`, `df.x2`, and so on.

### Example: `synthetic`

The following code illustrates fitting the `synthetic` example:

```julia
# 1. Load packages
using SparseSVM

# 2. Load data
df = SparseSVM.dataset("synthetic")
L, X = string.(df.class), Matrix{Float64}(df[!,2:end])

# 3. Initialize a classifier with an intercept but without any specialized kernel (linear).
#
#    Note: We have to specify the label for the 'positive' class based on values in L
#    In this case we use "A". 
#
problem = BinarySVMProblem(L, X, "A", intercept=true, kernel=nothing)

# 4. Fit the L2-SVM model with Majorization-Minimization (MM) algorithm
lambda = 1.0
result = @time SparseSVM.fit(MMSVD(), problem, lambda,
    maxiter=10^4,                     # maximum number of iterations
    gtol=1e-6,                        # set control parameter on magnitude of gradients
    cb=SparseSVM.VerboseCallback(5),  # print convergence history every 5 iterations
)

# 5. Check training accuracy.
percent_correct = 100 * sum(SparseSVM.classify(problem, X) .== L) / length(L)

# 6. Check the number of nonzero model parameters (excluding the intercept).
nvars = length(SparseSVM.active_variables(problem))

# 7. Check number of support vectors.
nsv = length(SparseSVM.support_vectors(problem))

println("Accuracy: $(percent_correct) %, $(nvars) active variables, $(nsv) support vectors")
```

Now let's try fitting a sparse model

```julia
lambda = 1.0
sparsity = 498 / 500 # want only 2 nonzeros
result = @time SparseSVM.fit(MMSVD(), problem, lambda, sparsity,
    ninner=10^4,                                # maximum number of iterations
    nouter=10^2,                                # maxmium number of outer iterations; i.e. rho values to test
    gtol=1e-6,                                  # set control parameter on magnitude of gradients
    dtol=1e-3,                                  # set control parameter on distance
    rhof=SparseSVM.geometric_progression(1.2),  # define the rho sequence; i.e. rho(t+1) = 1.2 * rho(t)
    cb=SparseSVM.VerboseCallback(5),            # print convergence history every 5 iterations
)

percent_correct = 100 * sum(SparseSVM.classify(problem, X) .== L) / length(L)
nvars = length(SparseSVM.active_variables(problem))
nsv = length(SparseSVM.support_vectors(problem))

println("Accuracy: $(percent_correct) %, $(nvars) active variables, $(nsv) support vectors")
```

### Example: `spiral`

The following code illustrates fitting the `spiral` example:

```julia
# 1. Load packages.
using SparseSVM, KernelFunctions

# 2. Load data.
df = SparseSVM.dataset("spiral")
L, X = string.(df.class), Matrix{Float64}(df[!,2:end])

# 3. Load classifier object with Gaussian kernel. In this case we use OVO to split the problem.
problem = MultiSVMProblem(L, X, intercept=true, strategy=OVO(), kernel=RBFKernel())

# 4. Fit a classifier without any sparsity constraints
lambda = 1.0
result = @time SparseSVM.fit(MMSVD(), problem, lambda,
    maxiter=10^4,                     # maximum number of iterations
    gtol=1e-6,                        # set control parameter on magnitude of gradients
    cb=SparseSVM.VerboseCallback(5),  # print convergence history every 5 iterations
)

# 5. Check training accuracy.
percent_correct = 100 * sum(SparseSVM.classify(problem, X) .== L) / length(L)

# 6. Check the number of nonzero model parameters (excluding the intercept).
nvars = length(SparseSVM.active_variables(problem))

# 7. Check number of support vectors.
nsv = length(SparseSVM.support_vectors(problem))

println("Accuracy: $(percent_correct) %, $(nvars) active variables, $(nsv) support vectors")
```

```julia
lambda = 1.0
sparsity = 0.5 # try to keep support vectors to 500
result = @time SparseSVM.fit(MMSVD(), problem, lambda, sparsity,
    ninner=10^4,                                # maximum number of iterations
    nouter=10^2,                                # maxmium number of outer iterations; i.e. rho values to test
    gtol=1e-6,                                  # set control parameter on magnitude of gradients
    dtol=1e-3,                                  # set control parameter on distance
    rhof=SparseSVM.geometric_progression(1.2),  # define the rho sequence; i.e. rho(t+1) = 1.2 * rho(t)
    cb=SparseSVM.VerboseCallback(5),            # print convergence history every 5 iterations
)

percent_correct = 100 * sum(SparseSVM.classify(problem, X) .== L) / length(L)
nvars = length(SparseSVM.active_variables(problem))
nsv = length(SparseSVM.support_vectors(problem))

println("Accuracy: $(percent_correct) %, $(nvars) active variables, $(nsv) support vectors")
```

## Scripts

Here we describe the scripts used in our numerical experiments.

* The subdirectory `experiments/` contains the project scripts.
* Results are written to the `results/` subdirectory.
* The file `experiments/common.jl` sets up commonly used commands between scripts.
* The file `experiments/examples.jl` defines default parameter values and other settings across each dataset.

**Note**: Users may want to edit Line 34 of `experiments/common.jl`
```julia
##### Make sure we set up BLAS threads correctly #####
BLAS.set_num_threads(10)
```
to have the number of BLAS threads match the number of cores on the user's machine.

One should also replace the number of threads `-t 4` which an appropriate value that takes advantage of the user's machine (we assume 4 should be safe).

**In the following command line examples, we assume the directory `SparseSVM` is visible from the current directory.**

### Experiment 1: `1-sparse-recovery.jl`

Results are saved to `results/experiment2`.

```bash
julia -t 4 --project=SparseSVM SparseSVM/experiments/1-sparse-recovery.jl
```

### Experiment 2: `2-cross-validation.jl`

This script requires a minimum of 2 arguments:

- `SUBDIR`: a directory name. Results will be saved to `results/example/SUBDIR`.
- `example1 example2 ...`: a list of example names. The list below includes the full suite.

```bash
julia -t 4 --project=SparseSVM SparseSVM/experiments/2-cross-validation.jl latest iris synthetic synthetic-hard bcw splice optdigits-linear letters-linear TCGA-HiSeq spiral spiral-hard
```

This run will save to `results/example/latest`.

### Experiment 3: `3-libsvm.jl`

This script requires a minimum of 2 arguments:

- `SUBDIR`: a directory name. Results will be saved to `results/example/SUBDIR`.
- `example1 example2 ...`: a list of example names. The list below includes the full suite.

```bash
julia -t 4 --project=SparseSVM SparseSVM/experiments/3-libsvm.jl latest iris synthetic synthetic-hard bcw splice optdigits-linear letters-linear TCGA-HiSeq
```

This run will save to `results/example/latest`.

### Figure 2: `figure2.jl`

This requires two arguments: an input directory with results and an output directory.

```bash
julia --project=SparseSVM SparseSVM/experiments/figure2.jl results/experiment2 figures
```

This reads from `results/experiment2` and saves to `figures/Fig2.pdf`.

### Figure 3: `figure3.jl`

This requires two arguments: an input subdirectory with results and an output directory.

```bash
julia --project=SparseSVM SparseSVM/experiments/figure3.jl latest figures
```

This reads from `results/TCGA-HiSeq/latest` and saves to `figures/Fig3.pdf`.

### Table 2: `table2.jl`

This requires three arguments:

- an input subdirectory with results
- an output directory, and
- a replicate number (if results contain 10 replicates, valid inputs are 1 through 10)

```bash
julia --project=SparseSVM SparseSVM/experiments/table2.jl latest tables 2
```

This reads from `results/synthetic/latest` and saves to directory `tables/Table2.tex`.

### Table 3: `table3.jl`

This requires at least three arguments:

- an input subdirectory with results
- an output directory, and
- an example name or a list of example names

```bash
julia --project=SparseSVM SparseSVM/experiments/table3.jl latest tables synthetic synthetic-hard bcw iris splice optdigits-linear letters-linear TCGA-HiSeq spiral spiral-hard
```

This reads from `results/example/latest` for each example and saves to `tables/Table3.tex`.

### Table 4: `table4.jl`

This requires at least three arguments:

- an input subdirectory with results
- an output directory, and
- an example name or a list of example names

```bash
julia --project=SparseSVM SparseSVM/experiments/table4.jl latest tables synthetic synthetic-hard bcw iris splice optdigits-linear letters-linear TCGA-HiSeq
```

This reads from `results/example/latest` for each example and saves to `tables/Table4.tex`.

### Stability Plots: `stability.jl`

This requires at least three arguments:

- an input subdirectory with results
- an output directory, and
- an example name or a list of example names

```bash
julia --project=SparseSVM SparseSVM/experiments/stability.jl latest figures synthetic synthetic-hard bcw iris splice optdigits-linear spiral spiral-hard
```

This reads from `results/example/latest` for each example and saves to a directory called `figures`.
