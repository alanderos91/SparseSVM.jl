# SparseSVM

This project contains Julia code to reproduce results in "Algorithms for Sparse Support Vector Machines".

This project should exist as a separate folder, say `SparseSVM`, in a user's filesystem, and should *not* be installed as package.

## Installation

These instructions assume one has installed Julia 1.6 (or higher), which is available at [https://julialang.org/downloads/](https://julialang.org/downloads/).

The files `Project.toml` and `Manifest.toml` are used to download and install packages used in developing the project's code.
This setup need only run once.
Please follow the instructions below to install SparseSVM correctly:

1. Start a new Julia session by either double-clicking the Julia application shipped from the official downloads page (e.g. on Windows and MacOS systems), or invoking the Julia command (e.g. `julia` by default if configured).

2. At the Julia prompt (`julia>`), type `]` to enter *package mode*. The prompt should now read
```julia-repl
(@v1.6) pkg>
```

3. **In package mode**, `activate` the environment specified by `SparseSVM`. For example, if the project code is contained in a folder named `SparseSVM`, the command prompt should resemble
```julia-repl
(@v1.6) pkg> activate /path/to/SparseSVM
```
Confirm the path to the project folder and then hit `Enter` to execute.
The prompt should now read
```julia-repl
(SparseSVM) pkg>
```

4. **In package mode**, `instantiate` all project dependencies; e.g.
```julia-repl
(SparseSVM) pkg> resolve
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
(@v1.6) pkg> activate /path/to/SparseSVM
```
before attempting to use the package code or run any scripts.

---

To load the project code, simply run

```julia
using SparseSVM
```

Datasets are loaded with the `SparseSVM.dataset` command; e.g.

```julia
df = SparseSVM.dataset("synthetic")
```

where the object `df` is a `DataFrame` from the DataFrames.jl package.
The first time a particular dataset is loaded you will be prompted to proceed with the setup/download.
Subsequent loading commands will use the cached version which typically lives in `~/.julia/datadeps` as specified by the DataDeps.jl package.

* Labels/targets always appear in the first column; that is, `df[!,1]` or `df.target`.
* Features are stored in the remaining columns; that is, `df[!,2:end]` or `df.x1`, `df.x2`, and so on.

### Example: `synthetic`

The following code illustrates fitting the `synthetic` example:

```julia
# 1. load packages
using SparseSVM

# 2. load data
df = SparseSVM.dataset("synthetic")
y, X = Vector(df.target), Matrix(df[!,2:end])

# 3. build classifier without any specialized kernel
classifier = BinaryClassifier(X, y, first(y), intercept=true, kernel=nothing)

# 4. fit the SVM model with Majorization-Minimization (MM) algorithm
alg = sparse_direct!
tol = 1e-6
sparsity = 0.997
iter, obj, dist, gradsq = trainMM(classifier, alg, tol, sparsity,
    ninner=10^4,    # number of inner iterations
    nouter=100,     # number of outer itreations
    mult=1.5,       # multiplier for rho; annealing schedule
    verbose=true,   # display convergence data
    init=true       # initialize weights
)

# 5. check training accuracy
sum(classifier(X) .== y) / length(y) * 100

# 6. check the number of nonzero model coefficients (excluding the intercept)
count(!isequal(0), classifier.weights[1:end-1])
```

### Example: `spiral`

The following code illustrates fitting the `spiral` example:

```julia
# 1. load packages
using SparseSVM, KernelFunctions

# 2. load data
df = SparseSVM.dataset("spiral")
y, X = Vector(df.target), Matrix(df[!,2:end])

# 3. build classifier object with Gaussian kernel
classifier = MultiClassifier(X, y, intercept=true, strategy=OVO(), kernel=RBFKernel())

# 4. fit the SVM model with Steepest Descent (SD) algorithm
alg = sparse_steepest!
tol = 1e-6
sparsity = 0.5
iter, obj, dist, gradsq = trainMM(classifier, alg, tol, sparsity,
    ninner=10^4,    # number of inner iterations
    nouter=100,     # number of outer itreations
    mult=1.5,       # multiplier for rho; annealing schedule
    verbose=true,   # display convergence data
    init=true       # initialize weights
)

# 5. check training accuracy
sum(classifier(X) .== y) / length(y) * 100

# 6. check the number of nonzero model coefficients (excluding the intercept)
#    on a particular SVM
count(!isequal(0), classifier.svm[1].weights[1:end-1])
```

## Scripts

Here we describe the scripts used in our numerical experiments.

* The subdirectory `experiments/` contains the project scripts.
* Results are written to the `results/` subdirectory.
* The file `experiments/common.jl` sets up commonly used commands between scripts.
* The file `experiments/examples.jl` defines default parameter values and other settings across each dataset.
* The file `experiments/LIBSVM_wrappers.jl` sets up some functions to make using `LIBSVM.jl` easier in Experiment 4.

**Note**: Users may want to edit Line 6 of `experiments/common.jl`
```julia
##### Make sure we set up BLAS threads correctly #####
BLAS.set_num_threads(8)
```
to have the default threads (8) match the number of cores on the user's machine.

**In the following command line examples, we assume the directory `SparseSVM` is visible from the current directory.**

### Experiment 1: `1-ic-sensitivity.jl`

```bash
julia --project=SparseSVM SparseSVM/experiments/1-ic-sensitivity.jl synthetic spiral letter-recognition TCGA-PANCAN-HiSeq
```

### Experiment 2: `2-sparsity-accuracy.jl`

```bash
julia --project=SparseSVM SparseSVM/experiments/2-sparsity-accuracy.jl
```

### Experiment 3: `3-cross-validation.jl`

```bash
julia --project=SparseSVM SparseSVM/experiments/3-cross-validation.jl synthetic iris spiral breast-cancer-wisconsin splice letter-recognition TCGA-PANCAN-HiSeq optdigits
```

### Experiment 4: `4-libsvm.jl`

```bash
julia --project=SparseSVM SparseSVM/experiments/4-libsvm.jl synthetic iris spiral letter-recognition breast-cancer-wisconsin splice TCGA-PANCAN-HiSeq optdigits
```

### Figure 1:

Results are stored in the subdirectory `figures`.

```bash
julia --project=SparseSVM SparseSVM/experiments/figure1.jl SparseSVM/results figures synthetic spiral letter-recognition TCGA-PANCAN-HiSeq
```

### Figure 2:

Results are stored in the subdirectory `figures`.

```bash
julia --project=SparseSVM SparseSVM/experiments/figure2.jl SparseSVM/results figures
```

### Figure 3:

Results are stored in the subdirectory `figures`.

```bash
julia --project=SparseSVM SparseSVM/experiments/figure3.jl SparseSVM/results figures
```

### Table 2: `table2.jl`

Results are stored in the subdirectory `tables`.

```bash
julia --project=SparseSVM SparseSVM/experiments/table2.jl SparseSVM/results tables
```

### Table 3: `table3.jl`

Results are stored in the subdirectory `tables`.

```bash
julia --project=SparseSVM SparseSVM/experiments/table3.jl SparseSVM/results tables
```

### Table 4: `table4.jl`

Results are stored in the subdirectory `tables`.

```bash
julia --project=SparseSVM SparseSVM/experiments/table4.jl SparseSVM/results tables
```