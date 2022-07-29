# load common packages + functions
include("common.jl")

# Print logging messages to STDOUT
global_logger(ConsoleLogger(stdout))

# Header for cross validatiton table.
cv_table_header() = join(("algorithm","replicate","fold","lambda","sparsity","nnz",
"iters","risk","loss","objective","gradient","norm","distance",
"time","train","validation","test"), ',')

# Header for comparison table.
comparison_table_header() = join(("algorithm", "model", "lambda", "sparsity", "nnz", "train", "test", "iterations", "risk", "loss", "objective", "distance", "gradient", "norm", "nsv"), ',')

function run_experiment(paths, dataset, svm_type, algorithm, lambda_grid, kwargs, need_shuffle)
    # Unpack tuple arguments.
    svm_kwargs, fit_kwargs, cv_kwargs = kwargs
    settings_fname, results_fname, compare_fname = paths
    alg_str = string(typeof(algorithm))
    sparsity_grid = [0.0]

    # Save values of keyword arguments.
    open(settings_fname, "w") do io
        write(io, "========== SVM settings ==========\n\n")
        for (key, val) in pairs(svm_kwargs)
            write(io, key, "=", string(val), "\n")
        end
        write(io, "\n========== fit settings ==========\n\n")
        for (key, val) in pairs(fit_kwargs)
            write(io, key, "=", string(val), "\n")
        end
        write(io, "\n========== cv settings  ==========\n\n")
        for (key, val) in pairs(cv_kwargs)
            write(io, key, "=", string(val), "\n")
        end
        write(io, "\n==========   sparsity   ==========\n\n")
        write(io, "sparsity_grid=[", 0.0, "]", "\n")
        write(io, "\n==========    lambda    ==========\n\n")
        write(io, "lambda_grid=[", join(lambda_grid, ','), "]", "\n")
    end
    
    # Load the data.
    df = SparseSVM.dataset(dataset)
    labeled_data = Vector(string.(df[:, 1])), Matrix{Float64}(df[:, 2:end])

    # Create the train and test data sets.
    if need_shuffle
        shuffled_data = shuffleobs(labeled_data, obsdim=1, rng=cv_kwargs.rng)
    else
        shuffled_data = labeled_data
    end
    cv_set, test_set = splitobs(shuffled_data, at=cv_kwargs.at, obsdim=1)
    problem = create_classifier(svm_type, shuffled_data, svm_kwargs)

    # Precompile.
    tmp_cv_kwargs = (; cv_kwargs..., nreplicates=3, nfolds=3,)
    cb = RepeatedCVCallback{CVStatisticsCallback}(sparsity_grid, lambda_grid, 3, 3)
    SparseSVM.repeated_cv(algorithm, problem, lambda_grid;
        scoref=SparseSVM.prediction_accuracies,
        fit_kwargs...,
        tmp_cv_kwargs...,
        show_progress=false,
        cb=cb,
    )

    # Run cross-validation.
    cb = RepeatedCVCallback{CVStatisticsCallback}(sparsity_grid, lambda_grid, cv_kwargs.nfolds, cv_kwargs.nreplicates)
    cv_tmp = SparseSVM.repeated_cv(algorithm, problem, lambda_grid;
        scoref=SparseSVM.prediction_accuracies,
        fit_kwargs...,
        cv_kwargs...,
        show_progress=true,
        cb=cb,
    )

    # Reshape data.
    cv_scores = extract_cv_data(cv_tmp)
    cv_extras = extract_cv_data(cb)

    # Save results to file.
    open(results_fname, "a") do io
        x = cv_scores.time
        is, js, ks, rs = axes(x)
        for r in rs, k in ks, j in js, i in is
            cv_data = (alg_str, r, k, lambda_grid[j], sparsity_grid[i],
                cv_extras.nnz[i,j,k,r],
                cv_extras.iters[i,j,k,r],
                cv_extras.risk[i,j,k,r],
                cv_extras.loss[i,j,k,r],
                cv_extras.objective[i,j,k,r],
                cv_extras.gradient[i,j,k,r],
                cv_extras.norm[i,j,k,r],
                cv_extras.distance[i,j,k,r],
                cv_scores.time[i,j,k,r],
                cv_scores.train[i,j,k,r],
                cv_scores.validation[i,j,k,r],
                cv_scores.test[i,j,k,r],
            )
            write(io, join(cv_data, ','), "\n")
        end
    end
    
    # Average over folds and replicates.
    dims = (3,4)
    score_grid = dropdims(mean(cv_scores.validation, dims=dims), dims=dims)

    # Select optimal set of hyperparameters. 
    _, _, score_opt = SparseSVM.search_hyperparameters(sparsity_grid, lambda_grid, score_grid, minimize=false)
    validation_accuracy, sparsity_opt, lambda_opt = score_opt
    @info "Optimal hyperparameters" sparsity_opt lambda_opt validation_accuracy

    # Fit sparse model with cv_set using selected hyperparameters.
    F = StatsBase.fit(cv_kwargs.data_transform, cv_set[2], dims=1)
    SparseSVM.__adjust_transform__(F)
    StatsBase.transform!(F, cv_set[2])
    StatsBase.transform!(F, test_set[2])

    problemA = create_classifier(svm_type, cv_set, svm_kwargs)
    resultA = SparseSVM.fit(algorithm, problemA, lambda_opt)

    # Compare the three models.
    cv_L, cv_X = getobs(cv_set, obsdim=1)
    test_L, test_X = getobs(test_set, obsdim=1)
    train_accuracyA = prediction_accuracy(problemA, cv_L, cv_X)
    test_accuracyA = prediction_accuracy(problemA, test_L, test_X)
    itersA, statsA = extract_iters_and_stats(resultA)

    @info "Sparse SVM results" train_accuracy=train_accuracyA test_accuracy=test_accuracyA iterations=itersA statsA...

    open(compare_fname, "a") do io
        nnz = count(!isequal(0), last(SparseSVM.get_params_proj(problemA)))
        nsv = SparseSVM.support_vectors(problemA) |> length
        rowA = (alg_str, "sparse", lambda_opt, sparsity_opt, nnz, train_accuracyA, test_accuracyA, itersA, statsA..., nsv)
        write(io, join(rowA, ','), "\n")
    end

    flush(stdout)

    return nothing
end

##### MAIN #####
include("examples.jl")

if length(ARGS) < 2
    error("""
    Need at least two arguments.

    - For i = 1, ARGS[1] should be a directory
    - For i > 1, ARGS[i] should be a dataset
    """)
end

# Check for unknown examples.
dir = ARGS[1]
examples = String[]
for example in ARGS[2:end]
    if example in EXAMPLES
        push!(examples, example)
    else
        @warn "Unknown example `$(example)`. Check spelling."
    end
end

# Run selected examples.
for example in examples
    # options
    dataset, SVM, svm_kwargs, fit_kwargs, cv_kwargs, need_shuffle = OPTIONS[example]
    if SVM <: MultiSVMProblem && svm_kwargs.kernel isa Nothing
        # Always use OVR for linear multiclass classifiers to match LIBLINEAR.
        svm_kwargs = (;svm_kwargs..., strategy=OVR(),)
    elseif SVM <: MultiSVMProblem && !(svm_kwargs.kernel isa Nothing)
        # Always use OVO for nonlinear multiclass classifiers to match LIBSVM.
        svm_kwargs = (;svm_kwargs..., strategy=OVO(),)
    end
    fit_kwargs = (;verbose=false, tolerance=fit_kwargs.gtol,)
    kwargs = (svm_kwargs, fit_kwargs, cv_kwargs)
    lambda_grid = LAMBDA_GRID[example]

    # Set directory to store results.
    full_dir = joinpath("results", example, dir)
    if !isdir(full_dir)
        mkpath(full_dir)
    end

    results_fname = joinpath(full_dir, "cv-libsvm-result.out")
    open(results_fname, "w") do io
        write(io, cv_table_header(), "\n")
    end

    compare_fname = joinpath(full_dir, "cv-libsvm-comparison.out")
    open(compare_fname, "w") do io
        write(io, comparison_table_header(), "\n")
    end

    # run
    for algorithm in (L2SVM(), L1SVM())
        alg_str = string("alg=", string(typeof(algorithm)))
        settings_fname = joinpath(full_dir, "cv-"*alg_str*".settings")
        paths = (settings_fname, results_fname, compare_fname)

        print("\n")
        @info "Running example '$(example)'" dataset problem=SVM algorithm preshuffle=need_shuffle dir=full_dir settings=settings_fname cv_table=results_fname comparison_table=compare_fname

        run_experiment(paths, dataset, SVM, algorithm, lambda_grid, kwargs, need_shuffle)
    end
end
