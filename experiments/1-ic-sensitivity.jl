include("common.jl")

function run_experiment(fname, algorithm, dataset, ctype;
    ntrials::Int=100,
    proportion_train::Real=0.8,
    tol::Real=1e-6,
    s::Float64=0.5,
    nouter::Int=20,
    ninner::Int=1000,
    mult::Real=1.5,
    scale::Symbol=:zscore,
    strategy=MultiClassStrategy=OVR(),
    kernel::Union{Nothing,Kernel}=nothing,
    intercept::Bool=false,
    )
    # Put options into a list.
    options = (
        :fname => fname,
        :algorithm => algorithm,
        :dataset => dataset,
        :ctype => ctype,
        :ntrials => ntrials,
        :proportion_train => proportion_train,
        :tol => tol,
        :s => s,
        :nouter => nouter,
        :ninner => ninner,
        :mult => mult,
        :scale => scale,
        :strategy => strategy,
        :kernel => kernel,
        :intercept => intercept,
    )

    # Load the data
    df = SparseSVM.dataset(dataset)
    y, X = Vector(df[!, 1]), Matrix{Float64}(df[!, 2:end])
    
    if dataset == "TCGA-PANCAN-HiSeq"
        # The TCGA data has a few columns full of 0s that should be dropped
        idxs = findall(i -> isequal(0, maximum(X[:,i])), 1:size(X,2))
        selected = setdiff(1:size(X,2), idxs)

        # Limit to the first 10000 genes
        X = X[:, selected[1:10000]]
    end

    _rescale_!(Val(scale), X)
    labeled_data = (X, y)

    # Create the train and test data sets.
    (train_X, train_targets), (test_X, test_targets) = splitobs(labeled_data, at=proportion_train, obsdim=1)

    # Process options
    f = get_algorithm_func(algorithm)

    # Initialize classifier.
    classifier = make_classifier(ctype, train_X, train_targets, first(train_targets),
        kernel=kernel, intercept=intercept, strategy=strategy)
    local_rng = StableRNG(1903)

    # Create closure to run trainMM.
    A, Asvd = SparseSVM.get_A_and_SVD(classifier)
    execute(ninner, nouter) = @timed trainMM!(classifier, A, f, tol, s, fullsvd=Asvd, nouter=nouter, ninner=ninner, mult=mult, init=false, verbose=false)
    
    # Open file to write results on disk. Save settings on a separate log.
    dir = joinpath("results", dataset)
    !isdir(dir) && mkdir(dir)
    open(joinpath(dir, "$(fname).log"), "w") do io
        for (key, val) in options
            writedlm(io, [key val], '=')
        end
    end
    results = open(joinpath(dir, "$(fname).out"), "w")
    writedlm(results, ["trial" "sparsity" "sv" "time" "iter" "obj" "dist" "gradsq" "train_acc" "test_acc"])

    # helper function to write results
    function write_result(trial, result)
        # Get timing and convergence data.
        t = result.time
        iters, obj, dist, gradsq = result.value
        
        # Check support vectors.
        sv = count_support_vecs(classifier)

        # Test the classifier.
        train_acc = round(accuracy_score(classifier, train_X, train_targets)*100, sigdigits=4)
        test_acc = round(accuracy_score(classifier, test_X, test_targets)*100, sigdigits=4)
        sparsity = round(s*100, sigdigits=4)

        # Write results to file.
        writedlm(results, Any[trial sparsity sv t iters obj dist gradsq train_acc test_acc])
        flush(results)
        
        return nothing
    end

    # Run the algorithm with univariate OLS estimates.
    initialize_weights!(classifier, A)
    result = execute(ninner, nouter)
    write_result(0, result)

    # Run the algorithm several times with randomized weights.
    @showprogress 1 "Testing $(algorithm)..." for trial in 1:ntrials
        randomize_weights!(classifier, local_rng)
        result = execute(ninner, nouter)
        write_result(trial, result)
    end

    close(results)
end

##### MAIN #####
include("examples.jl")

# Check for unknown examples.
examples = String[]
for example in ARGS
    if example in EXAMPLES
        push!(examples, example)
    else
        @warn "Unknown example `$(example)`. Check spelling."
    end
end

# Run selected examples.
for example in examples
    println("Running `$(example)` benchmark")
    for opt in (SD, MM)
        fname = generate_filename(1, opt)

        # options
        ctype, kwargs = OPTIONS[example]

        # precompile
        tmpkwargs = (kwargs..., ninner=2, nouter=2,)
        run_experiment(fname, opt, example, ctype; tmpkwargs...)
        cleanup_precompile(example, fname)

        # run
        run_experiment(fname, opt, example, ctype; kwargs...)
    end
end
