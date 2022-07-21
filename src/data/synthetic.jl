function simulate_synthetic(local_dir, dataset)
    # Simulate the data.
    tmpfile = joinpath(local_dir, "$(dataset).tmp")
    m, n, rng = 10^3, 500, StableRNG(2000)
    L, X = synthetic(1,1,1e-2,0.4,1e-4; rng=rng, prob=1.0, m=m, n=n)
    df = hcat(DataFrame(class=L), DataFrame(X, :auto))
    perm = Random.randperm(rng, size(df, 1))
    foreach(col -> permute!(col, perm), eachcol(df))
    CSV.write(tmpfile, df, writeheader=true)

    # Standardize format.
    local_path = SparseSVM.process_dataset(tmpfile, dataset;
        label_mapping=string,
        header=true,
        class_index=1,
        variable_indices=2:ncol(df),
        ext=".csv",
    )

    # Store column information.
    column_info = [["class", "causal1", "causal2"]; ["redundant$(i)" for i in 3:n]]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(joinpath(local_dir, "$(dataset).info"), column_info_df; writeheader=false, delim=',')
    
    return local_path
end

push!(
    MESSAGES[],
    """
    ## Dataset: synthetic

    **2 classes / 1000 instances / 500 variables**
    
    A simulated multivariate normal.

    Classes are determined by the first two variables using the signs of `X*b`
    using `b[1] = 10`, `b[2] = -10`, and `b[j] = 0` otherwise.
    
    Covariance structure is as follows

    ```julia
    Σ[1,1] = 1
    Σ[2,2] = 1
    Σ[i,i] = 1e-2 for i=3, 4, …, 500
    Σ[1,2] = 0.4
    Σ[i,j] = 1e-4
    ```
    """
)

push!(REMOTE_PATHS[], "<simulate:synthetic>")

push!(CHECKSUMS[], "76716cadf67586103ddcd55c8e16fae01af1750338bfe821903de47155c4583e")

push!(FETCH_METHODS[], (unused_path, local_dir) -> simulate_synthetic(local_dir, "synthetic"))

push!(POST_FETCH_METHODS[], identity)

push!(DATASETS[], "synthetic")

function simulate_synthetic_hard(local_dir, dataset)
    # Simulate the data.
    tmpfile = joinpath(local_dir, "$(dataset).tmp")
    m, n, rng = 10^3, 500, StableRNG(2000)
    L, X = synthetic(1,1,1e-2,0.4,1e-4; rng=rng, prob=0.9, m=m, n=n)
    df = hcat(DataFrame(class=L), DataFrame(X, :auto))
    perm = Random.randperm(rng, size(df, 1))
    foreach(col -> permute!(col, perm), eachcol(df))
    CSV.write(tmpfile, df, writeheader=true)

    # Standardize format.
    local_path = SparseSVM.process_dataset(tmpfile, dataset;
        label_mapping=string,
        header=true,
        class_index=1,
        variable_indices=2:ncol(df),
        ext=".csv",
    )

    # Store column information.
    column_info = [["class", "causal1", "causal2"]; ["redundant$(i)" for i in 3:n]]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(joinpath(local_dir, "$(dataset).info"), column_info_df; writeheader=false, delim=',')

    return local_path
end

push!(
    MESSAGES[],
    """
    ## Dataset: synthetic-hard

    **2 classes / 1000 instances / 500 variables**
    
    A simulated multivariate normal.

    Classes are determined by the first two variables using the signs of `X*b`
    using `b[1] = 10`, `b[2] = -10`, and `b[j] = 0` otherwise.
    
    Covariance structure is as follows

    ```julia
    Σ[1,1] = 1
    Σ[2,2] = 1
    Σ[i,i] = 1e-2 for i=3, 4, …, 500
    Σ[1,2] = 0.4
    Σ[i,j] = 1e-4
    ```

    Bayes error is expected to be ≈0.1 due to random class inversions.
    """
)

push!(REMOTE_PATHS[], "<simulate:synthetic-hard>")

push!(CHECKSUMS[], "7c1e5000fd94933eac39dbf1594fc1a4fce6eca4f8484b552c937e838a833378")

push!(FETCH_METHODS[], (unused_path, local_dir) -> simulate_synthetic_hard(local_dir, "synthetic-hard"))

push!(POST_FETCH_METHODS[], identity)

push!(DATASETS[], "synthetic-hard")
