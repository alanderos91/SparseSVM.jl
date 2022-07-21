function simulate_spiral(local_dir, dataset)
    # Simulate the data.
    tmpfile = joinpath(local_dir, "$(dataset).tmp")
    rng = StableRNG(1903)
    L, X = spiral((600, 300, 100);
        rng=rng,
        max_radius=7.0,
        x0=-3.5,
        y0=3.5,
        angle_start=pi/8,
        prob=1.0,
    )
    x, y = view(X, :, 1), view(X, :, 2)
    df = DataFrame(class=L, x=x, y=y)
    perm = Random.randperm(rng, size(df, 1))
    foreach(col -> permute!(col, perm), eachcol(df))
    CSV.write(tmpfile, df)

    # Standardize format.
    local_path = SparseSVM.process_dataset(tmpfile, dataset;
        label_mapping=string,
        header=true,
        class_index=1,
        variable_indices=2:ncol(df),
        ext=".csv",
    )

    # Store column information.
    column_info = ["class", "x", "y"]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(joinpath(local_dir, "$(dataset).info"), column_info_df; writeheader=false, delim=',')

    return local_path
end

push!(
    MESSAGES[],
    """
    ## Dataset: spiral

    **3 classes / 1000 instances / 2 variables**
    
    Adapted from: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

    Simulation of a noisy pattern with 3 spirals.
    """
)

push!(REMOTE_PATHS[], "<simulate:spiral>")

push!(CHECKSUMS[], "97fa5e682081e01a94d3104260f9f317ec7c1e2a08e95ecb1d047c5c1bfaf196")

push!(FETCH_METHODS[], (unused_path, local_dir) -> simulate_spiral(local_dir, "spiral"))

push!(POST_FETCH_METHODS[], identity)

push!(DATASETS[], "spiral")

function simulate_spiral_hard(local_dir, dataset)
    # Simulate the data.
    tmpfile = joinpath(local_dir, "$(dataset).tmp")
    rng = StableRNG(1903)
    L, X = spiral((600, 300, 100);
        rng=rng,
        max_radius=7.0,
        x0=-3.5,
        y0=3.5,
        angle_start=pi/8,
        prob=0.9,
    )
    x, y = view(X, :, 1), view(X, :, 2)
    df = DataFrame(class=L, x=x, y=y)
    perm = Random.randperm(rng, size(df, 1))
    foreach(col -> permute!(col, perm), eachcol(df))
    CSV.write(tmpfile, df)

    # Standardize format.
    local_path = SparseSVM.process_dataset(tmpfile, dataset;
        label_mapping=string,
        header=true,
        class_index=1,
        variable_indices=2:ncol(df),
        ext=".csv",
    )

    # Store column information.
    column_info = ["class", "x", "y"]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(joinpath(local_dir, "$(dataset).info"), column_info_df; writeheader=false, delim=',')

    return local_path
end

push!(
    MESSAGES[],
    """
    ## Dataset: spiral-hard

    **3 classes / 1000 instances / 2 variables**
    
    Adapted from: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

    Simulation of a noisy pattern with 3 spirals.

    Bayes error is expected to be ≈0.1 due to random class inversions.
    """
)

push!(REMOTE_PATHS[], "<simulate:spiral-hard>")

push!(CHECKSUMS[], "8e4e2f53d7745b4dee52d5f7d3a279315f321e5616220215bedee9c30848245f")

push!(FETCH_METHODS[], (unused_path, local_dir) -> simulate_spiral_hard(local_dir, "spiral-hard"))

push!(POST_FETCH_METHODS[], identity)

push!(DATASETS[], "spiral-hard")
