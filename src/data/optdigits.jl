function process_optdigits(local_path, dataset)
    # Read both training and testing files.
    dir = dirname(local_path)
    tra = joinpath(dir, "optdigits.tra")
    tes = joinpath(dir, "optdigits.tes")
    df = vcat(
        CSV.read(tra, DataFrame, header=false),
        CSV.read(tes, DataFrame, header=false),
    )
    tmpfile = joinpath(dir, "$(dataset).tmp")
    CSV.write(tmpfile, df; writeheader=false, delim=',')

    # Standardize format.
    SparseSVM.process_dataset(tmpfile, dataset;
        label_mapping=string,
        header=false,
        class_index=ncol(df),
        variable_indices=1:ncol(df)-1,
        ext=".csv",
    )
        
    # Store column information.
    info_file = joinpath(dir, "$(dataset).info")
    column_info = [["digit"]; ["$(i)x$(j)" for i in 1:8 for j in 1:8]]
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    # Clean up by removing separate training and testing files.
    rm(tra)
    rm(tes)

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: optdigits

    **10 classes / 5620 instances / 64 variables**

    See: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
    """
)

push!(
    REMOTE_PATHS[],
    [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
    ]
)

push!(
    CHECKSUMS[],
    [
        "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd",
        "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8",
    ]
)

push!(
    FETCH_METHODS[],
    [
        DataDeps.fetch_default,
        DataDeps.fetch_default,
    ]
)

push!(
    POST_FETCH_METHODS[],
    [
        identity,
        path -> process_optdigits(path, "optdigits"),
    ]
)

push!(DATASETS[], "optdigits")
