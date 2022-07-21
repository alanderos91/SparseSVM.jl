function process_iris(local_path, dataset)
    # Standardize format.
    dir = dirname(local_path)
    SparseSVM.process_dataset(local_path, dataset;
        label_mapping=string,
        header=false,
        class_index=5,
        variable_indices=1:4,
        ext=".csv",
    )

    # Store column information.
    column_info = Vector{String}(undef, 5)
    column_info[1] = "species"
    column_info[2] = "sepal_length_cm"
    column_info[3] = "sepal_width_cm"
    column_info[4] = "petal_length_cm"
    column_info[5] = "petal_width_cm"
    
    info_file = joinpath(dir, "$(dataset).info")
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: iris

    **3 classes / 150 instances / 4 variables**
    
    See: https://archive.ics.uci.edu/ml/datasets/iris
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

push!(CHECKSUMS[], "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_iris(path, "iris"))

push!(DATASETS[], "iris")
