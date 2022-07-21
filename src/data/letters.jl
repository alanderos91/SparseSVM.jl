function process_letters(local_path, dataset)
    # Standardize format.
    dir = dirname(local_path)
    SparseSVM.process_dataset(local_path, dataset;
        label_mapping=string,
        header=false,
        class_index=1,
        variable_indices=2:17,
        ext=".csv.gz",
    )

    # Store column information.
    column_info = Vector{String}(undef, 17)
    column_info[1] = "letter"
    column_info[2] = "xbox"
    column_info[3] = "ybox"
    column_info[4] = "width"
    column_info[5] = "height"
    column_info[6] = "onpix"
    column_info[7] = "xbar"
    column_info[8] = "ybar"
    column_info[9] = "x2bar"
    column_info[10] = "y2bar"
    column_info[11] = "xybar"
    column_info[12] = "x2ybar"
    column_info[13] = "xy2bar"
    column_info[14] = "xege"
    column_info[15] = "xegvy"
    column_info[16] = "yege"
    column_info[17] = "yegvx"
    
    info_file = joinpath(dir, "$(dataset).info")
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: letters

    **26 classes / 20000 instances / 16 variables**

    See: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data")

push!(CHECKSUMS[], "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_letters(path, "letters"))

push!(DATASETS[], "letters")
