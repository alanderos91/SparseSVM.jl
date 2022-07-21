function process_bcw(local_path, dataset)
    # Define a mapping to change labels.
    function label_mapping(old_label)
        if old_label == 2
            return "benign"
        elseif old_label == 4
            return "malignant"
        else
            error("Unknown label $(old_label)")
        end
    end

    # Standardize format.
    dir = dirname(local_path)
    SparseSVM.process_dataset(local_path, dataset;
        label_mapping=label_mapping,
        missingstring=["?"],
        header=false,
        class_index=11,
        variable_indices=2:10,
        ext=".csv",
    )    

    # Store column information.
    column_info = Vector{String}(undef, 10)
    column_info[1] = "diagnosis"
    column_info[2] = "clump_thickness"
    column_info[3] = "cell_size_uniformity"
    column_info[4] = "cell_shape_uniformity"
    column_info[5] = "marginal_adhesion"
    column_info[6] = "single_cell_epithelial_size"
    column_info[7] = "bare_nuclei"
    column_info[8] = "bland_chromatin"
    column_info[9] = "normal_nucleoli"
    column_info[10] = "mitoses"
    
    info_file = joinpath(dir, "$(dataset).info")
    column_info_df = DataFrame(columns=column_info)
    CSV.write(info_file, column_info_df; writeheader=false, delim=',')

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: bcw (Breast Cancer Wisconsin)

    **2 classes / 683 instances (16 dropped) / 9 variables**
    
    See: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

    16 instances from the original dataset are dropped due to missing values.
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data")

push!(CHECKSUMS[], "402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_bcw(path, "bcw"))

push!(DATASETS[], "bcw")