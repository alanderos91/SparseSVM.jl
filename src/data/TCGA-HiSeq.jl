function process_TCGA_HiSeq(local_path, dataset)
    # Decompress the data and combine variables and labels.
    DataDeps.unpack(local_path, keep_originals=false)

    dir = dirname(local_path)
    tmpdir = joinpath(dir, "TCGA-PANCAN-HiSeq-801x20531")
    data = CSV.read(joinpath(tmpdir, "data.csv"), DataFrame, header=true)
    labels = CSV.read(joinpath(tmpdir, "labels.csv"), DataFrame, header=true)
    df = innerjoin(labels, data, on=:Column1)

    # Drop columns or rows with constant values.
    idx_keep_cols = findall(x -> any(!isequal(first(x)), x), eachcol(df))
    idx_keep_rows = findall(x -> any(!isequal(first(x)), x), eachrow(df))
    df = df[idx_keep_rows, idx_keep_cols]

    # Write to a temporary file.
    tmpfile = joinpath(dir, "$(dataset).tmp")
    CSV.write(tmpfile, df, writeheader=true)

    # Standardize format.
    SparseSVM.process_dataset(tmpfile, dataset,
        label_mapping=string,
        header=true,
        class_index=2,
        variable_indices=3:ncol(df),
        ext=".csv.gz"
    )

    # Save information on selected genes and instances.
    if !isempty(idx_keep_cols)
        info_file = joinpath(dir, "$(dataset).cols")
        CSV.write(info_file, DataFrame(idx=idx_keep_cols); writeheader=false, delim=',')
    end

    if !isempty(idx_keep_rows)
        info_file = joinpath(dir, "$(dataset).rows")
        CSV.write(info_file, DataFrame(idx=idx_keep_rows); writeheader=false, delim=',')
    end

    # Clean up.
    rm(tmpdir, recursive=true)

    return nothing
end

push!(
    MESSAGES[],
    """
    ## Dataset: TCGA-HiSeq

    **5 classes / 801 instances / 20265 variables**

    See: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

    Original labels for genes can be found at https://www.synapse.org/#!Synapse:syn4301332.
    The relevant file is unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp, which has genes along rows and
    instance along columns.

    Extracting the correct gene labels requires cross-referencing Tissue Source Site (TSS) and Study codes.

    - TSS: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
    - Study: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations

    267 genes are dropped due to having zero expression through all samples.
    """
)

push!(REMOTE_PATHS[], "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz")

push!(CHECKSUMS[], "e6ea3628f9656efa2d7cf382bf5a403e2cc214df3d90312b1928f5650b43a559")

push!(FETCH_METHODS[], DataDeps.fetch_default)

push!(POST_FETCH_METHODS[], path -> process_TCGA_HiSeq(path, "TCGA-HiSeq"))

push!(DATASETS[], "TCGA-HiSeq")
