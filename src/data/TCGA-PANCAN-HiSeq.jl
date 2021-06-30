using DataDeps
using CSV, DataFrames

register(DataDep(
    "TCGA-PANCAN-HiSeq",
    """
    Dataset: TCGA-PANCAN-HiSeq
    Author: Samuele Fiorini, University of Genoa
    Website: https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

    Redistributed under Creative Commons license (http://creativecommons.org/licenses/by/3.0/legalcode)

    Observations: 801
    Features:     20531
    Classes:      5
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz",
    "e6ea3628f9656efa2d7cf382bf5a403e2cc214df3d90312b1928f5650b43a559",
    post_fetch_method = (path -> begin
        DataDeps.unpack(path, keep_originals=false)

        # Read the data and labels files.
        tmpdir = "TCGA-PANCAN-HiSeq-801x20531"
        data = CSV.read(joinpath(tmpdir, "data.csv"), DataFrame, header=true)
        labels = CSV.read(joinpath(tmpdir, "labels.csv"), DataFrame, header=true)

        # Create temporary version with columns Column1, Class, gene_0, gene_1, ...
        tmpdf = innerjoin(labels, data, on=:Column1)

        # Process the file as usual.
        SparseSVM.process_dataset(tmpdf,
            target_index=2,
            feature_indices=3:ncol(tmpdf),
            ext=".csv.gz"
        )

        # Clean up.
        rm(tmpdir, recursive=true)
    end),
))
