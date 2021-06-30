using DataDeps

register(DataDep(
    "optdigits",
    """
    Dataset: optdigits
    Donors: E. Alpaydin, C. Kaynak
    Department of Computer Engineering
    Bogazici University, 80815 Istanbul Turkey
    Website: https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

    Observations: 5620
    Features:     64
    Classes:      10
    """,
    [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
    ],
    "8f0dbee3cf326eb016eef623a392c9f92fc15e201c06c5104a1500828289af65",
    #
    # Disclaimer: Want to combine both .tra and .tes into a single .csv.
    # Design of post_fetch_method may not have been designed with this in mind.
    # Workaround here is to do nothing to first file, then assume first file has been
    # downloaded and can be loaded in the second function.
    # This works in practice, but will fail when running preupload_check.
    # 
    # Reason: `path` is randomly generated and differs between the two calls.
    #
    post_fetch_method = [
    identity,
    path -> begin
        # Read both training and testing files.
        dir = dirname(path)
        tra = joinpath(dir, "optdigits.tra")
        tes = joinpath(dir, "optdigits.tes")
        df = vcat(
            CSV.read(tra, DataFrame, header=false),
            CSV.read(tes, DataFrame, header=false)
        )
        
        # Process the data as usual.
        SparseSVM.process_dataset(df,
            target_index=ncol(df),
            feature_indices=1:ncol(df)-1,
            ext=".csv",
        )

        # Clean up by removing separate training and testing files.
        rm(tra)
        rm(tes)
    end],
))
