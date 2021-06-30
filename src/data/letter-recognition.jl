using DataDeps

register(DataDep(
    "letter-recognition",
    """
    Dataset: letter-recognition
    Author: David J. Slate
    Odesta Corporation; 1890 Maple Ave; Suite 115; Evanston, IL 60201 
    Website: https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

    Observations: 20000
    Features:     16
    Classes:      26
    """,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data",
    "2b89f3602cf768d3c8355267d2f13f2417809e101fc2b5ceee10db19a60de6e2",
    post_fetch_method = (path -> begin
        SparseSVM.process_dataset(path,
            header=false,
            target_index=1,
            feature_indices=2:17,
            ext=".csv.gz",
        )
    end),
))
