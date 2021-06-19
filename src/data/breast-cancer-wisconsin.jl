using DataDeps

register(DataDep(
  "breast-cancer-wisconsin",
  """
  Dataset: breast-cancer-wisconsin
  Author: Dr. WIlliam H. Wolberg
  Donors: Olvi Mangasarian
    Received by David W. Aha
  Website: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original)

  Observations: 699 (16 missing)
  Features:     9
  Classes:      2
  """,
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
  "402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64",
  post_fetch_method = (path -> begin
    SparseSVM.process_dataset(path,
      missingstrings=["?"],
      header=false,
      target_index=11,
      feature_indices=2:10,
      ext=".csv",
    )
  end),
))
