using DataDeps

register(DataDep(
  "breast-cancer-wisconsin",
  """
  Dataset: breast-cancer-wisconsin
  Author: Dr. WIlliam H. Wolberg
  Donors: Olvi Mangasarian
    Received by David W. Aha

  Observations: 699
  Features:     10
  Classes:      2
  """,
  "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
  "402c585309c399237740f635ef9919dc512cca12cbeb20de5e563a4593f22b64",
  post_fetch_method = (path -> begin
    SparseSVM.process_dataset(path,
      header=false,
      target_index=11,
      feature_indices=2:10,
      ext=".csv",
    )
  end),
))
