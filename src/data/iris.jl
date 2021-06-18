using DataDeps

register(DataDep(
  "iris",
  """
  Dataset: iris
  Author: R. A. Fisher (donated by Michael Marshall)
  Website: https://archive.ics.uci.edu/ml/datasets/iris

  Observations: 150
  Features:     4
  Classes:      3
  """,
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
  "6f608b71a7317216319b4d27b4d9bc84e6abd734eda7872b71a458569e2656c0",
  post_fetch_method = (path -> begin
    SparseSVM.process_dataset(path,
      target_index=5,
      feature_indices=1:4,
    )
  end),
))
