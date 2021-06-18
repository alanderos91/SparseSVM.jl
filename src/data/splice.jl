using DataDeps

register(DataDep(
  "splice",
  """
  Dataset: splice
  Donors: G. Towell, M. Noordewier, and J. Shavlik
  Website (Original): https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)
  Obtained from: https://www.openml.org/d/40670
  See also: https://www.rdocumentation.org/packages/mlbench/versions/2.1-1/topics/DNA

  Observations: 3186
  Features:     181
  Classes:      3
  """,
  "https://www.openml.org/data/get_csv/4965245/dna.arff",
  "2775c83180277226d445e66acdc6370f298e5d4811ef20b8ce9d111485600b5c",
  post_fetch_method = (path -> begin
    SparseSVM.process_dataset(path,
      header=true,
      target_index=181,
      feature_indices=1:180,
      ext=".csv",
    )
  end),
))
