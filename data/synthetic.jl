using Random, DataFrames, FileIO
Random.seed!(5357)
(m, n) = (10^3, 500)
y = randn(m);
@. y = sign(y);
X = randn(m, n) .+ y; # this should give two 'nicely' separated Guassian clusters

DataFrame([X y], :auto) |> FileIO.save("data/synthetic.csv")
