using Random, Statistics, DataFrames, FileIO

# Parse inputs.
n = parse(Int, ARGS[1])
p = parse(Int, ARGS[2])
d = parse(Int, ARGS[3])
seed = parse(Int, ARGS[4])
dname = ARGS[5]

# Simulate data.
X = randn(n, p)         # n samples, p features
μ = mean(X, dims=1)     # mean of each feature
σ = std(X, dims=1)      # standard deviation of each feature
@. X = (X - μ) / σ      # standardize
X = [X ones(n)]         # add column of 1s for bias

J = unique!(rand(1:p, d))
while length(J) < d
    Δ = setdiff(1:p, J)
    global J = [J; rand(Δ, d - length(J))]
    unique!(J)
end
sort!(J)

β = zeros(p+1)          # coefficients + bias; bias = 0
β[J] .= randn(d)        # sample from standard normal
y = sign.(X*β)          # simulate classes

# Write to file.
data = [X[:, 1:p] y]
DataFrame(data, :auto) |> FileIO.save("data/$(dname).csv")
DataFrame([J β[J]], [:index, :coefficient]) |> FileIO.save("data/$(dname)-info.csv")
