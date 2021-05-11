using Random, Statistics, DataFrames, FileIO

# Parse inputs.
n = parse(Int, ARGS[1])
p = parse(Int, ARGS[2])
d = parse(Int, ARGS[3])
seed = parse(Int, ARGS[4])
dname = ARGS[5]

# Set RNG with seed.
rng = MersenneTwister(seed)

# Simulate data.
X = zeros(n, p)         # n samples, p features
for xi in eachrow(X)
    z = rand((-1,1))
    xi[:] .= z .+ randn(rng, p)
end
μ = mean(X, dims=1)     # mean of each feature
σ = std(X, dims=1)      # standard deviation of each feature
@. X = (X - μ) / σ      # standardize
X = [X ones(n)]         # add column of 1s for bias

J = unique!(rand(rng, 1:p, d))
while length(J) < d
    Δ = setdiff(1:p, J)
    global J = [J; rand(rng, Δ, d - length(J))]
    unique!(J)
end
sort!(J)

β = zeros(p+1)              # coefficients + bias; bias = 0
β[J] .= 2 .+ randn(rng, d)   # sample from N(2,1)
y = sign.(X*β)              # simulate classes

# Write to file.
data = [X[:, 1:p] y]
DataFrame(data, :auto) |> FileIO.save("data/$(dname).csv")
DataFrame([J β[J]], [:index, :coefficient]) |> FileIO.save("data/$(dname)-info.csv")
