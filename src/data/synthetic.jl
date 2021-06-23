using DataDeps

register(DataDep(
  "synthetic",
  """
  Dataset: synthetic

  A simulated multivariate normal.
  Classes are determined by the first two dimensions, which correspond to signs of X*beta with beta[1] = 10, beta[2] = -10.
  Covariance structure is as follows:

  Σ[1,1] = 1, Σ[2,2] = 3, Σ[i,i] = 2
  Σ[1,2] = 0.9
  Σ[i,j] = 1e-3*randn()

  Observations: 1000
  Features:     500
  Classes:      2
  """,
  "script", # nothing to download
  "514ef72019b85634859baf459b38f37295d5fcc84d06be7d8e4a71a99a74cbdf";
  fetch_method=function(unused, localdir)
    #
    rng = MersenneTwister(2000)
    (m, n) = (10^3, 500)

    # covariance matrix
    Σ = Matrix{Float64}(2*I, n, n)
    for j in 1:n, i in j+1:n
        Σ[j,i] = 1e-3*randn(rng)
    end
    Σ[1,1] = 1.0
    Σ[2,2] = 3.0
    Σ[1,2] = 0.9

    L, _ = cholesky(Symmetric(Σ))

    # simulate data
    X = Matrix{Float64}(undef, m, n)
    for i in axes(X, 1)
        @views X[i, :] .= L*randn(rng, n)
    end

    # coefficients
    beta = zeros(n)
    beta[1] = 10.0
    beta[2] = -10.0

    # targets
    y = sign.(X*beta)
    target = map(yi -> yi > 0 ? 'A' : 'B', y)

    local_file = joinpath(localdir, "data.csv")
    df = hcat(DataFrame(target=target), DataFrame(X, :auto))
    CSV.write(local_file, df)
    return local_file
  end
))
