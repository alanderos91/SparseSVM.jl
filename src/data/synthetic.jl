using DataDeps

register(DataDep(
    "synthetic",
    """
    Dataset: synthetic

    A simulated multivariate normal. Bayes error is approximately 0%.

    Classes are determined by the first two dimensions, which correspond to signs of X*beta with beta[1] = 10, beta[2] = -10.
    Covariance structure is as follows:

    Σ[1,1] = 1
    Σ[2,2] = 1
    Σ[i,i] = 1e-2 for i=3, 4, …, 500
    Σ[1,2] = 0.4
    Σ[i,j] = 1e-4

    Observations: 1000
    Features:     500
    Classes:      2
    """,
    "script", # nothing to download
    "678860f52d913c30233a148cf91ca014d1a7b31a40b62503e76dc4e378d047f2";
    fetch_method=function(unused, localdir)
        L, X = synthetic(1,1,1e-2,0.4,1e-4; rng=StableRNG(2000), prob=1.0, m=10^3, n=500)

        local_file = joinpath(localdir, "data.csv")
        df = hcat(DataFrame(target=L), DataFrame(X, :auto))
        CSV.write(local_file, df)
        return local_file
    end
))

register(DataDep(
    "synthetic-be=0.1",
    """
    Dataset: synthetic-be=0.1

    A simulated multivariate normal. Bayes error is approximately 10%.

    Classes are determined by the first two dimensions, which correspond to signs of X*beta with beta[1] = 10, beta[2] = -10.
    Covariance structure is as follows:

    Σ[1,1] = 1
    Σ[2,2] = 1
    Σ[i,i] = 1e-2 for i=3, 4, …, 500
    Σ[1,2] = 0.4
    Σ[i,j] = 1e-4

    Observations: 1000
    Features:     500
    Classes:      2
    """,
    "script", # nothing to download
    "8fa43851a77d2d5bd5e9cecb5b7cf696179f76103d36e92cc146fee93d53b891";
    fetch_method=function(unused, localdir)
        L, X = synthetic(1,1,1e-2,0.4,1e-4; rng=StableRNG(2000), prob=0.9, m=10^3, n=500)

        local_file = joinpath(localdir, "data.csv")
        df = hcat(DataFrame(target=L), DataFrame(X, :auto))
        CSV.write(local_file, df)
        return local_file
    end
))
