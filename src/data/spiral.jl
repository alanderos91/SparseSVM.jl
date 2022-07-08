using DataDeps

register(DataDep(
    "spiral",
    """
    Dataset: spiral
    Credit: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

    A simulated dataset of three noisy spirals. Bayes error is approximately 0%.

    Observations: 1000
    Features:     2
    Classes:      3
    """,
    "spiral", # nothing to download
    "f69c1ca22b338e7df6cd8501bca2bfafa65c65a79b2a16568a60f9b2bea848bc";
    fetch_method=function(unused, localdir)
        rng = StableRNG(1903)
        L, X = spiral((600, 300, 100);
            rng=rng,
            max_radius=7.0,
            x0=-3.5,
            y0=3.5,
            angle_start=pi/8,
            prob=1.0,
        )

        x, y = view(X, :, 1), view(X, :, 2)
        local_file = joinpath(localdir, "data.csv")
        df = DataFrame(target=L, x1=x, x2=y)
        perm = Random.randperm(rng, size(df, 1))
        foreach(col -> permute!(col, perm), eachcol(df))
        CSV.write(local_file, df)
        return local_file
    end
))

register(DataDep(
    "spiral-be=0.1",
    """
    Dataset: spiral-be=0.1
    Credit: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

    A simulated dataset of three noisy spirals. Bayes error is approximately 10%.

    Observations: 1000
    Features:     2
    Classes:      3
    """,
    "spiral-be=0.1", # nothing to download
    "8ea571992e2c21b10d6ef6ed9f8ab15c6df0cba2b27324f28a10535848f4ca5f";
    fetch_method=function(unused, localdir)
        rng = StableRNG(1903)
        L, X = spiral((600, 300, 100);
            rng=rng,
            max_radius=7.0,
            x0=-3.5,
            y0=3.5,
            angle_start=pi/8,
            prob=0.9,
        )

        x, y = view(X, :, 1), view(X, :, 2)
        local_file = joinpath(localdir, "data.csv")
        df = DataFrame(target=L, x1=x, x2=y)
        perm = Random.randperm(rng, size(df, 1))
        foreach(col -> permute!(col, perm), eachcol(df))
        CSV.write(local_file, df)
        return local_file
    end
))
