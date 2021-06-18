using DataDeps

register(DataDep(
  "spiral",
  """
  Dataset: spiral
  Credit: https://smorbieu.gitlab.io/generate-datasets-to-understand-some-clustering-algorithms-behavior/

  A simulated dataset of three noisy spirals.

  Observations: 1000
  Features:     2
  Classes:      3
  """,
  "script", # nothing to download
  "abe3846fe75d15bb86985ab61d8b94112e4230fa4db732a56018a76146b5e946";
  fetch_method=function(unused, localdir)
    # Parameters
    N           = 1000
    max_A       = 600
    max_B       = 300
    max_C       = N - max_A - max_B
    max_radius  = 7.0
    x0          = -3.5
    y0          = 3.5
    angle_start = π / 8
    seed        = 1903

    # Simulate the data.
    rng = MersenneTwister(seed)
    target, x, y = Vector{Char}(undef, N), zeros(N), zeros(N)
    for i in 1:N
      if i ≤ max_A
        (class, k, n, θ) = ('A', i, max_A, angle_start)
        noise = 0.1
      elseif i ≤ max_A + max_B
        (class, k, n, θ) = ('B', i-max_A+1, max_B, angle_start + 2π/3)
        noise = 0.2
      else
        (class, k, n, θ) = ('C', i-max_A-max_B+1, max_C, angle_start + 4π/3)
        noise = 0.3
      end
      # Compute coordinates.
      angle = θ + π * k / n
      radius = max_radius * (1 - k / (n + n / 5))

      target[i] = class
      x[i] = x0 + radius*cos(angle) + noise*randn(rng)
      y[i] = y0 + radius*sin(angle) + noise*randn(rng)
    end
    local_file = joinpath(localdir, "data.csv")
    df = DataFrame(target=target, x1=x, x2=y)
    perm = Random.randperm(rng, size(df, 1))
    foreach(col -> permute!(col, perm), eachcol(df))
    CSV.write(local_file, df)
    return local_file
  end
))
