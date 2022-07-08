function synthetic(a,b,c,d,e; rng::AbstractRNG=StableRNG(2000), prob::Real=1.0, m::Int=10^3, n::Int=500)
    if n < 2
        error("Number of features (n=$(n)) should be greater than 2.")
    end
    if prob < 0 || prob > 1
        error("Probability (prob=$(prob)) must satisfy 0 ≤ prob ≤ 1.")
    end

    # covariance matrix
    Σ = Matrix{Float64}(c*I, n, n)
    for j in 1:n, i in j+1:n
        Σ[j,i] = e
    end
    Σ[1,1] = a
    Σ[2,2] = b

    # Cluster A
    Σ1 = copy(Σ)
    Σ1[1,2] = +d

    # Cluster B
    Σ2 = copy(Σ)
    Σ2[1,2] = -d

    if !isposdef(Symmetric(Σ1)) || !isposdef(Symmetric(Σ2))
        error("At least one covariance matrix is not positive definite. Try decreasing parameters d or e relative to a, b, and c.")
    end

    # Simulate instances.
    X = Matrix{Float64}(undef, m, n)
    L1, _ = cholesky(Symmetric(Σ1))
    L2, _ = cholesky(Symmetric(Σ2))
    cluster = Vector{String}(undef, m)
    for i in axes(X, 1)
        # Sample features from Class A
        if rand(rng) > 0.5
            @views X[i, :] .= L1*randn(rng, n)
            cluster[i] = "A"
        else # Class B
            @views X[i, :] .= L2*randn(rng, n)
            cluster[i] = "B"
        end
    end
    
    StatsBase.transform!(StatsBase.fit(ZScoreTransform, X, dims=1), X)

    # Set coefficients
    beta = zeros(n)
    beta[1] = 10.0
    beta[2] = -10.0

    # Assign labels.
    y, L = Vector{Float64}(undef, m), Vector{String}(undef, m)
    inversions = 0
    for i in eachindex(y)
        xi = view(X, i, :)
        yi = sign(xi'*beta)
        if rand(rng) < prob
            y[i], L[i] = yi, ifelse(yi == 1, "A", "B")
        else
            y[i], L[i] = ifelse(cluster[i] == "A", 1, -1), cluster[i]
            inversions += 1
        end
    end
    println("[  synthetic: $(m) instances / $(n) features  ]")
    println("  ∘ Pr(y | x) = $(prob)")
    println("  ∘ $inversions class inversions ($(inversions/m) Bayes error)")
    return L, X
end

function spiral(class_sizes;
        rng::AbstractRNG=StableRNG(1903),
        max_radius::Real=7.0,
        x0::Real=-3.5,
        y0::Real=3.5,
        angle_start::Real=π/8,
        prob::Real=1.0,
    )
    if length(class_sizes) != 3
        error("Must specify 3 classes (length(class_sizes)=$(length(class_sizes))).")
    end
    if max_radius <= 0
        error("Maximum radius (max_radius=$(max_radius)) must be > 0.")
    end
    if angle_start < 0
        error("Starting angle (angle_start=$(angle_start)) should satisfy 0 ≤ θ ≤ 2π.")
    end
    if prob < 0 || prob > 1
        error("Probability (prob=$(prob)) must satisfy 0 ≤ prob ≤ 1.")
    end

    # Extract parameters.
    N = sum(class_sizes)
    max_A, max_B, max_C = class_sizes

    # Simulate the data.
    L, X = Vector{String}(undef, N), Matrix{Float64}(undef, N, 2)
    x, y = view(X, :, 1), view(X, :, 2)
    inversions = 0
    for i in 1:N
        if i ≤ max_A
            (class, k, n, θ) = ("A", i, max_A, angle_start)
            noise = 0.1
        elseif i ≤ max_A + max_B
            (class, k, n, θ) = ("B", i-max_A+1, max_B, angle_start + 2π/3)
            noise = 0.2
        else
            (class, k, n, θ) = ("C", i-max_A-max_B+1, max_C, angle_start + 4π/3)
            noise = 0.3
        end

        # Compute coordinates.
        angle = θ + π * k / n
        radius = max_radius * (1 - k / (n + n / 5))

        x[i] = x0 + radius*cos(angle) + noise*randn(rng)
        y[i] = y0 + radius*sin(angle) + noise*randn(rng)
        if rand(rng) < prob
            L[i] = class
        else
            L[i] = rand(rng, setdiff(["A", "B", "C"], [class]))
            inversions += 1
        end
    end

    println("[  spiral: $(N) instances / 2 features / 3 classes  ]")
    println("  ∘ Pr(y | x) = $(prob)")
    println("  ∘ $inversions class inversions ($(inversions/N) Bayes error)")

    return L, X
end