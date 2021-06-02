using SparseSVM, KernelFunctions
using SparseSVM: nsamples, nfeatures, get_design_matrix, project_sparsity_set!
using LinearAlgebra, Random, Test

@testset "SVMBatch" begin
  (m, n) = (100, 50)
  X = randn(m, n)
  y = rand((-1.0, 1.0), m)

  ### linear case ###
  data = SVMBatch(X, y)

  # default: do not build K matrix
  @test data.K === nothing

  # dimensions should match X matrix
  @test nsamples(data) == m
  @test nfeatures(data) == n

  # design matrix should be m × n (no intercept) or m × (n+1) (intercept)
  A = get_design_matrix(data, false)
  @test size(A, 1) == nsamples(data)
  @test size(A, 2) == nfeatures(data)
  @test A == data.X

  A = get_design_matrix(data, true)
  @test size(A, 1) == nsamples(data)
  @test size(A, 2) == nfeatures(data) + 1
  @test A == [data.X ones(m)]

  ### nonlinear case ###
  κ = RBFKernel()
  data = SVMBatch(X, y, kernelf=κ)

  # kernel matrix should match with chosen kernel function
  @test typeof(data.kernelf) == typeof(κ)
  @test data.K == kernelmatrix(κ, X, obsdim=1)

  # dimensions should match X matrix
  @test nsamples(data) == m
  @test nfeatures(data) == n

  # design matrix should be m × m (no intercept) or m × (m+1) (intercept)
  A = get_design_matrix(data, false)
  @test size(A, 1) == nsamples(data)
  @test size(A, 2) == nsamples(data)
  @test A == data.K * Diagonal(data.y)

  A = get_design_matrix(data, true)
  @test size(A, 1) == nsamples(data)
  @test size(A, 2) == nsamples(data) + 1
  @test A == [data.K * Diagonal(data.y) ones(m)]

  ### floating point conversion ###
  for ftype in (Float16, Float32, Float64)
    @test eltype(SVMBatch(X, y, ftype=ftype).X) == ftype
    @test eltype(SVMBatch(X, y, ftype=ftype).y) == ftype
    @test eltype(SVMBatch(X, y, kernelf=RBFKernel(), ftype=ftype).K) == ftype
  end

  ### error handling ###
  Xint = rand(Int, m, n)
  yint = rand((-1, 1), m)

  @test_throws MethodError SVMBatch(X, yint)                        # X and y must have floating point elements
  @test_throws MethodError SVMBatch(Xint, y)
  @test_throws MethodError SVMBatch(Xint, yint)
  @test_throws DomainError SVMBatch(randn(m, n), rand(m))           # y must entries in {-1, 1}
  @test_throws DimensionMismatch SVMBatch(randn(m, n), rand(m+1))   # X and y must have compatible dimensions
  @test_throws DimensionMismatch SVMBatch(randn(m+1, n), rand(m))
end

@testset "BinaryClassifier" begin
  (m, n) = (100, 50)
  X = randn(m, n)
  y = rand((-1.0, 1.0), m)

  ### linear case ###

  # w/o intercept
  classifier = BinaryClassifier(SVMBatch(X, y), intercept=false)
  @test classifier.intercept == false
  @test length(classifier.weights) == n

  # w/ intercept
  classifier = BinaryClassifier(SVMBatch(X, y), intercept=true)
  @test classifier.intercept == true
  @test length(classifier.weights) == n + 1

  ### nonlinear case ###

  # w/o intercept
  classifier = BinaryClassifier(SVMBatch(X, y, kernelf=RBFKernel()), intercept=false)
  @test classifier.intercept == false
  @test length(classifier.weights) == m

  # w/ intercept
  classifier = BinaryClassifier(SVMBatch(X, y, kernelf=RBFKernel()), intercept=true)
  @test classifier.intercept == true
  @test length(classifier.weights) == m + 1
end

@testset "Projection" begin
  n = 100
  x = zeros(n)
  idx = collect(1:n)

  # case: more nonzeros than k
  k = 10
  for i in eachindex(x)
    x[i] = rand() > 0.5 ? -i : i
  end
  shuffle!(x)
  perm = sortperm(x, rev=true, by=abs)
  xproj = project_sparsity_set!(copy(x), idx, k)
  @test all(i -> xproj[perm[i]] == x[perm[i]], 1:k)
  @test all(i -> xproj[perm[i]] == 0, k+1:n)

  # case: less nonzeros than k; no change
  c = round(Int, 0.5*k)
  fill!(x, 0)
  for i in 1:c
    x[i] = rand() > 0.5 ? -i : i
  end
  shuffle!(x)
  perm = sortperm(x, rev=true, by=abs)
  xproj = project_sparsity_set!(copy(x), idx, k)
  @test all(i -> xproj[i] == x[i], 1:n)

  # case: exactly k nonzeros
  fill!(x, 0)
  for i in 1:k
    x[i] = rand() > 0.5 ? -i : i
  end
  shuffle!(x)
  perm = sortperm(x, rev=true, by=abs)
  xproj = project_sparsity_set!(copy(x), idx, k)
  @test all(i -> xproj[i] == x[i], 1:n)

  # helper function should give length n vector without intercept
  p = randn(n)
  pvec, idx = SparseSVM.get_model_coefficients(p, false)
  @test length(pvec) == n
  @test all(i -> p[i] == pvec[i], 1:length(pvec))

  # helper function should give view into p with intercept
  p = randn(n)
  pvec, idx = SparseSVM.get_model_coefficients(p, true)
  @test length(pvec) == n-1
  @test all(i -> p[i] == pvec[i], 1:length(pvec))
end