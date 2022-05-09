using Base: BinaryPlatforms
using SparseSVM, KernelFunctions
using SparseSVM: nsamples, nfeatures, get_design_matrix, project_sparsity_set!
using LinearAlgebra, Random, Test

@testset "SVMBatch" begin
  (m, n) = (100, 50)
  X = randn(m, n)
  y = rand((-1.0, 1.0), m)
  label2target = Dict(-1.0 => "B", 1.0 => "A")

  ### linear case ###
  data = SVMBatch(X, y, label2target)

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
  data = SVMBatch(X, y, label2target, κ)

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

  ### error handling ###
  Xint = rand(Int, m, n)
  yint = rand((-1, 1), m)
  label2target = Dict(-1 => "B", 1 => "A")

  @test_throws MethodError SVMBatch(X, yint, label2target)                        # X and y must have floating point elements
  @test_throws MethodError SVMBatch(Xint, y, label2target)
  @test_throws MethodError SVMBatch(Xint, yint, label2target)
  @test_throws DomainError SVMBatch(randn(m, n), rand(m), label2target)           # y must entries in {-1, 1}
  @test_throws DimensionMismatch SVMBatch(randn(m, n), rand(m+1), label2target)   # X and y must have compatible dimensions
  @test_throws DimensionMismatch SVMBatch(randn(m+1, n), rand(m), label2target)
end

@testset "BinaryClassifier" begin
  (m, n) = (100, 50)
  X = randn(m, n)
  targets1 = rand((-1.0, 1.0), m)
  targets2 = rand(("A", "B"), m)
  targets3 = rand(("A", "B", "C"), m)

  ### check targets and codings
  function check_mapping(c::BinaryClassifier{T}, input_targets, refclass, otherclass) where T
    # check that positive and negative labels are set correctly
    @test c.data.label2target[one(T)] == refclass
    @test c.data.label2target[-one(T)] == otherclass

    # verify coding in y
    invlabel = c.data.label2target
    idxA = findall(>(0), c.data.y)
    idxB = findall(<(0), c.data.y)
    truth = [t == refclass ? refclass : otherclass for t in input_targets]
    @test all(i -> truth[i] == refclass, idxA)
    @test all(i -> truth[i] == otherclass, idxB)
  end
  
  ##### case: using -1.0 and 1.0 as targets directly
  check_mapping(BinaryClassifier(X, targets1, -1.0), targets1, 1.0, -1.0)      # always select 1.0 for refclass in {-1,1} targets
  check_mapping(BinaryClassifier(X, targets1, 1.0), targets1, 1.0, -1.0)

  ##### case: using "A" and "B" as targets
  check_mapping(BinaryClassifier(X, targets2, "A"), targets2, "A", "B")        # select A
  check_mapping(BinaryClassifier(X, targets2, "B"), targets2, "B", "A")        # select B

  ##### case: using A and not A with multiclass targets
  check_mapping(BinaryClassifier(X, targets3, "A"), targets3, "A", "not_A")    # select A

  ##### case: reference class does not appear in targets - throw an error message
  @test_throws ErrorException BinaryClassifier(X, targets1, 2.0)
  @test_throws ErrorException BinaryClassifier(X, targets2, "C")
  @test_throws ErrorException BinaryClassifier(X, targets3, "D")

  ### floating point conversion ###
  function ftype_linear_case(c::BinaryClassifier, T)
    @test eltype(c.weights) == T
    @test eltype(c.data.X) == T
    @test eltype(c.data.y) == T
  end

  function ftype_nonlinear_case(c::BinaryClassifier, T)
    @test eltype(c.weights) == T
    @test eltype(c.data.X) == T
    @test eltype(c.data.y) == T
    @test eltype(c.data.K) == T
  end

  # eltype should default to Float64
  ftype_linear_case(BinaryClassifier(X, targets2, "A"), Float64)
  ftype_nonlinear_case(BinaryClassifier(X, targets2, "A", kernel=RBFKernel()), Float64)

  # otherwise, convert to the specified type
  for T in (Float16, Float32, Float64)
    ftype_linear_case(BinaryClassifier{T}(X, targets2, "A"), T)
    ftype_nonlinear_case(BinaryClassifier{T}(X, targets2, "A", kernel=RBFKernel()), T)
  end

  ### linear case ###

  # w/o intercept
  classifier = BinaryClassifier(X, targets2, "A", intercept=false)
  @test classifier.intercept == false
  @test length(classifier.weights) == n

  # w/ intercept
  classifier = BinaryClassifier(X, targets2, "A", intercept=true)
  @test classifier.intercept == true
  @test length(classifier.weights) == n + 1

  ### nonlinear case ###

  # w/o intercept
  classifier = BinaryClassifier(X, targets2, "A", kernel=RBFKernel(), intercept=false)
  @test classifier.intercept == false
  @test length(classifier.weights) == m

  # w/ intercept
  classifier = BinaryClassifier(X, targets2, "A", kernel=RBFKernel(), intercept=true)
  @test classifier.intercept == true
  @test length(classifier.weights) == m + 1
end

@testset "Projection" begin
  n = 100
  x = zeros(n)
  idx = collect(1:n)
  idx_buffer = similar(idx)

  # case: more nonzeros than k
  k = 10
  for i in eachindex(x)
    x[i] = rand() > 0.5 ? -i : i
  end
  shuffle!(x)
  perm = sortperm(x, rev=true, by=abs)
  xproj = project_sparsity_set!(copy(x), idx, k, idx_buffer)
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
  xproj = project_sparsity_set!(copy(x), idx, k, idx_buffer)
  @test all(i -> xproj[i] == x[i], 1:n)

  # case: exactly k nonzeros
  fill!(x, 0)
  for i in 1:k
    x[i] = rand() > 0.5 ? -i : i
  end
  shuffle!(x)
  perm = sortperm(x, rev=true, by=abs)
  xproj = project_sparsity_set!(copy(x), idx, k, idx_buffer)
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