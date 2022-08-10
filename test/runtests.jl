using Base: BinaryPlatforms
using SparseSVM, KernelFunctions
using LinearAlgebra, Random, Test

SVM = SparseSVM

@testset "MultiClassStrategy" begin
    @test SVM.determine_number_svms(OVO(), 10) == 45
    @test SVM.determine_number_svms(OVR(), 10) == 10
end

@testset "automatic OneVsRest encodings" begin
    @test SVM.has_inferrable_encoding(Int) == true
    @test SVM.has_inferrable_encoding(String) == true
    @test SVM.has_inferrable_encoding(Symbol) == true
    @test SVM.has_inferrable_encoding(Char) == false
end

@testset "design matrix" begin
    y, data = rand((-1.0, 1.0), 20), randn(20, 5)

    X, KY = SVM.create_X_and_K(nothing, y, data)
    @test X == data
    @test (KY isa Nothing) == true

    X, KY = SVM.create_X_and_K(nothing, y, data)
    @test X == data
    @test (KY isa Nothing) == true

    X, KY = SVM.create_X_and_K(RBFKernel(), y, data)
    @test X == data
    @test KY ≈ kernelmatrix(RBFKernel(), data, obsdim=1) * Diagonal(y)

    X, KY = SVM.create_X_and_K(RBFKernel(), y, data)
    @test X == data
    @test KY ≈ kernelmatrix(RBFKernel(), data, obsdim=1) * Diagonal(y)
end

@testset "BinarySVMProblem" begin
    n, p = 100, 5

    @testset "encoding w/ 2 categories" begin
        labeled_data = (rand(("a", "b"), n), randn(n, p)) # tuple of labels and data
        linear_classifier(pl, nl=nothing) = BinarySVMProblem(labeled_data..., pl,
            negative_label=nl,
            intercept=false,
            kernel=nothing,
        )

        @testset "error cases" begin
            @test_throws ErrorException linear_classifier('a')
            @test_throws ErrorException linear_classifier("x")
        end

        @testset "defaults" begin
            prob = linear_classifier("a", nothing)
            @test SVM.poslabel(prob) == "a"
            @test SVM.neglabel(prob) == "b"
            @test SVM.__classify__(prob,  1) == "a"
            @test SVM.__classify__(prob, -1) == "b"
        end

        @testset "labels w/ specified negative_label" begin
            prob = linear_classifier("a", "b")
            @test SVM.poslabel(prob) == "a"
            @test SVM.neglabel(prob) == "b"
            @test SVM.__classify__(prob,  1) == "a"
            @test SVM.__classify__(prob, -1) == "b"

            prob = linear_classifier("a", "something_else")
            @test SVM.poslabel(prob) == "a"
            @test SVM.neglabel(prob) == "something_else"
            @test SVM.__classify__(prob,  1) == "a"
            @test SVM.__classify__(prob, -1) == "something_else"
        end

        @testset "switch positive label" begin
            prob = linear_classifier("b", nothing)
            @test SVM.poslabel(prob) == "b"
            @test SVM.neglabel(prob) == "a"
            @test SVM.__classify__(prob,  1) == "b"
            @test SVM.__classify__(prob, -1) == "a"
        end
    end

    @testset "encoding w/ >2 categories" begin
        labeled_data = (rand(("a", "b", "c"), n), randn(n, p))
        linear_classifier(pl, nl=nothing) = BinarySVMProblem(labeled_data..., pl,
            negative_label=nl,
            intercept=false,
            kernel=nothing
        )

        @testset "error cases" begin
            @test_throws ErrorException linear_classifier("a", "b")
        end

        @testset "defaults" begin
            prob = linear_classifier("a", nothing)
            @test SVM.poslabel(prob) == "a"
            @test SVM.neglabel(prob) == "not_a"
            @test SVM.__classify__(prob,  1) == "a"
            @test SVM.__classify__(prob, -1) == "not_a"
        end

        @testset "labels w/ specified negative_label" begin
            prob = linear_classifier("a", "!")
            @test SVM.poslabel(prob) == "a"
            @test SVM.neglabel(prob) == "!"
            @test SVM.__classify__(prob,  1) == "a"
            @test SVM.__classify__(prob, -1) == "!"
        end

        @testset "switch positive label" begin
            prob = linear_classifier("b", nothing)
            @test SVM.poslabel(prob) == "b"
            @test SVM.neglabel(prob) == "not_b"
            @test SVM.__classify__(prob,  1) == "b"
            @test SVM.__classify__(prob, -1) == "not_b"
        end
    end

    @testset "linear case" begin
        labeled_data = (rand(("a", "b"), n), randn(n, p)) # tuple of labels and data
        function test_classifier(intercept)
            prob = BinarySVMProblem(labeled_data..., "a", intercept=intercept, kernel=nothing)
            T = SVM.floattype(prob)
            X = SVM.get_design_matrix(prob)

            @test labeled_data[2] == X
            @test (prob.KY isa Nothing) == true
            @test sort!(unique(prob.y)) == T[-1, 1]
            @test SVM.probdims(prob) == (n, p, 2)
            @test prob.intercept == intercept
            @test all(arr -> length(arr) == p+intercept, (prob.coeff, prob.coeff_prev, prob.proj, prob.grad))
            @test length(prob.res.main) == n
            @test length(prob.res.dist) == p+intercept

            return nothing
        end

        @testset "w/ intercept" begin
            test_classifier(true)
        end

        @testset "w/o intercept" begin
            test_classifier(false)
        end
    end

    @testset "nonlinear case" begin
        labeled_data = (rand(("a", "b"), n), randn(n, p)) # tuple of labels and data
        function test_classifier(intercept)
            prob = BinarySVMProblem(labeled_data..., "a", intercept=intercept, kernel=RBFKernel())
            T = SVM.floattype(prob)
            KY = SVM.get_design_matrix(prob)

            @test KY ≈ kernelmatrix(RBFKernel(), labeled_data[2], obsdim=1) * Diagonal(prob.y)
            @test prob.X == labeled_data[2]
            @test sort!(unique(prob.y)) == T[-1, 1]
            @test SVM.probdims(prob) == (n, p, 2)
            @test prob.intercept == intercept
            @test all(arr -> length(arr) == n+intercept, (prob.coeff, prob.coeff_prev, prob.proj, prob.grad))
            @test length(prob.res.main) == n
            @test length(prob.res.dist) == n+intercept

            return nothing
        end

        @testset "w/ intercept" begin
            test_classifier(true)
        end

        @testset "w/o intercept" begin
            test_classifier(false)
        end
    end
end

@testset "MultiSVMProblem" begin
    n, p, c = 200, 5, 5
    labeled_data = (rand(string.('a':'a'+4), n), randn(n, p))
    classifier(intercept, kernel, strategy) = MultiSVMProblem(labeled_data...,
        intercept=intercept,
        kernel=kernel,
        strategy=strategy,
    )

    @testset "One versus One (OVO)" begin
        prob = classifier(false, nothing, OVO())
        @test SVM.probdims(prob) == (n, p, c)
        @test length(prob.svm) == binomial(c, 2)
    end

    @testset "One versus Rest (OVR)" begin
        prob = classifier(false, nothing, OVR())
        @test SVM.probdims(prob) == (n, p, c)
        @test length(prob.svm) == c
    end
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
    xproj = SVM.project_l0_ball!(copy(x), idx, k, idx_buffer)
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
    xproj = SVM.project_l0_ball!(copy(x), idx, k, idx_buffer)
    @test all(i -> xproj[i] == x[i], 1:n)
    
    # case: exactly k nonzeros
    fill!(x, 0)
    for i in 1:k
        x[i] = rand() > 0.5 ? -i : i
    end
    shuffle!(x)
    perm = sortperm(x, rev=true, by=abs)
    xproj = SVM.project_l0_ball!(copy(x), idx, k, idx_buffer)
    @test all(i -> xproj[i] == x[i], 1:n)
end
