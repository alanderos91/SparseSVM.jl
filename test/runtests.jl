using Base: BinaryPlatforms
using SparseSVM, KernelFunctions
using LinearAlgebra, Random, Test

SSVM = SparseSVM

@testset "MultiClassStrategy" begin
    @test SSVM.determine_number_svms(OVO(), 10) == 45
    @test SSVM.determine_number_svms(OVR(), 10) == 10
end

@testset "automatic OneVsRest encodings" begin
    @test SSVM.has_inferrable_encoding(Int) == true
    @test SSVM.has_inferrable_encoding(String) == true
    @test SSVM.has_inferrable_encoding(Symbol) == true
    @test SSVM.has_inferrable_encoding(Char) == false
end

@testset "design matrix" begin
    data = randn(20, 5)

    X, K = SSVM.create_X_and_K(nothing, data, false)
    @test X == data
    @test (K isa Nothing) == true

    X, K = SSVM.create_X_and_K(nothing, data, true)
    @test @views(X[:,1:end-1]) == data
    @test all(isequal(1), @views(X[:,end]))
    @test (K isa Nothing) == true

    X, K = SSVM.create_X_and_K(RBFKernel(), data, false)
    @test X == data
    @test K ≈ kernelmatrix(RBFKernel(), data, obsdim=1)

    X, K = SSVM.create_X_and_K(RBFKernel(), data, true)
    @test X == data
    @test @views(K[:,1:end-1]) ≈ kernelmatrix(RBFKernel(), data, obsdim=1)
    @test all(isequal(1), @views(K[:,end]))
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
            @test SSVM.poslabel(prob) == "a"
            @test SSVM.neglabel(prob) == "b"
            @test SSVM.__classify__(prob,  1) == "a"
            @test SSVM.__classify__(prob, -1) == "b"
        end

        @testset "labels w/ specified negative_label" begin
            prob = linear_classifier("a", "b")
            @test SSVM.poslabel(prob) == "a"
            @test SSVM.neglabel(prob) == "b"
            @test SSVM.__classify__(prob,  1) == "a"
            @test SSVM.__classify__(prob, -1) == "b"

            prob = linear_classifier("a", "something_else")
            @test SSVM.poslabel(prob) == "a"
            @test SSVM.neglabel(prob) == "something_else"
            @test SSVM.__classify__(prob,  1) == "a"
            @test SSVM.__classify__(prob, -1) == "something_else"
        end

        @testset "switch positive label" begin
            prob = linear_classifier("b", nothing)
            @test SSVM.poslabel(prob) == "b"
            @test SSVM.neglabel(prob) == "a"
            @test SSVM.__classify__(prob,  1) == "b"
            @test SSVM.__classify__(prob, -1) == "a"
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
            @test SSVM.poslabel(prob) == "a"
            @test SSVM.neglabel(prob) == "not_a"
            @test SSVM.__classify__(prob,  1) == "a"
            @test SSVM.__classify__(prob, -1) == "not_a"
        end

        @testset "labels w/ specified negative_label" begin
            prob = linear_classifier("a", "!")
            @test SSVM.poslabel(prob) == "a"
            @test SSVM.neglabel(prob) == "!"
            @test SSVM.__classify__(prob,  1) == "a"
            @test SSVM.__classify__(prob, -1) == "!"
        end

        @testset "switch positive label" begin
            prob = linear_classifier("b", nothing)
            @test SSVM.poslabel(prob) == "b"
            @test SSVM.neglabel(prob) == "not_b"
            @test SSVM.__classify__(prob,  1) == "b"
            @test SSVM.__classify__(prob, -1) == "not_b"
        end
    end

    @testset "linear case" begin
        labeled_data = (rand(("a", "b"), n), randn(n, p)) # tuple of labels and data
        function test_classifier(intercept)
            prob = BinarySVMProblem(labeled_data..., "a", intercept=intercept, kernel=nothing)
            T = SSVM.floattype(prob)
            X = SSVM.get_design_matrix(prob)

            if intercept
                @test labeled_data[2] == @views(X[:,1:end-1])
                @test all(isequal(1), @views(X[:,end]))
            else
                @test labeled_data[2] == X
            end

            @test (prob.K isa Nothing) == true
            @test sort!(unique(prob.y)) == T[-1, 1]
            @test SSVM.probdims(prob) == (n, p, 2)
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
            T = SSVM.floattype(prob)
            K = SSVM.get_design_matrix(prob)

            if intercept
                @test @views(K[:,1:end-1]) ≈ kernelmatrix(RBFKernel(), labeled_data[2], obsdim=1)
                @test all(isequal(1), @views(K[:,end]))
            else
                @test K ≈ kernelmatrix(RBFKernel(), labeled_data[2], obsdim=1)
            end

            @test prob.X == labeled_data[2]
            @test sort!(unique(prob.y)) == T[-1, 1]
            @test SSVM.probdims(prob) == (n, p, 2)
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
        @test SSVM.probdims(prob) == (n, p, c)
        @test length(prob.svm) == binomial(c, 2)
    end

    @testset "One versus Rest (OVR)" begin
        prob = classifier(false, nothing, OVR())
        @test SSVM.probdims(prob) == (n, p, c)
        @test length(prob.svm) == c
    end
end
# @testset "Projection" begin
#     n = 100
#     x = zeros(n)
#     idx = collect(1:n)
#     idx_buffer = similar(idx)
    
#     # case: more nonzeros than k
#     k = 10
#     for i in eachindex(x)
#         x[i] = rand() > 0.5 ? -i : i
#     end
#     shuffle!(x)
#     perm = sortperm(x, rev=true, by=abs)
#     xproj = project_sparsity_set!(copy(x), idx, k, idx_buffer)
#     @test all(i -> xproj[perm[i]] == x[perm[i]], 1:k)
#     @test all(i -> xproj[perm[i]] == 0, k+1:n)
    
#     # case: less nonzeros than k; no change
#     c = round(Int, 0.5*k)
#     fill!(x, 0)
#     for i in 1:c
#         x[i] = rand() > 0.5 ? -i : i
#     end
#     shuffle!(x)
#     perm = sortperm(x, rev=true, by=abs)
#     xproj = project_sparsity_set!(copy(x), idx, k, idx_buffer)
#     @test all(i -> xproj[i] == x[i], 1:n)
    
#     # case: exactly k nonzeros
#     fill!(x, 0)
#     for i in 1:k
#         x[i] = rand() > 0.5 ? -i : i
#     end
#     shuffle!(x)
#     perm = sortperm(x, rev=true, by=abs)
#     xproj = project_sparsity_set!(copy(x), idx, k, idx_buffer)
#     @test all(i -> xproj[i] == x[i], 1:n)
    
#     # helper function should give length n vector without intercept
#     p = randn(n)
#     pvec, idx = SparseSVM.get_model_coefficients(p, false)
#     @test length(pvec) == n
#     @test all(i -> p[i] == pvec[i], 1:length(pvec))
    
#     # helper function should give view into p with intercept
#     p = randn(n)
#     pvec, idx = SparseSVM.get_model_coefficients(p, true)
#     @test length(pvec) == n-1
#     @test all(i -> p[i] == pvec[i], 1:length(pvec))
# end
