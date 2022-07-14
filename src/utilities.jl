#
# Helper function to write interface for accessors. These will depend on the format used.
#
#   - Val{:last}  ==> [w, b] format; b is in the last index.
#   - Val{:first} ==> [b, w] format; b is in teh first index.
#
#   - Vectors: β = [w, b] or [b, w], where w represents slopes/weights/coefficients and b is the intercept
#   - Matrices: B = [β₁ β₂ ... βₖ], where column βᵢ contains parameters for SVM i.
#   - VectorOfVectors: B = [β₁, β₂, ... βₖ]. B is a ragged array in which subset βᵢ contains parameters for SVM i.
#

__get_intercept_index__(arr) = __get_intercept_index__(INTERCEPT_INDEX, arr)

# [w, b] format
__get_intercept_index__(::Val{:last}, arr::AbstractVector) = lastindex(arr)
__get_intercept_index__(::Val{:last}, arr::AbstractMatrix) = lastindex(arr, 1)

# [b, w] format
__get_intercept_index__(::Val{:first}, arr::AbstractVector) = firstindex(arr)
__get_intercept_index__(::Val{:first}, arr::AbstractMatrix) = firstindex(arr, 1)

function __set_intercept_component__!(arr, val)
    index = __get_intercept_index__(arr)
    arr[index] = val
end

__coeff_range__(nparams) = __coeff_range__(INTERCEPT_INDEX, nparams)

# [w, b] format
__coeff_range__(::Val{:last}, nparams) = Base.OneTo(nparams)

# [b, w] format
__coeff_range__(::Val{:first}, nparams) = 2:nparams

function __slope_and_coeff_views__(arr::AbstractVector, intercept)
    len = length(arr)

    if intercept
        int_idx = __get_intercept_index__(arr)
        coeff_idx = __coeff_range__(len-1)
        b, w = arr[int_idx], view(arr, coeff_idx)
    else
        b, w = zero(eltype(arr)), view(arr, Base.OneTo(len))
    end

    return b, w
end

function __slope_and_coeff_views__(arr::Matrix, intercept)
    nrow, ncol = size(arr)

    if intercept
        int_idx = __get_intercept_index__(arr)
        coeff_idx = __coeff_range__(nrow-1)
        b, w = view(arr, int_idx, :), view(arr, coeff_idx, :)
    else
        b, w = zeros(eltype(arr), ncol), view(arr, Base.OneTo(nrow), :)
    end

    return b, w
end

function __slope_and_coeff_views__(arr::VectorOfVectors, intercept)
    len = length(arr)
    b_1, w_1 = __slope_and_coeff_views__(arr[1], intercept)
    b, w = [b_1], [w_1]
    for k in 2:len
        b_k, w_k = __slope_and_coeff_views__(arr[k], intercept)
        push!(b, b_k)
        push!(w, w_k)
    end
    return b, w
end

#
#   Helper functions for evaluating residuals and computing gradients.
#
function __predicted_response__!(r, X, b, w)
    if iszero(b)
        mul!(r, X, w)
    else
        T = eltype(r)
        fill!(r, b)
        mul!(r, X, w, one(T), one(T))
    end
    return nothing
end

function __shifted_response__!(z, y, f)
    @inbounds for i in eachindex(z)
        yi, fi = y[i], f[i]
        z[i] = ifelse(yi*fi < 1, yi, fi)
    end
    return nothing
end

"""
Generic template for evaluating residuals.
This assumes that projections have been handled externally.
The following flags control how residuals are evaluated:

+ `need_main`: If `true`, evaluates regression residuals.
+ `need_dist`: If `true`, evaluates difference between parameters and their projection.

**Note**: The values for each flag should be known at compile-time!
"""
function __evaluate_residuals__(problem::BinarySVMProblem, extras, need_main::Bool, need_dist::Bool)
    @unpack n, y, coeff, proj, res = problem
    @unpack z = extras

    β, p, r, q = coeff, proj, res.main, res.dist
    b, w = get_params(problem)          # intercept + coefficients
    A = get_design_matrix(problem)      # X or KY
    T = SparseSVM.floattype(problem)
    
    if need_main # 1/sqrt(n) * (zₘ - A*β)
        c1 = convert(T, 1 / sqrt(n))
        __predicted_response__!(r, A, b, w)
        __shifted_response__!(z, y, r)
        axpby!(c1, z, -c1, r)
    end

    if need_dist # P(βₘ) - β
        @. q = p - β
    end

    return nothing
end

"""
Evaluate the gradiant of the regression problem. Assumes residuals have been evaluated.
"""
function __evaluate_gradient__(problem, lambda, rho, extras)
    @unpack n, y, grad, res, kernel, intercept = problem

    ∇g, r, q = grad, res.main, res.dist
    _, w = get_params(problem) 
    A = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)

    # ∇gᵨ(β ∣ βₘ) = -c1*[1ᵀr; Aᵀr] - c2*qₘ + c3*[0,w]
    c1, c2, c3 = convert(T, 1 / sqrt(n)), convert(T, rho), convert(T, lambda)
    _, ∇g_w = __slope_and_coeff_views__(∇g, intercept)

    if intercept
        __set_intercept_component__!(∇g, sum(r))
    end

    mul!(∇g_w, A', r)
    axpby!(-c2, q, -c1, ∇g)     # = -c1*[1ᵀr; Aᵀr] - c2*qₘ
    axpy!(c3, w, ∇g_w)          # + c3*[0,w]

    return nothing
end

function __evaluate_reg_gradient__(problem, lambda, extras)
    @unpack n, y, grad, res, kernel, intercept = problem

    ∇g, r = grad, res.main
    _, w = get_params(problem) 
    A = get_design_matrix(problem)
    T = SparseSVM.floattype(problem)

    # ∇gᵨ(β ∣ βₘ) = -c1*[1ᵀr; Aᵀr] + c2*[0,w]
    c1, c3 = convert(T, 1 / sqrt(n)), convert(T, lambda)
    _, ∇g_w = __slope_and_coeff_views__(∇g, intercept)

    if intercept
        __set_intercept_component__!(∇g, -sum(r) * c1)
    end

    mul!(∇g_w, A', r)           # = [1ᵀr; Aᵀr]
    axpby!(c3, w, -c1, ∇g_w)    # = -c1*[1ᵀr; Aᵀr] + c2*[0,w]

    return nothing
end

"""
Evaluate the penalized least squares criterion. Also updates the gradient.
This assumes that projections have been handled externally.
"""
function __evaluate_objective__(problem, lambda, rho, extras)
    @unpack res, grad, intercept = problem
    r, q, ∇g = res.main, res.dist, grad
    _, w = get_params(problem)

    __evaluate_residuals__(problem, extras, true, true)
    __evaluate_gradient__(problem, lambda, rho, extras)

    risk = dot(r, r)                        # R = 1/n * |zₘ - X*βₘ|²
    wnorm2 = dot(w, w)
    loss = 1//2 * (risk + lambda * wnorm2)  # L = R + λ|wₘ|²
    distsq = dot(q, q)                      # ∑ⱼ (P(βₘ) - βₘ)²
    objv = loss + 1//2 * rho * distsq
    gradsq = dot(∇g, ∇g)

    return (; risk=risk, loss=loss, objective=objv, wnorm=sqrt(wnorm2), gradient=sqrt(gradsq), distance=sqrt(distsq))
end

function __evaluate_reg_objective__(problem, lambda, extras)
    @unpack coeff, res, grad, intercept = problem
    r, ∇g = res.main, grad
    _, w = get_params(problem)

    __evaluate_residuals__(problem, extras, true, false)
    __evaluate_reg_gradient__(problem, lambda, extras)

    risk = dot(r, r)                        # R = 1/n * |zₘ - X*βₘ|²
    wnorm2 = dot(w, w)
    loss = 1//2 * (risk + lambda * wnorm2)  # L = R + λ|wₘ|²
    objv = loss
    gradsq = dot(∇g, ∇g)

    return (; risk=risk, loss=loss, objective=objv, wnorm=sqrt(wnorm2), gradient=sqrt(gradsq), distance=zero(gradsq))
end

"""
Apply acceleration to the current iterate `x` based on the previous iterate `y`
according to Nesterov's method with parameter `r=3` (default).
"""
function __apply_nesterov__!(x, y, iter::Integer, needs_reset::Bool, r::Int=3)
    if needs_reset # Reset acceleration scheme
        copyto!(y, x)
        iter = 1
    else # Nesterov acceleration 
        γ = (iter - 1) / (iter + r - 1)
        @inbounds for i in eachindex(x)
            xi, yi = x[i], y[i]
            zi = xi + γ * (xi - yi)
            x[i], y[i] = zi, xi
        end
        iter += 1
    end

    return iter
end

"""
Map a sparsity level `s` to an integer `k`, assuming `n` elements.
"""
sparsity_to_k(problem::AbstractSVM, s) = __sparsity_to_k__(problem.kernel, problem, s)
__sparsity_to_k__(::Nothing, problem::AbstractSVM, s) = round(Int, (1-s) * problem.p)
__sparsity_to_k__(::Kernel, problem::AbstractSVM, s) = round(Int, (1-s) * problem.n)

"""
Apply a projection to model coefficients.
"""
function apply_projection(projection, problem, k)
    @unpack coeff, proj = problem
    copyto!(proj, coeff)
    _, p = get_params_proj(problem)
    projection(p, k)

    return proj
end

struct GeometricProression <: Function
    multiplier::Float64
end

function (f::GeometricProression)(rho, iter, rho_max)
    convert(typeof(rho), min(rho_max, rho * f.multiplier))
end
"""
Define a geometric progression recursively by the rule
```
    rho_new = min(rho_max, rho * multiplier)
```
The result is guaranteed to have type `typeof(rho)`.
"""
function geometric_progression(multiplier::Real=1.2)
    return GeometricProression(multiplier)
end

convert_labels(model::BinarySVMProblem, L) = map(Li -> MLDataUtils.convertlabel(model.ovr_encoding, Li, model.ovr_encoding), L)
convert_labels(model::MultiSVMProblem, L) = L

function prediction_errors(model, train_set, validation_set, test_set)
    # Extract data for each set.
    Tr_Y, Tr_X = train_set
    V_Y, V_X = validation_set
    T_Y, T_X = test_set

    # Helper function to make predictions on each subset and evaluate errors.
    classification_error = function(model, L, X)
        # Sanity check: Y and X have the same number of rows.
        length(L) != size(X, 1) && error("Labels ($(length(L))) not compatible with data X ($(size(X))).")

        # Translate data to the model's encoding.
        L_translated = convert_labels(model, L)

        # Classify response in vertex space; may use @batch.
        Lhat = SparseSVM.classify(model, X)

        # Sweep through predictions and tally the mistakes.
        nincorrect = sum(L_translated .!= Lhat)

        return 100 * nincorrect / length(L)
    end

    Tr = classification_error(model, Tr_Y, Tr_X)
    V = classification_error(model, V_Y, V_X)
    T = classification_error(model, T_Y, T_X)

    return (Tr, V, T)
end

function set_initial_coefficients!(problem::BinarySVMProblem, v::Real)
    _, w = get_params_prev(problem)
    fill!(w, v)
end

function set_initial_coefficients!(problem::MultiSVMProblem, v::Real)
    foreach(svm -> set_initial_coefficients!(svm, v), problem.svm)
end

function set_initial_coefficients_and_intercept!(problem::BinarySVMProblem, v)
    fill!(problem.coeff_prev, v)
end

function set_initial_coefficients_and_intercept!(problem::MultiSVMProblem, v::Real)
    foreach(svm -> set_initial_coefficients_and_intercept!(svm, v), problem.svm)
end

"""
Placeholder for callbacks in main functions.
"""
__do_nothing_callback__(iter, problem, hyperparams, state) = nothing
# __do_nothing_callback__(fold, problem, train_problem, data, lambda, sparsity, model_size, result) = nothing

function verbose_cb(iter, problem, hyperparams, state, every=1)
    if iter == 0
        @printf("\n%-5s\t%-8s\t%-8s\t%-8s\t%-8s\t%-12s\t%-8s\t%-8s\n", "iter", "rho", "risk", "loss", "objective", "1/|w|", "|gradient|", "distance")
    end
    if iter % every == 0
        @printf("%4d\t%4.3e\t%4.3e\t%4.3e\t%4.3e\t%8.3e\t%4.3e\t%4.3e\n", iter, hyperparams.rho, state.risk, state.loss, state.objective, 1/(state.wnorm), state.gradient, state.distance)
    end

    return nothing
end

__svd_wrapper__(A::StridedMatrix) = svd(A, full=false)
__svd_wrapper__(A::AbstractMatrix) = svd!(copy(A), full=false)

function __adjust_transform__(F::ZScoreTransform)
    has_nan = any(isnan, F.scale) || any(isnan, F.mean)
    has_inf = any(isinf, F.scale) || any(isinf, F.mean)
    has_zero = any(iszero, F.scale)
    if has_nan
        error("Detected NaN in z-score.")
    elseif has_inf
        error("Detected Inf in z-score.")
    elseif has_zero
        for idx in eachindex(F.scale)
            x = F.scale[idx]
            F.scale[idx] = ifelse(iszero(x), one(x), x)
        end
    end
    return F
end

function __adjust_transform__(F::NormalizationTransform)
    has_nan = any(isnan, F.norms)
    has_inf = any(isinf, F.norms)
    has_zero = any(iszero, F.norms)
    if has_nan
        error("Detected NaN in norms.")
    elseif has_inf
        error("Detected Inf in norms.")
    elseif has_zero
        for idx in eachindex(F.norms)
            x = F.norms[idx]
            F.norms[idx] = ifelse(iszero(x), one(x), x)
        end
    end
    return F
end

__adjust_transform__(F::NoTransformation) = F
