"""
Normalization (unit norm transformation)
"""
struct NormalizationTransform{T<:Real} <: StatsBase.AbstractDataTransform
    len::Int
    dims::Int
    norms::Vector{T}

    function NormalizationTransform(l::Int, dims::Int, n::Vector{T}) where T
        len = length(n)
        len == l || len == 0 || throw(DimensionMismatch("Inconsistent dimensions."))
        new{T}(l, dims, n)
    end
end

function Base.getproperty(t::NormalizationTransform, p::Symbol)
    if p === :indim || p === :outdim
        return t.len
    else
        return getfield(t, p)
    end
end

"""
    fit(NormalizationTransform, X; dims=nothing, center=true, scale=true)
Fit standardization parameters to vector or matrix `X`
and return a `NormalizationTransform` transformation object.
# Keyword arguments
* `dims`: if `1` fit normalization parameters in column-wise fashion;
  if `2` fit in row-wise fashion. The default is `nothing`, which is equivalent to `dims=2` with a deprecation warning.
# Examples
```jldoctest
julia> using StatsBase
julia> X = [0.0 -0.5 0.5; 0.0 1.0 2.0]
2×3 Matrix{Float64}:
 0.0  -0.5  0.5
 0.0   1.0  2.0
julia> dt = fit(NormalizationTransform, X, dims=2)
NormalizationTransform{Float64}(2, 2, [0.7071067811865476, 2.23606797749979])
julia> StatsBase.transform(dt, X)
2×3 Matrix{Float64}:
 0.0  -0.707107  0.707107
 0.0   0.447214  0.894427
```
"""
function StatsBase.fit(::Type{NormalizationTransform}, X::AbstractMatrix{<:Real};
        dims::Union{Integer,Nothing}=nothing)
    if dims === nothing
        Base.depwarn("fit(t, x) is deprecated: use fit(t, x, dims=2) instead", :fit)
        dims = 2
    end
    if dims == 1
        n, l = size(X)
        n >= 1 || error("X must contain at least one row.")
        norms = [norm(xi) for xi in eachcol(X)]
    elseif dims == 2
        l, n = size(X)
        n >= 1 || error("X must contain at least one column.")
        norms = [norm(xi) for xi in eachrow(X)]
    else
        throw(DomainError(dims, "fit only accept dims to be 1 or 2."))
    end
    T = eltype(X)
    return NormalizationTransform(l, dims, vec(norms))
end

function StatsBase.fit(::Type{NormalizationTransform}, X::AbstractVector{<:Real};
        dims::Integer=1)
    if dims != 1
        throw(DomainError(dims, "fit only accepts dims=1 over a vector. Try fit(t, x, dims=1)."))
    end

    T = eltype(X)
    norms = [norm(X)]
    return NormalizationTransform(1, dims, norms)
end

function StatsBase.transform!(y::AbstractMatrix{<:Real}, t::NormalizationTransform, x::AbstractMatrix{<:Real})
    if t.dims == 1
        l = t.len
        size(x,2) == size(y,2) == l || throw(DimensionMismatch("Inconsistent dimensions."))
        n = size(y,1)
        size(x,1) == n || throw(DimensionMismatch("Inconsistent dimensions."))

        norms = t.norms

        if isempty(norms)
            copyto!(y, x)
        else
            broadcast!(/, y, x, norms')
        end
    elseif t.dims == 2
        t_ = NormalizationTransform(t.len, 1, t.norms)
        transform!(y', t_, x')
    end
    return y
end

function reconstruct!(x::AbstractMatrix{<:Real}, t::NormalizationTransform, y::AbstractMatrix{<:Real})
    if t.dims == 1
        l = t.len
        size(x,2) == size(y,2) == l || throw(DimensionMismatch("Inconsistent dimensions."))
        n = size(y,1)
        size(x,1) == n || throw(DimensionMismatch("Inconsistent dimensions."))

        norms = t.norms

        if isempty(norms)
            copyto!(x, y)
        else
            broadcast!(*, x, y, norms')
        end
    elseif t.dims == 2
        t_ = NormalizationTransform(t.len, 1, t.norms)
        reconstruct!(x', t_, y')
    end
    return x
end
