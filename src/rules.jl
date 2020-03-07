using ChainRulesCore

function ChainRulesCore.rrule(::typeof(bmm), A::AbstractArray{<:Real, 3}, B::AbstractArray{<:Real, 3})
    function bmm_pullback(Δ)
        ∇A = @thunk bmm(Δ, BTranspose(B))
        ∇B = @thunk bmm(BTranspose(A), Δ)
        return (NO_FIELDS, ∇A, ∇B)
    end

    return bmm(A, B), bmm_pullback
end

function ChainRulesCore.rrule(::typeof(batched_tr), A::AbstractArray{T, 3}) where T
    function batched_tr_pullback(Δ)
        return (NO_FIELDS, @thunk(BScale(Δ, size(A, 1))))
    end
    return batched_tr(A), batched_tr_pullback
end

using ZygoteRules: @adjoint

@adjoint function bmm(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    function bmm_pullback(Δ)
        ∇A = bmm(Δ, BTranspose(B))
        ∇B = bmm(BTranspose(A), Δ)
        return ∇A, ∇B
    end
    return bmm(A, B), bmm_pullback
end

@adjoint function batched_tr(A::AbstractArray{T, 3}) where T
    function batched_tr_pullback(Δ)
        return BScale(Δ, size(A, 1))
    end
    return batched_tr(A), batched_tr_pullback
end
