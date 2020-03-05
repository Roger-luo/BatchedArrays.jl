using Base.Cartesian: @nexprs
export bmm, bmm!, addbmm!, bmm2x2!, bmm3x3!,
    batched_tr, batched_tr!, BTranspose, BScale

struct BTranspose{T, S <: AbstractArray{T, 3}} <: AbstractArray{T, 3}
    parent::S
end

Base.parent(x::BTranspose) = x.parent
Base.size(x::BTranspose) = size(x.parent)
Base.getindex(x::BTranspose, i::Int, j::Int, k::Int) = getindex(x.parent, j, i, k)
Base.setindex!(x::BTranspose, v, i::Int, j::Int, k::Int) = setindex!(x.parent, v, j, i, k)

struct BScale{T, S <: AbstractVector{T}} <: AbstractArray{T, 3}
    diags::S
    m::Int
end

Base.size(x::BScale{T}) where T = (x.m, x.m, length(x.diags))
Base.getindex(x::BScale{T}, i::Int, j::Int, k::Int) where T = i == j ? x.diags[k] : zero(T)

function bmm end

function bmm(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    C = similar(B, T, (size(A,1), size(B,2), size(A, 3)))
    return bmm!(C, A, B)
end

"""
    bmm!(C, A, B)

multiply `A` and `B` and stores the result into `C`.
"""
function bmm! end

# For ... in [AbstractArray{T, 3}, BTranspose, BScale]
#   For ... in [AbstractArray{T, 3}, BTranspose, BScale]

# AbstractArray{T, 3}
function bmm!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    mA, nA, bA = size(A)
    mB, nB, bB = size(B)

    if nA != mB || bA != bB
        throw(DimensionMismatch("A has dimensions ($mA,$nA,$bB) but B has dimensions ($mB,$nB,$bB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 2 && nA == 2 && nB == 2
        return bmm2x2!(C, A, B)
    end

    if mA == 3 && nA == 3 && nB == 3
        return bmm3x3!(C, A, B)
    end

    return bmm_fallback!(C, A, B)
end

"""shape irralvent fallback methods"""
function bmm_fallback! end

const _BLAS_TYPE_TABLE_ = [('N', :(AbstractArray{T, 3})), ('T', :(BTranspose{T}))]
for (transA, AType) in _BLAS_TYPE_TABLE_
    for (transB, BType) in _BLAS_TYPE_TABLE_
        @eval function bmm_fallback!(C::Array{T, 3}, A::$AType, B::$BType) where T
            bmm_wrapper!(C, $transA, $transB, parent(A), parent(B))
            return C
        end        
    end
end

function bmm_wrapper!(C::AbstractArray{T, 3}, transA::AbstractChar, transB::AbstractChar, A::AbstractArray{T, 3},
    B::AbstractArray{T, 3}) where T
    if stride(A, 1) == stride(B, 1) == stride(C, 1) == 1 && stride(A, 2) >= size(A, 1) && stride(B, 2) >= size(B, 1) && stride(C, 2) >= size(C, 1)
        batched_gemm!(transA, transB, one(T), A, B, zero(T), C)
    else
        error("non-strided Array bmm is not implemented")
    end
end

# specialize on BScale
function bmm_fallback!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3), j in 1:size(C, 2), i in 1:size(C, 1)
        C[i, j, k] = A[i, j, k] * B.diags[k]
    end
    return C
end

function bmm_fallback!(C::Array{T, 3}, A::BTranspose{T}, B::BScale{T}) where T
    invoke(bmm_fallback!, Tuple{Array{T, 3}, AbstractArray{T, 3}, BScale{T}}, C, A, B)
end

function bmm_fallback!(C::Array{T, 3}, A::BScale{T}, B::BTranspose{T}) where T
    bmm_fallback!(C, B, A)
end

function bmm_fallback!(C::Array{T, 3}, A::BScale{T}, B::AbstractArray{T, 3}) where T
    return bmm_fallback!(C, B, A)
end

function bmm_fallback!(C::Array{T, 3}, A::BScale{T}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3), j in 1:size(C, 2), i in 1:size(C, 1)
        if i == j
            C[i, j, k] = A.diags[k] * B.diags[k]
        else
            C[i, j, k] = zero(T)
        end
    end
    return C
end

#######
function bmm2x2! end
function bmm3x3! end
# For ... in [AbstractArray{T, 3}, BTranspose, BScale]
#   For ... in [AbstractArray{T, 3}, BTranspose, BScale]

# AbstractArray{T, 3}
function bmm2x2!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @inbounds for k in 1:size(C, 3)
        A11 = A[1,1,k]; A12 = A[1,2,k];
        A21 = A[2,1,k]; A22 = A[2,2,k];
        B11 = B[1,1,k]; B12 = B[1,2,k];
        B21 = B[2,1,k]; B22 = B[2,2,k];
        
        C[1,1,k] = A11*B11 + A12*B21
        C[1,2,k] = A11*B12 + A12*B22
        C[2,1,k] = A21*B11 + A22*B21
        C[2,2,k] = A21*B12 + A22*B22
    end
    return C
end

function bmm3x3!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @inbounds for k in 1:size(C, 3)
        A11 = A[1,1,k]; A12 = A[1,2,k]; A13 = A[1,3,k];
        A21 = A[2,1,k]; A22 = A[2,2,k]; A23 = A[2,3,k];
        A31 = A[3,1,k]; A32 = A[3,2,k]; A33 = A[3,3,k];

        B11 = B[1,1,k]; B12 = B[1,2,k]; B13 = B[1,3,k];
        B21 = B[2,1,k]; B22 = B[2,2,k]; B23 = B[2,3,k];
        B31 = B[3,1,k]; B32 = B[3,2,k]; B33 = B[3,3,k];
        
        C[1,1,k] = A11*B11 + A12*B21 + A13*B31
        C[1,2,k] = A11*B12 + A12*B22 + A13*B32
        C[1,3,k] = A11*B13 + A12*B23 + A13*B33

        C[2,1,k] = A21*B11 + A22*B21 + A23*B31
        C[2,2,k] = A21*B12 + A22*B22 + A23*B32
        C[2,3,k] = A21*B13 + A22*B23 + A23*B33

        C[3,1,k] = A31*B11 + A32*B21 + A33*B31
        C[3,2,k] = A31*B12 + A32*B22 + A33*B32
        C[3,3,k] = A31*B13 + A32*B23 + A33*B33
    end
    return C
end

function bmm2x2!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3)
        A11 = A[1,1,k]; A12 = A[1,2,k];
        A21 = A[2,1,k]; A22 = A[2,2,k];

        C[1, 1, k] = A11 * B.diags[k]
        C[2, 1, k] = A21 * B.diags[k]
        C[1, 2, k] = A12 * B.diags[k]
        C[2, 2, k] = A22 * B.diags[k]
    end
    return C
end

# BScale
function bmm2x2!(C::Array{T, 3}, A::BScale{T}, B::AbstractArray{T, 3}) where T
    return bmm2x2!(C, B, A)
end

function bmm2x2!(C::Array{T, 3}, A::BScale{T}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3)
        C[1, 1, k] = A.diags[k] * B.diags[k]
        C[2, 1, k] = zero(T)
        C[1, 2, k] = zero(T)
        C[2, 2, k] = A.diags[k] * B.diags[k]
    end
    return C
end

#######
"""
    addbmm!(C, A, B)

Multiply batched matrix `A`, `B` and accumulate the result to `C`.
"""
function addbmm! end
# For ... in [AbstractArray{T, 3}, BTranspose, BScale]
#   For ... in [AbstractArray{T, 3}, BTranspose, BScale]

# AbstractArray{T, 3}
function addbmm!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    mA, nA, bA = size(A)
    mB, nB, bB = size(B)

    if nA != mB || bA != bB
        throw(DimensionMismatch("A has dimensions ($mA,$nA,$bB) but B has dimensions ($mB,$nB,$bB)"))
    end

    if C === A || B === C
        throw(ArgumentError("output matrix must not be aliased with input matrix"))
    end

    if mA == 2 && nA == 2 && nB == 2
        return addbmm2x2!(C, A, B)
    end

    if mA == 3 && nA == 3 && nB == 3
        return addbmm3x3!(C, A, B)
    end

    return addbmm_fallback!(C, A, B)
end

"""shape irralvent fallback methods"""
function addbmm_fallback! end

for (transA, AType) in _BLAS_TYPE_TABLE_
    for (transB, BType) in _BLAS_TYPE_TABLE_
        @eval function addbmm_fallback!(C::Array{T, 3}, A::$AType, B::$BType) where T
            addbmm_wrapper!(C, $transA, $transB, parent(A), parent(B))
            return C
        end        
    end
end

function addbmm_wrapper!(C::AbstractArray{T, 3}, transA::AbstractChar, transB::AbstractChar, A::AbstractArray{T, 3},
    B::AbstractArray{T, 3}) where T
    if stride(A, 1) == stride(B, 1) == stride(C, 1) == 1 && stride(A, 2) >= size(A, 1) && stride(B, 2) >= size(B, 1) && stride(C, 2) >= size(C, 1)
        batched_gemm!(transA, transB, one(T), A, B, one(T), C)
    else
        error("non-strided Array bmm is not implemented")
    end
end

# specialize on BScale
function addbmm_fallback!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3), j in 1:size(C, 2), i in 1:size(C, 1)
        C[i, j, k] += A[i, j, k] * B.diags[k]
    end
    return C
end

function addbmm_fallback!(C::Array{T, 3}, A::BScale{T}, B::AbstractArray{T, 3}) where T
    return addbmm_fallback!(C, B, A)
end

function addbmm_fallback!(C::Array{T, 3}, A::BScale{T}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3), j in 1:size(C, 2), i in 1:size(C, 1)
        if i == j
            C[i, j, k] += A.diags[k] * B.diags[k]
        end
    end
    return C
end

function addbmm_fallback!(C::Array{T, 3}, A::BTranspose{T}, B::BScale{T}) where T
    return invoke(addbmm_fallback!, Tuple{Array{T, 3}, AbstractArray{T, 3}, BScale{T}}, C, A, B)
end

function addbmm_fallback!(C::Array{T, 3}, A::BScale{T}, B::BTranspose{T}) where T
    return addbmm_fallback!(C, B, A)
end


#######
function addbmm2x2! end
function addbmm3x3! end
# For ... in [AbstractArray{T, 3}, BTranspose, BScale]
#   For ... in [AbstractArray{T, 3}, BTranspose, BScale]

# AbstractArray{T, 3}
function addbmm2x2!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @inbounds for k in 1:size(C, 3)
        A11 = A[1,1,k]; A12 = A[1,2,k];
        A21 = A[2,1,k]; A22 = A[2,2,k];
        B11 = B[1,1,k]; B12 = B[1,2,k];
        B21 = B[2,1,k]; B22 = B[2,2,k];
        
        C[1,1,k] += A11*B11 + A12*B21
        C[1,2,k] += A11*B12 + A12*B22
        C[2,1,k] += A21*B11 + A22*B21
        C[2,2,k] += A21*B12 + A22*B22
    end
    return C
end

function addbmm3x3!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    @inbounds for k in 1:size(C, 3)
        A11 = A[1,1,k]; A12 = A[1,2,k]; A13 = A[1,3,k];
        A21 = A[2,1,k]; A22 = A[2,2,k]; A23 = A[2,3,k];
        A31 = A[3,1,k]; A32 = A[3,2,k]; A33 = A[3,3,k];

        B11 = B[1,1,k]; B12 = B[1,2,k]; B13 = B[1,3,k];
        B21 = B[2,1,k]; B22 = B[2,2,k]; B23 = B[2,3,k];
        B31 = B[3,1,k]; B32 = B[3,2,k]; B33 = B[3,3,k];
        
        C[1,1,k] += A11*B11 + A12*B21 + A13*B31
        C[1,2,k] += A11*B12 + A12*B22 + A13*B32
        C[1,3,k] += A11*B13 + A12*B23 + A13*B33

        C[2,1,k] += A21*B11 + A22*B21 + A23*B31
        C[2,2,k] += A21*B12 + A22*B22 + A23*B32
        C[2,3,k] += A21*B13 + A22*B23 + A23*B33

        C[3,1,k] += A31*B11 + A32*B21 + A33*B31
        C[3,2,k] += A31*B12 + A32*B22 + A33*B32
        C[3,3,k] += A31*B13 + A32*B23 + A33*B33
    end
    return C
end

function addbmm2x2!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3)
        A11 = A[1,1,k]; A12 = A[1,2,k];
        A21 = A[2,1,k]; A22 = A[2,2,k];

        C[1, 1, k] += A11 * B.diags[k]
        C[2, 1, k] += A21 * B.diags[k]
        C[1, 2, k] += A12 * B.diags[k]
        C[2, 2, k] += A22 * B.diags[k]
    end
    return C
end

# BScale
function addbmm2x2!(C::Array{T, 3}, A::BScale{T}, B::AbstractArray{T, 3}) where T
    return addbmm2x2!(C, B, A)
end

function addbmm2x2!(C::Array{T, 3}, A::BScale{T}, B::BScale{T}) where T
    @inbounds for k in 1:size(C, 3)
        C[1, 1, k] += A.diags[k] * B.diags[k]
        C[2, 2, k] += A.diags[k] * B.diags[k]
    end
    return C
end


"""
    batched_tr(A::AbstractArray{T, 3}) where T

Batched version of trace.
"""
function batched_tr(A::AbstractArray{T, 3}) where T
    @boundscheck size(A, 1) == size(A, 2) || error("Expect a square matrix")
    if size(A, 1) == 2
        batched_tr2x2!(similar(A, (size(A, 3), )), A)
    else
        batched_tr!(fill!(similar(A, (size(A, 3), )), 0), A)
    end
end

function batched_tr2x2!(B::AbstractVector{T}, A::AbstractArray{T, 3}) where T
    @inbounds for k in eachindex(B)
        B[k] = A[1, 1, k] + A[2, 2, k]
    end
    return B
end

function batched_tr!(B::AbstractVector{T}, A::AbstractArray{T, 3}) where T
    # Threads.@threads
    for k in 1:size(A, 3)
        @inbounds for i in 1:size(A, 1)
            B[k] += A[i, i, k]
        end
    end
    return B
end

function batched_tr!(B::AbstractVector{T}, A::BScale{T}) where T
    @inbounds for k in 1:size(A, 3)
        B[k] = A.diags[k] * A.m
    end
    return B
end
