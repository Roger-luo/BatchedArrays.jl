using LinearAlgebra

for (gemm, elty) in
    ((:dgemm_,:Float64),
     (:sgemm_,:Float32),
     (:zgemm_,:ComplexF64),
     (:cgemm_,:ComplexF32))

    @eval begin
        function BLAS.gemm!(
            transA::AbstractChar, transB::AbstractChar,
            m::Int, n::Int, ka::Int,
            alpha::$(elty), ptrA::Ptr{$(elty)}, lda::Int,
            ptrB::Ptr{$(elty)}, ldb::Int,
            beta::$(elty), ptrC::Ptr{$(elty)}, ldc::Int)

            ccall((BLAS.@blasfunc($gemm), BLAS.libblas), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{BLAS.BlasInt}, Ref{BLAS.BlasInt},
                Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BLAS.BlasInt},
                Ptr{$elty}, Ref{BLAS.BlasInt}, Ref{$elty}, Ptr{$elty},
                Ref{BLAS.BlasInt}),
                transA, transB, m, n,
                ka, alpha, ptrA, lda,
                ptrB, ldb, beta, ptrC,
                ldc)

            return ptrC
        end
    end
end

function batched_gemm!(transA::AbstractChar, transB::AbstractChar,
    alpha::T, A::AbstractArray{T, 3}, B::AbstractArray{T, 3},
    beta::T, C::AbstractArray{T, 3}) where T

    @assert size(A, 3) == size(B, 3) == size(C, 3) "batch size mismatch"
    m = size(A, transA == 'N' ? 1 : 2)
    ka = size(A, transA == 'N' ? 2 : 1)
    kb = size(B, transB == 'N' ? 1 : 2)
    n = size(B, transB == 'N' ? 2 : 1)
    if ka != kb || m != size(C,1) || n != size(C,2)
        throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
    end
    BLAS.chkstride1(A)
    BLAS.chkstride1(B)
    BLAS.chkstride1(C)

    ptrA = Base.unsafe_convert(Ptr{T}, A)
    ptrB = Base.unsafe_convert(Ptr{T}, B)
    ptrC = Base.unsafe_convert(Ptr{T}, C)

    # Threads.@threads
    for k in 1:size(A, 3)
        ptrAk = ptrA + (k-1) * stride(A, 3) * sizeof(T)
        ptrBk = ptrB + (k-1) * stride(B, 3) * sizeof(T)
        ptrCk = ptrC + (k-1) * stride(C, 3) * sizeof(T)
        BLAS.gemm!(transA, transB, m, n,
            ka, alpha, ptrAk, max(1, stride(A, 2)),
            ptrBk, max(1, stride(B,2)), beta, ptrCk, max(1, stride(C,2)))
    end

    return C
end
