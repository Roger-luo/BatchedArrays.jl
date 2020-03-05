using Test
using BatchedArrays

function naive_bmm!(C::Array{T, 3}, A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    for k in 1:size(C, 3)
        C[:, :, k] = A[:, :, k] * B[:, :, k]
    end
    return C
end

function naive_bmm(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where T
    C = similar(B, T, (size(A,1), size(B,2), size(A, 3)))
    naive_bmm!(C, A, B)
end

function naive_addbmm!(C, A, B)
    for k in 1:size(C, 3)
        C[:, :, k] += A[:, :, k] * B[:, :, k]
    end
    return C
end

gen_list(N) = [rand(N, N, 100), BTranspose(rand(N, N, 100)), BScale(rand(100), N)]

@testset "bmm, size $N" for N in (2, 3, 10)
    @testset "bmm($(nameof(typeof(A))), $(nameof(typeof(B))))" for A in gen_list(N), B in gen_list(N)
        @test bmm(A, B) ≈ naive_bmm(A, B)
    end
end

@testset "bmm, size $N" for N in (2, 3, 10)
    @testset "bmm($(nameof(typeof(A))), $(nameof(typeof(B))))" for A in gen_list(N), B in gen_list(N)        
        C = rand(N, N, 100)
        @test addbmm!(copy(C), A, B) ≈ naive_addbmm!(copy(C), A, B)
    end
end
