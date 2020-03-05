using Test
using LinearAlgebra
using BatchedArrays
using Revise

@testset "bmm & addbmm" begin
    include("bmm.jl")    
end

function naive_batched_tr!(B::Vector, A)
    for k in 1:length(B)
        B[k] = tr(A[:, :, k])
    end
    return B
end

naive_batched_tr(A) = naive_batched_tr!(similar(A, (size(A, 3), )), A)

@testset "batched_tr($(nameof(typeof(A))))" for A in [rand(2, 2, 100), BTranspose(rand(2, 2, 100)), BScale(rand(100), 2)]
    @test naive_batched_tr(A) ≈ batched_tr(A)
end
