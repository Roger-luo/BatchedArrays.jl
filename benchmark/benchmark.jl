using BatchedArrays
using Plots
using BenchmarkTools

function benchmark(sz, nbatch)
    times = []
    for N in sz
        A, B, C = (rand(N, N, nbatch) for _ in 1:3)
        t = @benchmark bmm!($C, $A, $B)
        push!(times, time(minimum(t)))
    end
    return times
end
