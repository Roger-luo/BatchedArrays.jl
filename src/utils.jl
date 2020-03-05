export flatten_m, @flatten

function flatten_m(__source__::LineNumberNode, x::Symbol, (d1, d2)::Tuple{Int, Int})
    body = Expr(:block)
    for j in 1:d2, i in 1:d1
        y = Symbol(x, i, j)
        push!(body.args, __source__)
        push!(body.args, :($y = $x[$i, $j, k]))
    end
    return body
end

"""
    @flatten <Symbol> d1 d2

Flatten matrix variable given its size. If only one dimension is given,
it will treat it as a square matrix.

# Example

```jldoctest
ulia> A = rand(2, 2)
2×2 Array{Float64,2}:
 0.236253  0.57472  
 0.653503  0.0972102

julia> @flatten A 2 2
0.09721022673674717

julia> A11
0.2362527850765206
```
"""
macro flatten(x::Symbol, d1::Int, d2::Int)
    return esc(flatten_m(__source__, x, (d1, d2)))
end

macro flatten(x::Symbol, d::Int)
    return esc(flatten_m(__source__, x, (d, d)))
end
