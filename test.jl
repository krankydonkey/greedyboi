function roll_odds(n::Int, results::Vector{Int}, s=6)
    if size(results)[1] == 0
        return 0
    end
    dice_left = circshift(n .- cumsum(results), 1)
    dice_left[1,:] .= n
    nums = prod(factorial.(dice_left) ./ (factorial.(results) .* factorial.(dice_left - results)))
    return sum(nums) * (1 / s)^n
end


function greater_rolls(n::Int, state::Vector{Int}, upper::Vector{Int}, other=Int64[])
    if size(upper)[1] == 0
        if all(y->y==1, other)
            return Int64[]
        else
            return [other]
        end
    else
        rolls = Vector{Int64}[]
        start = max(n - sum(state) - sum(upper[2:end]), 0) + state[1]
        stop = min(n - sum(state), upper[1]-1) + state[1]
        for i in range(start, stop)
            other2 = copy(other)
            push!(other2, i)
            append!(rolls, greater_rolls(n - i, state[2:end], upper[2:end], other2))
        end
        return rolls
    end
end


print(greater_rolls(5, [2, 1, 0, 0, 0, 0], [1, 1, 3, 3, 3, 3]))
print("\n")
