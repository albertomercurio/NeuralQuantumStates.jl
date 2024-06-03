export vector_to_kron_index, kron_index_to_vector, kron_index_to_vector!
export get_connected_states

# This is to avoid to call prod(@view(M[i+1:end])) when M is already a SubArray
function _prod(i, M::AbstractVector{T}) where {T}
    res = one(T)
    @inbounds for j in i+1:length(M)
        res *= M[j]
    end
    return res
end

function vector_to_kron_index(x::AbstractVector, M::AbstractVector)
    # For some reason this allocates memory when M is already a SubArray
    # q = reduce((acc, i) -> acc + x[i] * prod(@view(M[i+1:end])), eachindex(x), init=1)

    q = reduce((acc, (i, xi)) -> acc + xi * _prod(i, M), enumerate(x), init = 1)
    return q
end

function kron_index_to_vector!(x::AbstractVector, q, M::AbstractVector)
    n = length(M)

    @inbounds for i in n:-1:1
        x[i] = q % M[i]
        q = div(q, M[i])
    end

    return x
end

function kron_index_to_vector(q::T1, M::AbstractVector{T2}) where {T1<:Integer,T2<:Integer}
    T = promote_type(T1, T2)
    n = length(M)
    x = Vector{T}(undef, n)

    return kron_index_to_vector!(x, q, M)
end

function get_connected_states(
    q::QuantumOperator{HT,DT,CT},
    ψ::AbstractVector,
) where {HT,T1,T2,DT<:AbstractDict{<:AbstractVector{T1},<:AbstractMatrix{T2}},CT}
    hi = q.hilbert
    ψ_cache = similar(ψ)

    connected_states_idxs = T1[]
    mels = T2[]

    # Let's add the identity action first
    c = q.constant[]
    if !iszero(c)
        push!(connected_states_idxs, vector_to_kron_index(ψ, hi.dims))
        push!(mels, c)
    end

    for (acting_on, mat) in q.dict
        # This would be more efficient when using SparseMatrixCOO
        rows, cols, vals = findnz(mat)
        idx = vector_to_kron_index(@view(ψ[acting_on]), @view(hi.dims[acting_on]))

        idxs = findall(==(idx), rows)

        if length(idxs) > 0
            for i in idxs
                copyto!(ψ_cache, ψ) # re-initialize the cache
                ψ_acting_on = kron_index_to_vector(cols[i] - 1, @view(hi.dims[acting_on]))
                copyto!(@view(ψ_cache[acting_on]), ψ_acting_on)
                j = vector_to_kron_index(ψ_cache, hi.dims)

                idx_find = findfirst(==(j), connected_states_idxs)
                if idx_find !== nothing
                    mels[idx_find] += vals[i]
                else
                    push!(connected_states_idxs, j)
                    push!(mels, vals[i])
                end
            end
        end
    end

    connected_states = similar(ψ, length(hi), length(connected_states_idxs))
    for (i, idx) in enumerate(connected_states_idxs)
        copyto!(@view(connected_states[:, i]), kron_index_to_vector!(ψ_cache, idx - 1, hi.dims))
    end

    return connected_states, mels
end
