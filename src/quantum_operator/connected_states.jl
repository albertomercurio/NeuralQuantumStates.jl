export vector_to_kron_index, kron_index_to_vector!
export get_connected_states

# This is to avoid to call prod(@view(M[i+1:end])) when M is already a SubArray
function _prod(i, M::AbstractVector{T}) where {T}
    res = one(T)
    @inbounds for j in i+1:length(M)
        res *= M[j]
    end
    return res
end

function vector_to_kron_index(x::AbstractVector{T1}, M::AbstractVector{T2}) where {T1,T2}
    T = promote_type(T1, T2)

    q = reduce((acc, (i, xi)) -> acc + xi * _prod(i, M), enumerate(x), init = one(T))

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

function get_connected_states(
    q::QuantumOperator{HT,DT,CT},
    ψ::AbstractVector,
) where {HT,T1,T2,DT<:AbstractDict{<:AbstractVector{T1},<:AbstractMatrix{T2}},CT}
    hi = q.hilbert

    setup_cache!(q)

    mels = q.cache.mels
    ψ_cache = q.cache.ψ_cache
    connected_states_cache = q.cache.connected_states_cache[:connected_states]

    n_connected = 0

    # Let's add the identity action first
    # We reserve the first position for the identity action
    has_diagonal_terms = !iszero(q.constant[])
    c = q.constant[]
    mels[1] = c
    connected_states_cache[:, 1] .= ψ
    n_connected += 1

    for (acting_on, mat) in q.dict
        # This would be more efficient when using SparseMatrixCOO
        idx = vector_to_kron_index(@view(ψ[acting_on]), @view(hi.dims[acting_on]))

        rows = rowvals(mat)
        vals = nonzeros(mat)
        idxs = nzrange(mat, floor(T1, idx))

        if length(idxs) > 0
            for i in idxs
                # If it is a diagonal term, we can just add the diagonal value
                if rows[i] == i
                    has_diagonal_terms = true
                    mels[1] += vals[i]
                else
                    copyto!(ψ_cache, ψ) # re-initialize the cache

                    kron_index_to_vector!(@view(ψ_cache[acting_on]), rows[i] - 1, @view(hi.dims[acting_on]))

                    mels[n_connected+1] = vals[i]
                    connected_states_cache[:, n_connected+1] .= ψ_cache
                    n_connected += 1
                end
            end
        end
    end

    connected_states = connected_states_cache[:, 2-has_diagonal_terms:n_connected]

    return connected_states, mels[2-has_diagonal_terms:n_connected]
end