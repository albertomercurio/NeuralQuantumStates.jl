export vector_to_kron_index, kron_index_to_vector!
export get_connected_states

function vector_to_kron_index(x::AbstractVector{T1}, prod_dims::AbstractVector{T2}) where {T1,T2}
    q = dot(x, prod_dims) + 1

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

    q2 = setup_cache(q)

    mels = q2.cache.mels
    connected_states_cache = q2.cache.connected_states_cache
    prod_dims_cache = q2.cache.prod_dims_cache

    n_connected = 0

    # Let's add the identity action first
    # We reserve the first position for the identity action
    has_diagonal_terms = !iszero(q.constant[])
    c = q.constant[]
    mels[1] = c
    connected_states_cache[:, 1] .= ψ
    n_connected += 1

    for (i, (acting_on, mat)) in enumerate(q.dict)
        idx = vector_to_kron_index(@view(ψ[acting_on]), @view(prod_dims_cache[i, 1:length(acting_on)]))

        rows = rowvals(mat)
        vals = nonzeros(mat)
        idxs = nzrange(mat, floor(T1, idx))

        if length(idxs) > 0
            for j in idxs
                # If it is a diagonal term, we can just add the diagonal value
                if rows[j] == j
                    has_diagonal_terms = true
                    mels[1] += vals[j]
                else
                    ψ_tmp = @view(connected_states_cache[:, n_connected+1])
                    copyto!(ψ_tmp, ψ)

                    kron_index_to_vector!(@view(ψ_tmp[acting_on]), rows[j] - 1, @view(hi.dims[acting_on]))

                    mels[n_connected+1] = vals[j]
                    n_connected += 1
                end
            end
        end
    end

    connected_states = connected_states_cache[:, 2-has_diagonal_terms:n_connected]

    return connected_states, mels[2-has_diagonal_terms:n_connected]
end
