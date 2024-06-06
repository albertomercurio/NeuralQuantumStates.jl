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

function vector_to_kron_index(x::AbstractVector{T1}, M::AbstractVector{T2}) where {T1,T2}
    T = promote_type(T1, T2)

    # For some reason this allocates memory when M is already a SubArray
    # so we need to use the _prod function
    # q = reduce((acc, i) -> acc + x[i] * prod(@view(M[i+1:end])), eachindex(x), init=1)

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

function kron_index_to_vector(q::T1, M::AbstractVector{T2}) where {T1<:Integer,T2<:Integer}
    T = promote_type(T1, T2)
    n = length(M)
    x = Vector{T}(undef, n)

    return kron_index_to_vector!(x, q, M)
end

# Inplace version of findall
function findall!(A::AbstractArray)
    n = count(!iszero, A)
    I = @view(A[1:n])
    cnt = 1
    @inbounds for (i, a) in enumerate(A)
        if !iszero(a)
            I[cnt] = i
            cnt += 1
        end
    end
    return I
end

function get_connected_states(
    q::QuantumOperator{HT,DT,CT},
    ψ::AbstractVector,
) where {HT,T1,T2,DT<:AbstractDict{<:AbstractVector{T1},<:AbstractMatrix{T2}},CT}
    hi = q.hilbert

    setup_cache!(q)

    connected_states_idxs = q.cache.connected_states_idxs
    mels = q.cache.mels
    ψ_cache = q.cache.ψ_cache
    cols_cache = q.cache.cols_cache
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
        rows, cols, vals = findnz(mat)
        idx = vector_to_kron_index(@view(ψ[acting_on]), @view(hi.dims[acting_on]))

        idxs = findall!(cols_cache[1:length(cols)] .= cols .== idx)

        if length(idxs) > 0
            for i in idxs
                # If it is a diagonal term, we can just add the diagonal value
                if rows[i] == cols[i]
                    has_diagonal_terms = true
                    mels[1] += vals[i]
                else
                    copyto!(ψ_cache, ψ) # re-initialize the cache

                    kron_index_to_vector!(@view(ψ_cache[acting_on]), rows[i] - 1, @view(hi.dims[acting_on]))

                    # TODO: This commented code is slow. Check if idx_find might be === nothing
                    # idx_find = findfirst(==(ψ_cache), eachcol(@view(connected_states_cache[:, 2:n_connected])))

                    # if idx_find !== nothing
                    #     println("Found")
                    #     mels[idx_find] += vals[i]
                    # else
                    #     # connected_states_idxs[n_connected + 1] = j
                    #     mels[n_connected+1] = vals[i]
                    #     connected_states_cache[:, n_connected+1] .= ψ_cache
                    #     n_connected += 1
                    # end
                    
                    # We use this instead, without checking if the state is already in the cache
                    mels[n_connected+1] = vals[i]
                    connected_states_cache[:, n_connected+1] .= ψ_cache
                    n_connected += 1
                end
            end
        end
    end

    connected_states = @view(connected_states_cache[:, 2-has_diagonal_terms:n_connected])

    return connected_states, @view(mels[2-has_diagonal_terms:n_connected])
end
