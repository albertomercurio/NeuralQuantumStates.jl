
function _check_hilbert(q1::QuantumOperator{HT1}, q2::QuantumOperator{HT2}) where {HT1<:Hilbert,HT2<:Hilbert}
    if q1.hilbert != q2.hilbert
        throw(ArgumentError("Hilbert spaces are different"))
    end
end

_promote_quantum_operator(q::QuantumOperator, α::T) where {T<:Number} = _promote_quantum_operator(q, T)

function _promote_quantum_operator(
    q::QuantumOperator{HT,DT,CT},
    T1::Type{<:Number},
) where {HT<:Hilbert,T2<:Number,T3<:Number,KT,DT<:Dict{KT,<:AbstractMatrix{T2}},CT<:Ref{T3}}
    T = promote_type(T1, T2, T3)
    dict = Dict(key => T.(value) for (key, value) in q.dict)
    constant = T(q.constant[])

    return QuantumOperator(q.hilbert, dict, constant)
end

function _permute_kron(hilbert, acting_on, mat)
    issorted(acting_on) && return acting_on, mat

    n = size(mat, 1)
    perm = sortperm(acting_on)
    sizes = collect(hilbert.dims[acting_on])

    mat = reshape(mat, reverse!(repeat(sizes, 2))...)
    to_permute = 2 * length(perm) + 1 .- vcat(perm, perm .+ length(perm))
    reverse!(to_permute)
    mat = PermutedDimsArray(mat, to_permute)
    mat = reshape(mat, n, n)

    # We need to copy since the reshape and the PermutedDimsArray return a view
    return acting_on[perm], copy(mat)
end

function _multiply_keys_values(
    hilbert::Hilbert,
    key1::KT1,
    value1::VT1,
    key2::KT2,
    value2::VT2,
) where {KT1<:Tuple,KT2<:Tuple,VT1<:AbstractMatrix,VT2<:AbstractMatrix}
    key1 == key2 && return Dict(key1 => value1 * value2)

    if length(intersect(key1, key2)) == 0
        acting_on, mat_product = _permute_kron(hilbert, vcat(collect(key1), collect(key2)), kron(value1, value2))

        return Dict(Tuple(acting_on) => mat_product)
    end

    acting_on = sort!(union(key1, key2))
    acting_on1 = collect(key1)
    acting_on2 = collect(key2)
    mat1 = copy(value1)
    mat2 = copy(value2)

    @inbounds for i in eachindex(acting_on)
        if !(acting_on[i] in key1)
            mat1 = kron(mat1, I(hilbert.dims[acting_on[i]]))
            push!(acting_on1, acting_on[i])
        end

        if !(acting_on[i] in key2)
            mat2 = kron(mat2, I(hilbert.dims[acting_on[i]]))
            push!(acting_on2, acting_on[i])
        end
    end

    acting_on1, mat1 = _permute_kron(hilbert, acting_on1, mat1)
    acting_on2, mat2 = _permute_kron(hilbert, acting_on2, mat2)

    acting_on1 == acting_on2 || throw(ArgumentError("The acting_on should be the same"))

    return Dict(Tuple(acting_on1) => mat1 * mat2)
end
