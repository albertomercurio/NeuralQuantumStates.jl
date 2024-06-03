export QuantumOperator
# export to_sparse

struct QuantumOperator{HT<:Hilbert,DT<:AbstractDict{<:AbstractVector,<:AbstractMatrix},CT<:Ref{<:Number}}
    hilbert::HT
    dict::DT
    constant::CT
end

function QuantumOperator(
    hilbert::HT,
    dict::DT,
    constant::CT = 0,
) where {HT<:Hilbert,DT<:AbstractDict{<:AbstractVector,<:AbstractMatrix},CT<:Number}
    for (key, value) in dict
        # TODO: merge also the duplicates on the product
        allunique(key) ||
            throw(ArgumentError("The are operators acting on the same subsystem that need to be multiplied first"))
        # TODO: sort the acting_on and the products_sums_unique
        issorted(key) || throw(ArgumentError("The acting_on should be sorted"))
    end

    return QuantumOperator(hilbert, dict, Ref(constant))
end

function QuantumOperator(hi::Hilbert, ao::Int, mat::AbstractMatrix{T}, constant = zero(T)) where {T<:Number}
    return QuantumOperator(hi, Dict([ao] => mat), constant)
end

function Base.:(+)(q::QuantumOperator, α::Number)
    # TODO: do we need deepcopy here?
    return QuantumOperator(q.hilbert, _promote_quantum_operator(q, α).dict, q.constant[] + α)
end

Base.:(+)(α::Number, q1::QuantumOperator) = q1 + α

Base.:(-)(q1::QuantumOperator, α::Number) = q1 + (-1) * α

Base.:(-)(α::Number, q1::QuantumOperator) = α + (-1) * q1

Base.:(-)(q1::QuantumOperator) = -1 * q1

function Base.:(+)(q1::QuantumOperator, q2::QuantumOperator)
    _check_hilbert(q1, q2)

    return QuantumOperator(q1.hilbert, mergewith(+, q1.dict, q2.dict), q1.constant[] + q2.constant[])
end

function Base.:(-)(q1::QuantumOperator, q2::QuantumOperator)
    _check_hilbert(q1, q2)

    return q1 + (-1) * q2
end

function LinearAlgebra.lmul!(α::Number, q::QuantumOperator{HT,DT,CRT}) where {HT,DT,CT<:Number,CRT<:Ref{CT}}
    iszero(α) && (empty!(q.dict); q.constant[] = zero(CT); return q)

    for (key, value) in q.dict
        rmul!(value, α)
    end

    q.constant[] *= α

    return q
end

LinearAlgebra.rmul!(q::QuantumOperator, α::Number) = lmul!(α, q)

function Base.:(*)(α::Number, q::QuantumOperator)
    _q = _promote_quantum_operator(q, α)
    return lmul!(α, _q)
end

function Base.:(*)(q::QuantumOperator, α::Number)
    _q = _promote_quantum_operator(q, α)
    return rmul!(_q, α)
end

Base.:(/)(q::QuantumOperator, α::Number) = q * (1 / α)

function Base.:(*)(
    q1::QuantumOperator{HT1,DT1,CRT1},
    q2::QuantumOperator{HT2,DT2,CRT2},
) where {
    HT1,
    T1,
    DT1<:AbstractDict{<:AbstractVector,<:AbstractMatrix{T1}},
    CT1,
    CRT1<:Ref{CT1},
    HT2,
    T2,
    DT2<:AbstractDict{<:AbstractVector,<:AbstractMatrix{T2}},
    CT2,
    CRT2<:Ref{CT2},
}
    _check_hilbert(q1, q2)

    T = promote_type(T1, T2, CT1, CT2)

    # (α + ∑ᵢAᵢ)(β + ∑ᵢBᵢ) =
    # = αβ + α ∑ᵢBᵢ + β ∑ᵢAᵢ + ∑ᵢⱼAᵢBⱼ
    # = β(α + ∑ᵢAᵢ) + α ∑ᵢBᵢ + ∑ᵢⱼAᵢBⱼ

    q_out = _promote_quantum_operator(q1, T)

    # αβ + β ∑ᵢAᵢ
    rmul!(q_out, q2.constant[])

    # α ∑ᵢBᵢ
    mergewith!(+, q_out.dict, (q1.constant[] * q2).dict)

    # ∑ᵢⱼAᵢBⱼ
    for (key1, value1) in q1.dict
        for (key2, value2) in q2.dict
            dict_tmp = _multiply_keys_values(q1.hilbert, key1, value1, key2, value2)
            mergewith!(+, q_out.dict, dict_tmp)
        end
    end

    return QuantumOperator(q1.hilbert, q_out.dict, q1.constant[] * q2.constant[])
end

# THIS IS VERY SLOW!!!
# function to_sparse(
#     q::QuantumOperator{H,DT},
# ) where {H<:Hilbert,MT<:AbstractMatrix,DT<:AbstractDict{<:Tuple{Vararg{Integer}},MT}}
#     T = eltype(MT)
#     n = prod(q.hilbert.dims)
#     mat = spzeros(T, n, n)

#     for (key, value) in q.dict
#         acting_on = collect(key)
#         acting_on_left = 1:first(acting_on)-1
#         acting_on_right = setdiff(1:length(q.hilbert), union(acting_on_left, acting_on))
#         Id_left = I(prod(q.hilbert.dims[acting_on_left]))
#         Id_right = I(prod(q.hilbert.dims[acting_on_right]))

#         acting_on_tmp, mat_tmp = _permute_kron(q.hilbert, vcat(acting_on_left, acting_on, acting_on_right), kron(Id_left, value, Id_right))
#         mat += mat_tmp
#     end

#     return mat
# end
