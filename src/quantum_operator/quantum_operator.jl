export QuantumOperator
export setup_cache

struct QuantumOperator{HT<:Hilbert,DT<:AbstractDict{<:AbstractVector,<:AbstractMatrix},CT<:Ref{<:Number},IIT,CAT}
    hilbert::HT
    dict::DT
    constant::CT
    is_initialized::Val{IIT}
    cache::CAT
end

function QuantumOperator(
    hilbert::HT,
    dict::DT,
    constant::CT = 0,
) where {HT<:Hilbert,MT<:AbstractMatrix,DT<:AbstractDict{<:AbstractVector,MT},CT<:Number}
    for key in keys(dict)
        # TODO: merge also the duplicates on the product
        allunique(key) ||
            throw(ArgumentError("The are operators acting on the same subsystem that need to be multiplied first"))
        # TODO: sort the acting_on and the products_sums_unique
        issorted(key) || throw(ArgumentError("The acting_on should be sorted"))
    end

    is_initialized = Val(false)

    return QuantumOperator(hilbert, dict, Ref(constant), is_initialized, NamedTuple())
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
    _lmul_dict!(α, q.dict)

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

    q_out_dict = _convert_dict(q1.dict, T)

    # αβ + β ∑ᵢAᵢ
    q_out_c = q1.constant[] * q2.constant[]
    _lmul_dict!(q2.constant[], q_out_dict)

    # α ∑ᵢBᵢ
    mergewith!(+, q_out_dict, (q1.constant[] * q2).dict)

    # ∑ᵢⱼAᵢBⱼ
    for (key1, value1) in q1.dict
        for (key2, value2) in q2.dict
            dict_tmp = _multiply_keys_values(q1.hilbert, key1, value1, key2, value2)
            mergewith!(+, q_out_dict, dict_tmp)
        end
    end

    return QuantumOperator(q1.hilbert, q_out_dict, q_out_c)
end

setup_cache(q::QuantumOperator{HT,DT,CT,true}) where {HT,DT,CT} = q

function setup_cache(
    q::QuantumOperator{HT,DT,CRT,false},
) where {HT,KT,MT<:AbstractMatrix,DT<:AbstractDict{<:AbstractVector{KT},MT},CT<:Number,CRT<:Ref{CT}}
    dict = q.dict

    MTT = eltype(MT)
    T = promote_type(MTT, CT)
    mat1 = first(values(dict))

    max_nnz = mapreduce(nnz, max, values(dict))
    max_conn_size = mapreduce(_max_nonzeros_per_row, +, values(dict)) + !iszero(q.constant[]) + 1
    max_acting_on_size = mapreduce(length, max, keys(dict))

    mels = _get_dense_similar(mat1, T, max_conn_size)
    connected_states_cache = _get_dense_similar(mat1, length(q.hilbert), max_conn_size)
    prod_dims_cache = _get_dense_similar(mat1, KT, length(dict), max_acting_on_size)

    for (i, acting_on) in enumerate(keys(dict))
        acting_on_size = length(acting_on)
        # prod_dims_cache[i, 1] = 1 # This doesn't work with GPU arrays
        allowed_setindex!(prod_dims_cache, 1, i, 1) # This supports GPU arrays
        cumprod!(@view(prod_dims_cache[i, 2:acting_on_size]), @view(q.hilbert.dims[acting_on[2:end]]))
        reverse!(@view(prod_dims_cache[i, 1:acting_on_size]))
    end

    cache = (
        max_nnz = max_nnz,
        max_conn_size = max_conn_size,
        mels = mels,
        connected_states_cache = connected_states_cache,
        prod_dims_cache = prod_dims_cache,
    )

    is_initialized = Val(true)

    return QuantumOperator(q.hilbert, dict, q.constant, is_initialized, cache)
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
