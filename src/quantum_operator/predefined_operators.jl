export sigmax, sigmay, sigmaz, sigmap, sigmam

# This is much faster than calling spdiagm
_sigma_x_mat(T::Type{<:Number} = default_eltype) = sparse([1, 2], [2, 1], T[1, 1], 2, 2)
_sigma_y_mat(T::Type{<:Number} = default_eltype) = sparse([1, 2], [2, 1], T[1im, -1im], 2, 2)
_sigma_z_mat(T::Type{<:Number} = default_eltype) = sparse([1, 2], [1, 2], T[-1, 1], 2, 2)
_sigma_p_mat(T::Type{<:Number} = default_eltype) = sparse([2], [1], T[1], 2, 2)
_sigma_m_mat(T::Type{<:Number} = default_eltype) = sparse([1], [2], T[1], 2, 2)

sigmax(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_x_mat(T))
sigmay(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_y_mat(T))
sigmaz(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_z_mat(T))
sigmap(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_p_mat(T))
sigmam(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_m_mat(T))
