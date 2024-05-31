export sigmax, sigmay, sigmaz

_sigma_x_mat(T::Type{<:Number} = default_eltype) = spdiagm(1 => T[1], -1 => T[1])
_sigma_y_mat(T::Type{<:Number} = default_eltype) = spdiagm(1 => T[1im], -1 => T[-1im])
_sigma_z_mat(T::Type{<:Number} = default_eltype) = spdiagm(0 => T[-1, 1])

sigmax(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_x_mat(T))
sigmay(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_y_mat(T))
sigmaz(hilbert::Hilbert, acting_on::Int, T::Type{<:Number} = default_eltype) =
    QuantumOperator(hilbert, acting_on, _sigma_z_mat(T))
