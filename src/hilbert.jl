export Hilbert

struct Hilbert{NT<:AbstractVector}
    dims::NT
end

Hilbert(dim::Int) = Hilbert([dim])

Base.length(hilbert::Hilbert) = length(hilbert.dims)

Base.:(*)(hilbert1::Hilbert, hilbert2::Hilbert) = Hilbert(vcat(hilbert1.dims, hilbert2.dims))

Base.:(^)(hilbert::Hilbert, n::Int) = Hilbert(repeat(hilbert.dims, n))
