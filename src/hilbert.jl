export Hilbert

struct Hilbert{NT<:Tuple{Vararg{Integer}}}
    dims::NT
end

Base.:(*)(hilbert1::Hilbert, hilbert2::Hilbert) = Hilbert((hilbert1.dims..., hilbert2.dims...))

Base.:(^)(hilbert::Hilbert, n::Int) = Hilbert(Tuple(repeat(collect(hilbert.dims), n)))
