"""
sprob
"""

function _getstates(states::Vector{Symbol}, x::Vector{Symbol})::Vector{Int}
    h = Dict(s=>i for (i,s)=enumerate(states))
    [h[s] for s = x]
end

function sprob(Q::Matrix{Tv}; maxiter=5000, steps=20, rtol=1.0e-6) where Tv
    _sprob(Q, maxiter, steps, rtol)
end

function sprob(Q::SparseCSC{Tv,Ti}; maxiter=5000, steps=20, rtol=1.0e-6) where {Tv,Ti}
    _sprob(Q, maxiter, steps, rtol)
end

function sprob(Q::SparseMatrixCSC{Tv,Ti}; maxiter=5000, steps=20, rtol=1.0e-6) where {Tv,Ti}
    _sprob(Q, maxiter, steps, rtol)
end

function sprob(m::CTMCModel{Tv}; states = nothing, maxiter=5000, steps=20, rtol=1.0e-6) where Tv
    if states == nothing
        _sprob(m.Q, maxiter, steps, rtol)
    else
        s = _getstates(m.states, states)
        _sprob(m.Q, maxiter, steps, rtol)[s]
    end
end

function _sprob(Q::Matrix{Tv}, maxiter, steps, rtol)::Vector{Tv} where {Tv<:Number}
    gth(Q)
end

function _sprob(Q::SparseCSC{Tv,Ti}, maxiter, steps, rtol)::Vector{Tv} where {Tv<:Number,Ti}
    x, conv, iter, rerror = stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

function _sprob(Q::SparseMatrixCSC{Tv,Ti}, maxiter, steps, rtol)::Vector{Tv} where {Tv<:Number,Ti}
    x, conv, iter, rerror = stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

##

function _sprobsen(Q::Matrix{Tv}, pis::Vector{Tv}, b::Vector{Tv}, maxiter, steps, rtol)::Vector{Tv} where {Tv<:Number}
    stsen(Q, pis, b)
end

function _sprobsen(Q::SparseCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv}, maxiter, steps, rtol)::Vector{Tv} where {Tv<:Number,Ti}
    s, conv, iter, rerror = stsengs(Q, pis, b, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GSsen did not converge", iter, rerror
    end
    s
end

function _sprobsen(Q::SparseMatrixCSC{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv}, maxiter, steps, rtol)::Vector{Tv} where {Tv<:Number,Ti}
    s, conv, iter, rerror = stsengs(Q, pis, b, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GSsen did not converge", iter, rerror
    end
    s
end

##

mutable struct SymbolicCTMCProbExpression{Tv} <: AbstractVectorSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    Q::AbstractMatrixSymbolic{Tv}
    options::Dict{Symbol,Any}
    dim::Int
end

function _toexpr(x::SymbolicCTMCProbExpression)
    Expr(:call, x.op, _toexpr(x.Q))
end

function Base.show(io::IO, x::SymbolicCTMCProbExpression{Tv}) where Tv
    Base.show(io, "SymbolicCTMCProbExpression $(objectid(x))")
end

##

function sprob(Q::AbstractMatrixSymbolic{Tv}; maxiter=5000, steps=20, rtol=1.0e-6) where Tv
    _sprob(Q, maxiter, steps, rtol)
end

function _sprob(Q::Matrix{<:AbstractSymbolic{Tv}}, maxiter, steps, rtol) where {Tv<:Number}
    _sprob(convert(AbstractMatrixSymbolic{Tv}, Q), maxiter, steps, rtol)
end

function _sprob(Q::SparseCSC{<:AbstractSymbolic{Tv},Ti}, maxiter, steps, rtol) where {Tv<:Number,Ti}
    _sprob(convert(AbstractMatrixSymbolic{Tv}, Q), maxiter, steps, rtol)
end

function _sprob(Q::SparseMatrixCSC{<:AbstractSymbolic{Tv},Ti}, maxiter, steps, rtol) where {Tv<:Number,Ti}
    _sprob(convert(AbstractMatrixSymbolic{Tv}, Q), maxiter, steps, rtol)
end

function _sprob(Q::AbstractMatrixSymbolic{Tv}, maxiter, steps, rtol) where {Tv<:Number}
    SymbolicCTMCProbExpression{Tv}(Q.params, :sprob, Q, Dict(:maxiter=>maxiter, :steps=>steps, :rtol=>rtol), Q.dim[1])
end

"""
seval
"""

function _eval(::Val{:sprob}, f::SymbolicCTMCProbExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where {Tv<:Number}
    Q = seval(f.Q, env, cache)
    _sprob(Q, f.options[:maxiter], f.options[:steps], f.options[:rtol])
end

function _eval(::Val{:sprob}, f::SymbolicCTMCProbExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where {Tv<:Number}
    pis = seval(f, env, cache)
    Q = seval(f.Q, env, cache)
    dQ = seval(f.Q, dvar, env, cache)
    _sprobsen(Q, pis, dQ' * pis, f.options[:maxiter], f.options[:steps], f.options[:rtol])
end

function _eval(::Val{:sprob}, f::SymbolicCTMCProbExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where {Tv<:Number}
    pis = seval(f, env, cache)
    dpis_a = seval(f, dvar[1], env, cache)
    dpis_b = seval(f, dvar[1], env, cache)

    Q = seval(f.Q, env, cache)
    dQ_a = seval(f.Q, dvar[1], env, cache)
    dQ_b = seval(f.Q, dvar[2], env, cache)
    dQ_ab = seval(f.Q, dvar, env, cache)

    _sprobsen(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a, f.options[:maxiter], f.options[:steps], f.options[:rtol])
end
