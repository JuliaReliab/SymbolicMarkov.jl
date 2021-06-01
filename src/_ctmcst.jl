"""
Stationary Markov
"""

struct SymbolicCTMCExpression{Tv} <: AbstractSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    args::Any #Vector{<:AbstractSymbolic}
    options::Dict{Symbol,Any}
end

function _toexpr(x::SymbolicCTMCExpression)
    args = [_toexpr(e) for e = x.args]
    Expr(:call, x.op, args...)
end

function _toexpr(x::AbstractVector)
    Expr(:vect, [_toexpr(e) for e = x]...)
end

function _toexpr(x::AbstractMatrix)
    Expr(:vect, [_toexpr(e) for e = x]...)
end

function Base.show(io::IO, x::SymbolicCTMCExpression{Tv}) where Tv
    Base.show(io, x.args)
end

"""
getparams
"""

function _getparams(Q::Matrix{<:AbstractSymbolic{Tv}}) where Tv
    union([x.params for x = Q]...)
end

function _getparams(Q::SparseCSC{<:AbstractSymbolic{Tv}}) where Tv
    union([x.params for x = Q.val]...)
end

function _getparams(Q::SparseMatrixCSC{<:AbstractSymbolic{Tv}}) where Tv
    union([x.params for x = Q.nzval]...)
end

"""
gth, stgs
"""

function gth(Q::Matrix{<:AbstractSymbolic{Tv}}) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCExpression{Tv}(s, :gth, [Q], Dict{Symbol,Any}())
end

function stgs(Q::SparseCSC{<:AbstractSymbolic{Tv}}; maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCExpression{Tv}(s, :stgs, [Q], Dict{Symbol,Any}(:maxiter=>maxiter, :steps=>steps, :rtol=>rtol))
end

function stgs(Q::SparseMatrixCSC{<:AbstractSymbolic{Tv}}; maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCExpression{Tv}(s, :stgs, [Q], Dict{Symbol,Any}(:maxiter=>maxiter, :steps=>steps, :rtol=>rtol))
end

"""
operations
"""

function dot(x::SymbolicCTMCExpression{Tx}, y::Vector{<:AbstractSymbolic{Ty}}) where {Tx<:Number,Ty<:Number}
    Tv = promote_type(Tx,Ty)
    s = union(x.params, [u.params for u = y]...)
    SymbolicExpression{Tv}(s, :dot, [x, y])
end

"""
symeval(f, cache)
Return the value for expr f
"""

function _eval(::Val{:gth}, f::SymbolicCTMCExpression{Tv}, cache::SymbolicCache)::Vector{Tv} where Tv
    Q = symeval(f.args[1], cache)
    gth(Q)
end

function _eval(::Val{:stgs}, f::SymbolicCTMCExpression{Tv}, cache::SymbolicCache)::Vector{Tv} where Tv
    Q = symeval(f.args[1], cache)
    x, conv, iter, rerror = stgs(Q, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

"""
symeval(f, dvar, cache)
Return the first derivative of expr f
"""

function _eval(::Val{:stgs}, f::SymbolicCTMCExpression{Tv}, dvar::Symbol, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = symeval(f, cache)
    Q = symeval(f.args[1], cache)
    dQ = symeval(f.args[1], dvar, cache)
    s, conv, iter, rerror = stsengs(Q, pis, dQ' * pis, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GSsen did not converge", iter, rerror
    end
    s
end

"""
symeval(f, dvar, cache)
Return the second derivative of expr f
"""

function _eval(::Val{:stgs}, f::SymbolicCTMCExpression{Tv}, dvar::Tuple{Symbol,Symbol}, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = symeval(f, cache)
    dpis_a = symeval(f, dvar[1], cache)
    dpis_b = symeval(f, dvar[1], cache)

    Q = symeval(f.args[1], cache)
    dQ_a = symeval(f.args[1], dvar[1], cache)
    dQ_b = symeval(f.args[1], dvar[2], cache)
    dQ_ab = symeval(f.args[1], dvar, cache)

    s, conv, iter, rerror = stsengs(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GSsen2 did not converge", iter, rerror
    end
    s
end

