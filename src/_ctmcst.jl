"""
Stationary Markov
"""

mutable struct SymbolicCTMCStExpression{Tv} <: AbstractVectorSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    args::Any #Vector{<:AbstractSymbolic}
    options::Dict{Symbol,Any}
    dim::Int
end

function _toexpr(x::SymbolicCTMCStExpression)
    args = [_toexpr(e) for e = x.args]
    Expr(:call, x.op, args...)
end

function _toexpr(x::AbstractVector)
    Expr(:vect, [_toexpr(e) for e = x]...)
end

function _toexpr(x::AbstractMatrix)
    Expr(:vect, [_toexpr(e) for e = x]...)
end

function Base.show(io::IO, x::SymbolicCTMCStExpression{Tv}) where Tv
    Base.show(io, "SymbolicCTMCStExpression $(objectid(x))")
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
    SymbolicCTMCStExpression{Tv}(s, :gth, [Q], Dict{Symbol,Any}(), size(Q)[1])
end

function stgs(Q::SparseCSC{<:AbstractSymbolic{Tv}}; maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCStExpression{Tv}(s, :stgs, [Q], Dict{Symbol,Any}(:maxiter=>maxiter, :steps=>steps, :rtol=>rtol), size(Q)[1])
end

function stgs(Q::SparseMatrixCSC{<:AbstractSymbolic{Tv}}; maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCStExpression{Tv}(s, :stgs, [Q], Dict{Symbol,Any}(:maxiter=>maxiter, :steps=>steps, :rtol=>rtol), size(Q)[1])
end

"""
operations
"""

function dot(x::SymbolicCTMCStExpression{Tx}, y::Vector{<:AbstractSymbolic{Ty}}) where {Tx<:Number,Ty<:Number}
    Tv = promote_type(Tx,Ty)
    s = union(x.params, [u.params for u = y]...)
    SymbolicExpression{Tv}(s, :dot, [x, y])
end

"""
seval(f, env, cache)
Return the value for expr f
"""

function _eval(::Val{:gth}, f::SymbolicCTMCStExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    Q = seval(f.args[1], env, cache)
    gth(Q)
end

function _eval(::Val{:stgs}, f::SymbolicCTMCStExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    Q = seval(f.args[1], env, cache)
    x, conv, iter, rerror = stgs(Q, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

"""
seval(f, dvar, env, cache)
Return the first derivative of expr f
"""

function _eval(::Val{:gth}, f::SymbolicCTMCStExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = seval(f, env, cache)
    Q = seval(f.args[1], env, cache)
    dQ = seval(f.args[1], dvar, env, cache)
    stsen(Q, pis, dQ' * pis)
end

function _eval(::Val{:stgs}, f::SymbolicCTMCStExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = seval(f, env, cache)
    Q = seval(f.args[1], env, cache)
    dQ = seval(f.args[1], dvar, env, cache)
    s, conv, iter, rerror = stsengs(Q, pis, dQ' * pis, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GSsen did not converge", iter, rerror
    end
    s
end

"""
seval(f, dvar, env, cache)
Return the second derivative of expr f
"""

function _eval(::Val{:gth}, f::SymbolicCTMCStExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = seval(f, env, cache)
    dpis_a = seval(f, dvar[1], env, cache)
    dpis_b = seval(f, dvar[1], env, cache)

    Q = seval(f.args[1], env, cache)
    dQ_a = seval(f.args[1], dvar[1], env, cache)
    dQ_b = seval(f.args[1], dvar[2], env, cache)
    dQ_ab = seval(f.args[1], dvar, env, cache)

    stsen(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a)
end

function _eval(::Val{:stgs}, f::SymbolicCTMCStExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = seval(f, env, cache)
    dpis_a = seval(f, dvar[1], env, cache)
    dpis_b = seval(f, dvar[1], env, cache)

    Q = seval(f.args[1], env, cache)
    dQ_a = seval(f.args[1], dvar[1], env, cache)
    dQ_b = seval(f.args[1], dvar[2], env, cache)
    dQ_ab = seval(f.args[1], dvar, env, cache)

    s, conv, iter, rerror = stsengs(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GSsen2 did not converge", iter, rerror
    end
    s
end

# function _eval(::Val{:ctmcst}, f::SymbolicExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
#     Q = seval(f.args[1], env, cache)
#     dQ_a = seval(f.args[1], dvar[1], env, cache)
#     dQ_b = seval(f.args[1], dvar[2], env, cache)
#     dQ_ab = seval(f.args[1], dvar, env, cache)

#     pis = _stsolve(Q)
#     dpis_a = _stsensolve(Q, pis, dQ_a' * pis)
#     dpis_b = _stsensolve(Q, pis, dQ_b' * pis)
#     dpis_ab = _stsensolve(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a)
#     dpis_ab
# end

# function _eval(::Val{:ctmcst2}, f::SymbolicExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Tv where Tv
#     Q = seval(f.args[1], env, cache)
#     dQ_a = seval(f.args[1], dvar[1], env, cache)
#     dQ_b = seval(f.args[1], dvar[2], env, cache)
#     dQ_ab = seval(f.args[1], dvar, env, cache)
#     r = seval(f.args[2], env, cache)
#     dr_a = seval(f.args[2], dvar[1], env, cache)
#     dr_b = seval(f.args[2], dvar[2], env, cache)
#     dr_ab = seval(f.args[2], dvar, env, cache)

#     pis = _stsolve(Q)
#     dpis_a = _stsensolve(Q, pis, dQ_a' * pis)
#     dpis_b = _stsensolve(Q, pis, dQ_b' * pis)
#     dpis_ab = _stsensolve(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a)
#     @dot(dpis_ab, r) + @dot(dpis_a, dr_b) + @dot(dpis_b, dr_a) + @dot(pis, dr_ab)
# end

"""
_stsolve
Solve the stationary vector
"""

# function _stsolve(Q::Matrix{Tv})::Vector{Tv} where Tv
#     gth(Q)
# end

# SymbolicMarkovConfig

# function _stsolve(Q::AbstractSparseM{Tv,Ti})::Vector{Tv} where {Tv,Ti}
#     x, conv, = stgs(Q, maxiter=SymbolicMarkovConfig[:maxiter], steps=SymbolicMarkovConfig[:steps], reltol=SymbolicMarkovConfig[:reltol])
#     println(conv)
#     if conv == true
#         x
#     else
#         throw(ErrorException("stgs did not converge"))
#     end
# end

# function _stsensolve(Q::AbstractSparseM{Tv,Ti}, pis::Vector{Tv}, b::Vector{Tv})::Vector{Tv} where {Tv,Ti}
#     x, conv, = stsengs(Q, pis, b, maxiter=SymbolicMarkovConfig[:maxiter], steps=SymbolicMarkovConfig[:steps], reltol=SymbolicMarkovConfig[:reltol])
#     if conv == true
#         x
#     else
#         throw(ErrorException("stsengs did not converge"))
#     end
# end
