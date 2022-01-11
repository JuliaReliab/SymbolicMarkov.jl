"""
Transient Markov
"""

function tprob(ts, x0::Vector{Tv}, Q::AbstractMatrix{Tv}; ufact=1.01, eps=1.0e-8, rmax=500) where Tv
    _tprob(ts, x0, Q, ufact, eps, rmax)
end

function tprob(ts, m::CTMCModel{Tv}; states = nothing, ufact=1.01, eps=1.0e-8, rmax=500) where Tv
    if states == nothing
        _tprob(ts, m.initv, m.Q, ufact, eps, rmax)
    else
        s = _getstates(m.states, states)
        _tprob(ts, m.initv, m.Q, ufact, eps, rmax)[s]
    end
end

function _tprob(ts::Tv, x0::Vector{Tv}, Q::AbstractMatrix{Tv}, ufact, eps, rmax)::Vector{Tv} where {Tv<:Number}
    mexp(Q, x0, ts, transpose=:T, ufact=ufact, eps=eps, rmax=rmax)
end

"""
ctprob
"""

function ctprob(ts, x0::Vector{Tv}, Q::AbstractMatrix{Tv}; ufact=1.01, eps=1.0e-8, rmax=500) where Tv
    _ctprob(ts, x0, Q, ufact, eps, rmax)
end

function ctprob(ts, m::CTMCModel{Tv}; states = nothing, ufact=1.01, eps=1.0e-8, rmax=500) where Tv
    if states == nothing
        _ctprob(ts, m.initv, m.Q, ufact, eps, rmax)
    else
        s = _getstates(m.states, states)
        _ctprob(ts, m.initv, m.Q, ufact, eps, rmax)[s]
    end
end

function _ctprob(ts::Tv, x0::Vector{Tv}, Q::AbstractMatrix{Tv}, ufact, eps, rmax)::Vector{Tv} where {Tv<:Number}
    mexpc(Q, x0, ts, transpose=:T, ufact=ufact, eps=eps, rmax=rmax)[2]
end


"""
symbolic
"""

mutable struct SymbolicCTMCExpAvExpression{Tv} <: AbstractVectorSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    x0::AbstractVectorSymbolic{Tv}
    Q::AbstractMatrixSymbolic{Tv}
    ts
    options::Dict{Symbol,Any}
    dim::Int
end

function _toexpr(x::SymbolicCTMCExpAvExpression)
    Expr(:call, x.op, _toexpr(x.x0), _toexpr(x.Q))
end

function Base.show(io::IO, x::SymbolicCTMCExpAvExpression{Tv}) where Tv
    Base.show(io, "SymbolicCTMCExpAvExpression $(objectid(x))")
end

###

function _tprob(ts::Tv, x0::Vector{<:AbstractSymbolic{Tv}}, Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, ufact, eps, rmax) where {Tv<:Number}
    _tprob(ts, convert(AbstractVectorSymbolic{Tv}, x0), convert(AbstractMatrixSymbolic{Tv}, Q), ufact, eps, rmax)
end

function _tprob(ts::Tv, x0::AbstractVectorSymbolic{Tv}, Q::AbstractMatrixSymbolic{Tv}, ufact, eps, rmax) where {Tv<:Number}
    s = union(x0.params, Q.params)
    SymbolicCTMCExpAvExpression{Tv}(s, :tprob, x0, Q, ts, Dict(:ufact=>ufact, :eps=>eps, :rmax=>rmax), Q.dim[1])
end

function _ctprob(ts::Tv, x0::Vector{<:AbstractSymbolic{Tv}}, Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, ufact, eps, rmax) where {Tv<:Number}
    _ctprob(ts, convert(AbstractVectorSymbolic{Tv}, x0), convert(AbstractMatrixSymbolic{Tv}, Q), ufact, eps, rmax)
end

function _ctprob(ts::Tv, x0::AbstractVectorSymbolic{Tv}, Q::AbstractMatrixSymbolic{Tv}, ufact, eps, rmax) where {Tv<:Number}
    s = union(x0.params, Q.params)
    SymbolicCTMCExpAvExpression{Tv}(s, :ctprob, x0, Q, ts, Dict(:ufact=>ufact, :eps=>eps, :rmax=>rmax), Q.dim[1])
end

"""
seval
"""

function _eval(::Val{:tprob}, f::SymbolicCTMCExpAvExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where {Tv<:Number}
    x0 = seval(f.x0, env, cache)
    Q = seval(f.Q, env, cache)
    _tprob(f.ts, x0, Q, f.options[:ufact], f.options[:eps], f.options[:rmax])
end

function _eval(::Val{:tprob}, f::SymbolicCTMCExpAvExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where {Tv<:Number}
    Q = seval(f.Q, env, cache)
    dQ = seval(f.Q, dvar, env, cache)
    x = seval(f.x0, env, cache)
    dx = seval(f.x0, dvar, env, cache)
    ts = f.ts

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    xx = [x..., dx...]
    ret = n+1:2n
    mexp(QQ, xx, ts, transpose=:T, ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])[ret]
end

function _eval(::Val{:tprob}, f::SymbolicCTMCExpAvExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where {Tv<:Number}
     Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)

    dQ_a = seval(f.Q, dvar[1], env, cache)
    dx_a = seval(f.x0, dvar[1], env, cache)

    dQ_b = seval(f.Q, dvar[2], env, cache)
    dx_b = seval(f.x0, dvar[2], env, cache)

    dQ_ab = seval(f.Q, dvar, env, cache)
    dx_ab = seval(f.x0, dvar, env, cache)

    ts = f.ts

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    xx = [x..., dx_a..., dx_b..., dx_ab...]
    ret = 3n+1:4n
    mexp(QQ, xx, ts, transpose=:T, ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])[ret]
end

###

function _eval(::Val{:ctprob}, f::SymbolicCTMCExpAvExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where {Tv<:Number}
    x0 = seval(f.x0, env, cache)
    Q = seval(f.Q, env, cache)
    _ctprob(f.ts, x0, Q, f.options[:ufact], f.options[:eps], f.options[:rmax])
end

function _eval(::Val{:ctprob}, f::SymbolicCTMCExpAvExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where {Tv<:Number}
    Q = seval(f.Q, env, cache)
    dQ = seval(f.Q, dvar, env, cache)
    x = seval(f.x0, env, cache)
    dx = seval(f.x0, dvar, env, cache)
    ts = f.ts

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    xx = [x..., dx...]
    ret = n+1:2n
    mexpc(QQ, xx, ts, transpose=:T, ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])[2][ret]
end

function _eval(::Val{:ctprob}, f::SymbolicCTMCExpAvExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where {Tv<:Number}
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)

    dQ_a = seval(f.Q, dvar[1], env, cache)
    dx_a = seval(f.x0, dvar[1], env, cache)

    dQ_b = seval(f.Q, dvar[2], env, cache)
    dx_b = seval(f.x0, dvar[2], env, cache)

    dQ_ab = seval(f.Q, dvar, env, cache)
    dx_ab = seval(f.x0, dvar, env, cache)

    ts = f.ts

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    xx = [x..., dx_a..., dx_b..., dx_ab...]
    ret = 3n+1:4n
    mexpc(QQ, xx, ts, transpose=:T, ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])[2][ret]
end

"""
exrt, cexrt
"""

function exrt(ts, m::CTMCModel{Tv}; reward, ufact=1.01, eps=1.0e-8, rmax=500) where Tv
    dot(tprob(ts, m, ufact=ufact, eps=eps, rmax=rmax), m.reward[reward])
end

function cexrt(ts, m::CTMCModel{Tv}; reward, ufact=1.01, eps=1.0e-8, rmax=500) where Tv
    dot(ctprob(ts, m, ufact=ufact, eps=eps, rmax=rmax), m.reward[reward])
end

