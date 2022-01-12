"""
exrss
"""

function exrss(Q, r; maxiter=5000, steps=20, rtol=1.0e-6) where {Tv<:Number}
    dot(prob(Q, maxiter=maxiter, steps=steps, rtol=rtol), r)
end

function exrss(m::CTMCModel{Tv}; reward, maxiter=5000, steps=20, rtol=1.0e-6) where Tv
    exrss(m.Q, m.reward[reward], maxiter=maxiter, steps=steps, rtol=rtol)
end

"""
exrt, cexrt
"""

function exrt(ts::Tt, Q, x0, r; ufact=1.01, eps=1.0e-8, rmax=500) where {Tt<:Number}
    dot(tprob(ts, Q, x0, ufact=ufact, eps=eps, rmax=rmax), r)
end

function exrt(ts::Tt, m::CTMCModel{Tv}; reward, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv,Tt<:Number}
    exrt(ts, m.Q, m.initv, m.reward[reward], ufact=ufact, eps=eps, rmax=rmax)
end

function cexrt(ts::Tt, Q, x0, r; ufact=1.01, eps=1.0e-8, rmax=500) where {Tt<:Number}
    dot(ctprob(ts, Q, x0, ufact=ufact, eps=eps, rmax=rmax), r)
end

function cexrt(ts::Tt, m::CTMCModel{Tv}; reward, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv,Tt<:Number}
    cexrt(ts, m.Q, m.initv, m.reward[reward], ufact=ufact, eps=eps, rmax=rmax)
end

###

function exrt(ts::AbstractVector{Tt}, Q, x0, r; forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tt<:Number}
    _exrt(ts, Q, x0, r, forward, ufact, eps, rmax)
end

function cexrt(ts::AbstractVector{Tt}, Q, x0, r; forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tt<:Number}
    _cexrt(ts, Q, x0, r, forward, ufact, eps, rmax)
end

function exrt(ts::AbstractVector{Tt}, m::CTMCModel{Tv}; reward, forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv,Tt<:Number}
    exrt(ts, m.Q, m.initv, m.reward[reward], forward=forward, ufact=ufact, eps=eps, rmax=rmax)
end

function cexrt(ts::AbstractVector{Tt}, m::CTMCModel{Tv}; reward, forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv,Tt<:Number}
    cexrt(ts, m.Q, m.initv, m.reward[reward], forward=forward, ufact=ufact, eps=eps, rmax=rmax)
end

"""
exrt, cexrt for ts
"""

function _exrt(ts::AbstractVector{Tt}, Q::AbstractMatrix{Tv}, x::Vector{Tv}, r::Vector{Tv}, forward, ufact, eps, rmax) where {Tv<:Number,Tt<:Number}
    result, _, = tran(Q, x, r, ts, forward=forward, ufact=ufact, eps=eps, rmax=rmax)
    result
end

function _cexrt(ts::AbstractVector{Tt}, Q::AbstractMatrix{Tv}, x::Vector{Tv}, r::Vector{Tv}, forward, ufact, eps, rmax) where {Tv<:Number,Tt<:Number}
    _, cresult, = tran(Q, x, r, ts, forward=forward, ufact=ufact, eps=eps, rmax=rmax)
    cresult
end

"""
symbolic
"""

mutable struct SymbolicCTMCTranExpression{Tv} <: AbstractVectorSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    x0::AbstractVectorSymbolic{Tv}
    Q::AbstractMatrixSymbolic{Tv}
    r::AbstractVectorSymbolic{Tv}
    ts::AbstractVector{<:Number}
    options::Dict{Symbol,Any}
    dim::Int
end

function _toexpr(x::SymbolicCTMCTranExpression)
    Expr(:call, x.op, _toexpr(x.x0), _toexpr(x.Q), _toexpr(x.r))
end

function Base.show(io::IO, x::SymbolicCTMCTranExpression{Tv}) where Tv
    Base.show(io, "SymbolicCTMCTranExpression $(objectid(x))")
end

"""
exrt, cexrt
"""

function _exrt(ts::AbstractVector{Tt}, Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::Vector{<:AbstractSymbolic{Tv}}, r::Vector{<:AbstractSymbolic{Tv}}, forward, ufact, eps, rmax) where {Tv<:Number,Tt<:Number}
    _exrt(ts, convert(AbstractMatrixSymbolic{Tv}, Q), convert(AbstractVectorSymbolic{Tv}, x), convert(AbstractVectorSymbolic{Tv}, r), forward, ufact, eps, rmax)
end

function _cexrt(ts::AbstractVector{Tt}, Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::Vector{<:AbstractSymbolic{Tv}}, r::Vector{<:AbstractSymbolic{Tv}}, forward, ufact, eps, rmax) where {Tv<:Number,Tt<:Number}
    _cexrt(ts, convert(AbstractMatrixSymbolic{Tv}, Q), convert(AbstractVectorSymbolic{Tv}, x), convert(AbstractVectorSymbolic{Tv}, r), forward, ufact, eps, rmax)
end

function _exrt(ts::AbstractVector{Tt}, Q::AbstractMatrixSymbolic{Tv}, x::AbstractVectorSymbolic{Tv}, r::AbstractVectorSymbolic{Tv}, forward, ufact, eps, rmax) where {Tv<:Number,Tt<:Number}
    s = union(x.params, Q.params, r.params)
    SymbolicCTMCTranExpression{Tv}(s, :exrt, x, Q, r, ts, Dict(:forward=>forward, :ufact=>ufact, :eps=>eps, :rmax=>rmax), length(ts))
end

function _cexrt(ts::AbstractVector{Tt}, Q::AbstractMatrixSymbolic{Tv}, x::AbstractVectorSymbolic{Tv}, r::AbstractVectorSymbolic{Tv}, forward, ufact, eps, rmax) where {Tv<:Number,Tt<:Number}
    s = union(x.params, Q.params, r.params)
    SymbolicCTMCTranExpression{Tv}(s, :cexrt, x, Q, r, ts, Dict(:forward=>forward, :ufact=>ufact, :eps=>eps, :rmax=>rmax), length(ts))
end

"""
seval(f, env, cache)
Return the value for expr f
"""

function _eval(::Val{:exrt}, f::SymbolicCTMCTranExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)
    r = seval(f.r, env, cache)
    ts = f.ts
    result, cresult, = tran(Q, x, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result
end

function _eval(::Val{:exrt}, f::SymbolicCTMCTranExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)
    r = seval(f.r, env, cache)
    dQ = seval(f.Q, dvar, env, cache)
    dx = seval(f.x0, dvar, env, cache)
    dr = seval(f.r, dvar, env, cache)
    ts = f.ts

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    xx = [x..., dx...]
    rr = [dr..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result
end

function _eval(::Val{:exrt}, f::SymbolicCTMCTranExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)
    r = seval(f.r, env, cache)

    dQ_a = seval(f.Q, dvar[1], env, cache)
    dx_a = seval(f.x0, dvar[1], env, cache)
    dr_a = seval(f.r, dvar[1], env, cache)

    dQ_b = seval(f.Q, dvar[2], env, cache)
    dx_b = seval(f.x0, dvar[2], env, cache)
    dr_b = seval(f.r, dvar[2], env, cache)

    dQ_ab = seval(f.Q, dvar, env, cache)
    dx_ab = seval(f.x0, dvar, env, cache)
    dr_ab = seval(f.r, dvar, env, cache)
    ts = f.ts

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    xx = [x..., dx_a..., dx_b..., dx_ab...]
    rr = [dr_ab..., dr_b..., dr_a..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result
end

"""
seval for cexrt
"""

function _eval(::Val{:cexrt}, f::SymbolicCTMCTranExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)
    r = seval(f.r, env, cache)
    ts = f.ts
    result, cresult, = tran(Q, x, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    cresult
end

function _eval(::Val{:cexrt}, f::SymbolicCTMCTranExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)
    r = seval(f.r, env, cache)
    dQ = seval(f.Q, dvar, env, cache)
    dx = seval(f.x0, dvar, env, cache)
    dr = seval(f.r, dvar, env, cache)
    ts = f.ts

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    xx = [x..., dx...]
    rr = [dr..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    cresult
end

function _eval(::Val{:cexrt}, f::SymbolicCTMCTranExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.Q, env, cache)
    x = seval(f.x0, env, cache)
    r = seval(f.r, env, cache)

    dQ_a = seval(f.Q, dvar[1], env, cache)
    dx_a = seval(f.x0, dvar[1], env, cache)
    dr_a = seval(f.r, dvar[1], env, cache)

    dQ_b = seval(f.Q, dvar[2], env, cache)
    dx_b = seval(f.x0, dvar[2], env, cache)
    dr_b = seval(f.r, dvar[2], env, cache)

    dQ_ab = seval(f.Q, dvar, env, cache)
    dx_ab = seval(f.x0, dvar, env, cache)
    dr_ab = seval(f.r, dvar, env, cache)
    ts = f.ts

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    xx = [x..., dx_a..., dx_b..., dx_ab...]
    rr = [dr_ab..., dr_b..., dr_a..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    cresult
end
