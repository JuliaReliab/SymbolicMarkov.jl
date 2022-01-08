mutable struct SymbolicCTMCTranExpression{Tv} <: AbstractSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    args::Any #Vector{<:AbstractSymbolic}
    options::Dict{Symbol,Any}
end

function _toexpr(x::SymbolicCTMCTranExpression)
    args = [_toexpr(e) for e = x.args]
    Expr(:call, x.op, args...)
end

function Base.show(io::IO, x::SymbolicCTMCTranExpression{Tv}) where Tv
    Base.show(io, "SymbolicCTMCTranExpression $(objectid(x))")
end

"""
Transient Markov
"""

function tran(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, r::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{Tv};
    forward::Symbol=:T, ufact::Tv=1.01, eps::Tv=1.0e-8, rmax=500, cumulative=false) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCTranExpression{Tv}(s, :tran, [Q,x,r], Dict{Symbol,Any}(:ts=>ts, :forward=>forward, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative))
end

function tran(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, r::AbstractVector{<:AbstractSymbolic{Tv}}, ts::Tv;
    forward::Symbol=:T, ufact::Tv=1.01, eps::Tv=1.0e-8, rmax=500, cumulative=false) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCTranExpression{Tv}(s, :tran1, [Q,x,r], Dict{Symbol,Any}(:ts=>[Tv(0), ts], :forward=>forward, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative))
end

"""
seval(f, env, cache)
Return the value for expr f
"""

function _eval(::Val{:tran}, f::SymbolicCTMCTranExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    r = seval(f.args[3], env, cache)
    ts = f.options[:ts]
    result, cresult, = tran(Q, x, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    if f.options[:cumulative]
        cresult
    else
        result
    end
end

function _eval(::Val{:tran1}, f::SymbolicCTMCTranExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    r = seval(f.args[3], env, cache)
    ts = f.options[:ts]
    result, cresult, = tran(Q, x, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    if f.options[:cumulative]
        cresult[2]
    else
        result[2]
    end
end

# """
# seval(f, dvar, env, cache)
# Return the first derivative of expr f
# """

function _eval(::Val{:tran}, f::SymbolicCTMCTranExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    r = seval(f.args[3], env, cache)
    dQ = seval(f.args[1], dvar, env, cache)
    dx = seval(f.args[2], dvar, env, cache)
    dr = seval(f.args[3], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    xx = [x..., dx...]
    rr = [dr..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    
    if f.options[:cumulative]
        cresult
    else
        result
    end
end

function _eval(::Val{:tran1}, f::SymbolicCTMCTranExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    r = seval(f.args[3], env, cache)
    dQ = seval(f.args[1], dvar, env, cache)
    dx = seval(f.args[2], dvar, env, cache)
    dr = seval(f.args[3], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    xx = [x..., dx...]
    rr = [dr..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    
    if f.options[:cumulative]
        cresult[2]
    else
        result[2]
    end
end

# """
# seval(f, dvar, env, cache)
# Return the second derivative of expr f
# """

function _eval(::Val{:tran}, f::SymbolicCTMCTranExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    r = seval(f.args[3], env, cache)

    dQ_a = seval(f.args[1], dvar[1], env, cache)
    dx_a = seval(f.args[2], dvar[1], env, cache)
    dr_a = seval(f.args[3], dvar[1], env, cache)

    dQ_b = seval(f.args[1], dvar[2], env, cache)
    dx_b = seval(f.args[2], dvar[2], env, cache)
    dr_b = seval(f.args[3], dvar[2], env, cache)

    dQ_ab = seval(f.args[1], dvar, env, cache)
    dx_ab = seval(f.args[2], dvar, env, cache)
    dr_ab = seval(f.args[3], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    xx = [x..., dx_a..., dx_b..., dx_ab...]
    rr = [dr_ab..., dr_b..., dr_a..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])

    if f.options[:cumulative]
        cresult
    else
        result
    end
end

function _eval(::Val{:tran1}, f::SymbolicCTMCTranExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    r = seval(f.args[3], env, cache)

    dQ_a = seval(f.args[1], dvar[1], env, cache)
    dx_a = seval(f.args[2], dvar[1], env, cache)
    dr_a = seval(f.args[3], dvar[1], env, cache)

    dQ_b = seval(f.args[1], dvar[2], env, cache)
    dx_b = seval(f.args[2], dvar[2], env, cache)
    dr_b = seval(f.args[3], dvar[2], env, cache)

    dQ_ab = seval(f.args[1], dvar, env, cache)
    dx_ab = seval(f.args[2], dvar, env, cache)
    dr_ab = seval(f.args[3], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    xx = [x..., dx_a..., dx_b..., dx_ab...]
    rr = [dr_ab..., dr_b..., dr_a..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])

    if f.options[:cumulative]
        cresult[2]
    else
        result[2]
    end
end
