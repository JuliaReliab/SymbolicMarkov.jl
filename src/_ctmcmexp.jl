mutable struct SymbolicCTMCExpExpression{Tv} <: AbstractVectorSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    args::Any #Vector{<:AbstractSymbolic}
    options::Dict{Symbol,Any}
    dim::Int
end

function _toexpr(x::SymbolicCTMCExpExpression)
    args = [_toexpr(e) for e = x.args]
    Expr(:call, x.op, args...)
end

function Base.show(io::IO, x::SymbolicCTMCExpExpression{Tv}) where Tv
    Base.show(io, "SymbolicCTMCExpExpression $(objectid(x))")
end

"""
Transient Markov
"""

function mexp(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, ts::Tv;
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500, cumulative=false) where {Tv<:Number}
    sx = _getparams(x)
    sQ = _getparams(Q)
    s = union(sQ, sx)
    SymbolicCTMCExpExpression{Tv}(s, :mexp, [Q,x],
        Dict{Symbol,Any}(:ts=>ts, :transpose=>transpose, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative),
        size(Q)[1])
end

function mexp(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{Tv};
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500, cumulative=false) where {Tv<:Number}
    sx = _getparams(x)
    sQ = _getparams(Q)
    s = union(sQ, sx)
    SymbolicCTMCExpExpression{Tv}(s, :mexp, [Q,x],
        Dict{Symbol,Any}(:ts=>ts, :transpose=>transpose, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative),
        size(Q)[1])
end

function mexpc(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, ts::Tv;
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500, cumulative=false) where {Tv<:Number}
    sx = _getparams(x)
    sQ = _getparams(Q)
    s = union(sQ, sx)
    SymbolicCTMCExpExpression{Tv}(s, :mexpc, [Q,x],
        Dict{Symbol,Any}(:ts=>ts, :transpose=>transpose, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative),
        size(Q)[1])
end

function mexpc(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{Tv};
    transpose::Symbol=:N, ufact::Tv=Tv(1.01), eps::Tv=Tv(1.0e-8), rmax=500, cumulative=false) where {Tv<:Number}
    sx = _getparams(x)
    sQ = _getparams(Q)
    s = union(sQ, sx)
    SymbolicCTMCExpExpression{Tv}(s, :mexpc, [Q,x],
        Dict{Symbol,Any}(:ts=>ts, :transpose=>transpose, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative),
        size(Q)[1])
end

"""
seval(f, env, cache)
Return the value for expr f
"""

function _eval(::Val{:mexp}, f::SymbolicCTMCExpExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    ts = f.options[:ts]
    result = mexp(Q, x, ts, transpose=f.options[:transpose], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result
end

function _eval(::Val{:mexpc}, f::SymbolicCTMCExpExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    ts = f.options[:ts]
    result, cresult = mexpc(Q, x, ts, transpose=f.options[:transpose], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result, cresult
end

# """
# seval(f, dvar, env, cache)
# Return the first derivative of expr f
# """

function _eval(::Val{:mexp}, f::SymbolicCTMCExpExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    dQ = seval(f.args[1], dvar, env, cache)
    dx = seval(f.args[2], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    if f.options[:transpose] == :N
        xx = [dx..., x...]
        ret = 1:n
    else
        xx = [x..., dx...]
        ret = n+1:2n
    end
    result = mexp(QQ, xx, ts, transpose=f.options[:transpose], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result[ret]
end

## TODO: The result is wrong when the derivative by a variable which is not included in the formula
function _eval(::Val{:mexpc}, f::SymbolicCTMCExpExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)
    dQ = seval(f.args[1], dvar, env, cache)
    dx = seval(f.args[2], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)

    QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
    if f.options[:transpose] == :N
        xx = [dx..., x...]
        ret = 1:n
    else
        xx = [x..., dx...]
        ret = n+1:2n
    end
    result, cresult = mexpc(QQ, xx, ts, transpose=f.options[:transpose], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result[ret], cresult[ret]
end

# """
# seval(f, dvar, env, cache)
# Return the second derivative of expr f
# """

function _eval(::Val{:mexp}, f::SymbolicCTMCExpExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)

    dQ_a = seval(f.args[1], dvar[1], env, cache)
    dx_a = seval(f.args[2], dvar[1], env, cache)

    dQ_b = seval(f.args[1], dvar[2], env, cache)
    dx_b = seval(f.args[2], dvar[2], env, cache)

    dQ_ab = seval(f.args[1], dvar, env, cache)
    dx_ab = seval(f.args[2], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    if f.options[:transpose] == :N
        xx = [dx_ab..., dx_b..., dx_a..., x...]
        ret = 1:n
    else
        xx = [x..., dx_a..., dx_b..., dx_ab...]
        ret = 3n+1:4n
    end
    result = mexp(QQ, xx, ts, transpose=f.options[:transpose], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result[ret]
end


## TODO: The result is wrong when the derivative by a variable which is not included in the formula
function _eval(::Val{:mexpc}, f::SymbolicCTMCExpExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = seval(f.args[1], env, cache)
    x = seval(f.args[2], env, cache)

    dQ_a = seval(f.args[1], dvar[1], env, cache)
    dx_a = seval(f.args[2], dvar[1], env, cache)

    dQ_b = seval(f.args[1], dvar[2], env, cache)
    dx_b = seval(f.args[2], dvar[2], env, cache)

    dQ_ab = seval(f.args[1], dvar, env, cache)
    dx_ab = seval(f.args[2], dvar, env, cache)
    ts = f.options[:ts]

    n = length(x)
    QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
    if f.options[:transpose] == :N
        xx = [dx_ab..., dx_b..., dx_a..., x...]
        ret = 1:n
    else
        xx = [x..., dx_a..., dx_b..., dx_ab...]
        ret = 3n+1:4n
    end
    result, cresult = mexpc(QQ, xx, ts, transpose=f.options[:transpose], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    result[ret], cresult[ret]
end