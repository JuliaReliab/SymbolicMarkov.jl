"""
Transient Markov
"""

function tran(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, r::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{Tv};
    forward::Symbol=:T, ufact::Tv=1.01, eps::Tv=1.0e-8, rmax=500, cumulative=false) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCExpression{Tv}(s, :tran, [Q,x,r], Dict{Symbol,Any}(:ts=>ts, :forward=>forward, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative))
end

"""
symboliceval(f, env, cache)
Return the value for expr f
"""

function _eval(::Val{:tran}, f::SymbolicCTMCExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = symboliceval(f.args[1], env, cache)
    x = symboliceval(f.args[2], env, cache)
    r = symboliceval(f.args[3], env, cache)
    ts = f.options[:ts]
    result, cresult, = tran(Q, x, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    if f.options[:cumulative]
        cresult
    else
        result
    end
end

# """
# symboliceval(f, dvar, env, cache)
# Return the first derivative of expr f
# """

function _eval(::Val{:tran}, f::SymbolicCTMCExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = symboliceval(f.args[1], env, cache)
    x = symboliceval(f.args[2], env, cache)
    r = symboliceval(f.args[3], env, cache)
    dQ = symboliceval(f.args[1], dvar, env, cache)
    dx = symboliceval(f.args[2], dvar, env, cache)
    dr = symboliceval(f.args[3], dvar, env, cache)
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

# """
# symboliceval(f, dvar, env, cache)
# Return the second derivative of expr f
# """

function _eval(::Val{:tran}, f::SymbolicCTMCExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache) where Tv
    Q = symboliceval(f.args[1], env, cache)
    x = symboliceval(f.args[2], env, cache)
    r = symboliceval(f.args[3], env, cache)

    dQ_a = symboliceval(f.args[1], dvar[1], env, cache)
    dx_a = symboliceval(f.args[2], dvar[1], env, cache)
    dr_a = symboliceval(f.args[3], dvar[1], env, cache)

    dQ_b = symboliceval(f.args[1], dvar[2], env, cache)
    dx_b = symboliceval(f.args[2], dvar[2], env, cache)
    dr_b = symboliceval(f.args[3], dvar[2], env, cache)

    dQ_ab = symboliceval(f.args[1], dvar, env, cache)
    dx_ab = symboliceval(f.args[2], dvar, env, cache)
    dr_ab = symboliceval(f.args[3], dvar, env, cache)
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
