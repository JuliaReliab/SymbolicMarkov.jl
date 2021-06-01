"""
Transient Markov
"""

function tran(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, r::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{Tv};
    forward::Symbol=:T, ufact::Tv=1.01, eps::Tv=1.0e-8, rmax=500, cumulative=false) where {Tv<:Number}
    s = _getparams(Q)
    SymbolicCTMCExpression{Tv}(s, :tran, [Q,x,r], Dict{Symbol,Any}(:ts=>ts, :forward=>forward, :ufact=>ufact, :eps=>eps, :rmax=>rmax, :cumulative=>cumulative))
end

"""
symeval(f, cache)
Return the value for expr f
"""

function _eval(::Val{:tran}, f::SymbolicCTMCExpression{Tv}, cache::SymbolicCache) where Tv
    Q = symeval(f.args[1], cache)
    x = symeval(f.args[2], cache)
    r = symeval(f.args[3], cache)
    ts = f.options[:ts]
    result, cresult, = tran(Q, x, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    if f.options[:cumulative]
        cresult
    else
        result
    end
end

# """
# symeval(f, dvar, cache)
# Return the first derivative of expr f
# """

function _eval(::Val{:tran}, f::SymbolicCTMCExpression{Tv}, dvar::Symbol, cache::SymbolicCache) where Tv
    Q = symeval(f.args[1], cache)
    x = symeval(f.args[2], cache)
    r = symeval(f.args[3], cache)
    dQ = symeval(f.args[1], dvar, cache)
    dx = symeval(f.args[2], dvar, cache)
    dr = symeval(f.args[3], dvar, cache)
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
# symeval(f, dvar, cache)
# Return the second derivative of expr f
# """

function _eval(::Val{:tran}, f::SymbolicCTMCExpression{Tv}, dvar::Tuple{Symbol,Symbol}, cache::SymbolicCache) where Tv
    Q = symeval(f.args[1], cache)
    x = symeval(f.args[2], cache)
    r = symeval(f.args[3], cache)

    dQ_a = symeval(f.args[1], dvar[1], cache)
    dx_a = symeval(f.args[2], dvar[1], cache)
    dr_a = symeval(f.args[3], dvar[1], cache)

    dQ_b = symeval(f.args[1], dvar[2], cache)
    dx_b = symeval(f.args[2], dvar[2], cache)
    dr_b = symeval(f.args[3], dvar[2], cache)

    dQ_ab = symeval(f.args[1], dvar, cache)
    dx_ab = symeval(f.args[2], dvar, cache)
    dr_ab = symeval(f.args[3], dvar, cache)
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
