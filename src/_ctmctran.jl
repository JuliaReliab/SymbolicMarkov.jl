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
    xx = [x..., zeros(Tv, n)...]
    rr = [dr..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
    
    if !iszero(dx)
        tmp, ctmp, = tran(Q, dx, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
        result .+= tmp
        cresult .+= ctmp
    end

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
    xx = [x..., zeros(Tv, 3*n)...]
    rr = [dr_ab..., dr_b..., dr_a..., r...]
    result, cresult, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])

    if !iszero(dx_ab)
        tmp, ctmp, = tran(Q, dx_ab, r, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
        result .+= tmp
        cresult .+= ctmp
    end

    if !iszero(dx_a)
        QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (2,2,Q), (1,2,dQ_b)]))
        xx = [dx_a..., zeros(Tv, n)...]
        rr = [dr_b..., r...]
        tmp, ctmp, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
        result .+= tmp
        cresult .+= ctmp
    end

    if !iszero(dx_b)
        QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (2,2,Q), (1,2,dQ_a)]))
        xx = [dx_b..., zeros(Tv, n)...]
        rr = [dr_a..., r...]
        tmp, ctmp, = tran(QQ, xx, rr, ts, forward=f.options[:forward], ufact=f.options[:ufact], eps=f.options[:eps], rmax=f.options[:rmax])
        result .+= tmp
        cresult .+= ctmp
    end

    if f.options[:cumulative]
        cresult
    else
        result
    end
end

# function _eval(::Val{:ctmctran}, f::SymbolicExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
#     Q = symboliceval(f.args[1], env, cache)
#     dQ_a = symboliceval(f.args[1], dvar[1], env, cache)
#     dQ_b = symboliceval(f.args[1], dvar[2], env, cache)
#     dQ_ab = symboliceval(f.args[1], dvar, env, cache)
#     x0 = symboliceval(f.args[2], env, cache)
#     dx0_a = symboliceval(f.args[2], dvar[1], env, cache)
#     dx0_b = symboliceval(f.args[2], dvar[2], env, cache)
#     dx0_ab = symboliceval(f.args[2], dvar, env, cache)
#     t = symboliceval(f.args[3], env, cache)
#     dt_a = symboliceval(f.args[3], dvar[1], env, cache)
#     dt_b = symboliceval(f.args[3], dvar[2], env, cache)
#     dt_ab = symboliceval(f.args[3], dvar, env, cache)
#     transpose = symboliceval(f.args[4], env, cache)
    
#     (iszero(dt_a) && iszero(dt_b)) || throw(ErrorException("The derivative of t is not applicable yet."))

#     p1_ab = _mexp(Q, dx0_ab, t, transpose)
#     p1_a, res1_a = _mexpsen(Q, dQ_b, dx0_a, t, transpose)
#     p1_b, res1_b = _mexpsen(Q, dQ_a, dx0_b, t, transpose)
#     p2, res2_a, res2_b, res2_ab = _mexpsen(Q, dQ_a, dQ_b, dQ_ab, x0, t, transpose)
#     p1_ab + res1_a + res1_b + res2_ab
# end

# function _eval(::Val{:ctmctran2}, f::SymbolicExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Tv where Tv
#     Q = symboliceval(f.args[1], env, cache)
#     dQ_a = symboliceval(f.args[1], dvar[1], env, cache)
#     dQ_b = symboliceval(f.args[1], dvar[2], env, cache)
#     dQ_ab = symboliceval(f.args[1], dvar, env, cache)
#     x0 = symboliceval(f.args[2], env, cache)
#     dx0_a = symboliceval(f.args[2], dvar[1], env, cache)
#     dx0_b = symboliceval(f.args[2], dvar[2], env, cache)
#     dx0_ab = symboliceval(f.args[2], dvar, env, cache)
#     r = symboliceval(f.args[3], env, cache)
#     dr_a = symboliceval(f.args[3], dvar[1], env, cache)
#     dr_b = symboliceval(f.args[3], dvar[2], env, cache)
#     dr_ab = symboliceval(f.args[3], dvar, env, cache)
#     t = symboliceval(f.args[4], env, cache)
#     dt_a = symboliceval(f.args[4], dvar[1], env, cache)
#     dt_b = symboliceval(f.args[4], dvar[2], env, cache)
#     dt_ab = symboliceval(f.args[4], dvar, env, cache)
#     transpose = symboliceval(f.args[5], env, cache)
    
#     (iszero(dt_a) && iszero(dt_b)) || throw(ErrorException("The derivative of t is not applicable yet."))

#     p1_ab = _mexp(Q, dx0_ab, t, transpose)
#     p1_a, res1_a = _mexpsen(Q, dQ_b, dx0_a, t, transpose)
#     p1_b, res1_b = _mexpsen(Q, dQ_a, dx0_b, t, transpose)
#     p2, res2_a, res2_b, res2_ab = _mexpsen(Q, dQ_a, dQ_b, dQ_ab, x0, t, transpose)
#     x_ab = p1_ab + res1_a + res1_b + res2_ab
#     x_a = p1_a + res2_a
#     x_b = p1_b + res2_b
#     @dot(x_ab, r) + @dot(x_b, dr_a) + @dot(x_a, dr_b) + @dot(p2, dr_ab)
# end

# """
# _mexp
# Solve the transient vector
# """

# function _mexp(Q::AbstractMatrix{Tv}, x0::Vector{Tv}, t::Tv, transpose::Bool) where Tv
#     n = length(x0)
#     iszero(x0) && return zeros(Tv,n)
#     (iszero(t) || iszero(Q)) && return x0
#     return mexp(Q, x0, t, transpose=transpose ? Trans() : NoTrans(),
#         ufact=SymbolicMarkovConfig[:ufact], eps=SymbolicMarkovConfig[:eps], rmax=SymbolicMarkovConfig[:rmax])
# end

# function _mexpsen(Q::AbstractMatrix{Tv}, dQ::AbstractMatrix{Tv}, x0::Vector{Tv}, t::Tv, transpose::Bool) where Tv
#     n = length(x0)
#     iszero(x0) && return zeros(Tv,n), zeros(Tv,n)
#     (iszero(t) || iszero(Q)) && return x0, zeros(Tv,n)
#     iszero(dQ) && return _mexp(Q, x0, t), zeros(Tv,n)
#     QQ = SparseCSC(BlockCOO(2, 2, [(1,1,Q), (1,2,dQ), (2,2,Q)]))
#     xx = [x0..., zeros(Tv, n)...]
#     res = _mexp(QQ, xx, t, transpose)
#     return res[1:n], res[n+1:n+n]
# end

# function _mexpsen(Q::AbstractMatrix{Tv}, dQ_a::AbstractMatrix{Tv}, dQ_b::AbstractMatrix{Tv}, dQ_ab::AbstractMatrix{Tv}, x0::Vector{Tv}, t::Tv, transpose::Bool) where Tv
#     n = length(x0)
#     iszero(x0) && return zeros(Tv,n), zeros(Tv,n), zeros(Tv,n), zeros(Tv,n)
#     (iszero(t) || iszero(Q)) && return x0, zeros(Tv,n), zeros(Tv,n), zeros(Tv,n)
#     (iszero(dQ_a) && iszero(dQ_b)) && return _mexp(Q, x0, t), zeros(Tv,n), zeros(Tv,n), zeros(Tv,n)
#     QQ = SparseCSC(BlockCOO(4, 4, [(1,1,Q), (2,2,Q), (3,3,Q), (4,4,Q), (1,2,dQ_a), (1,3,dQ_b), (1,4,dQ_ab), (2,4,dQ_b), (3,4,dQ_a)]))
#     xx = [x0..., zeros(Tv, 3*n)...]
#     res = _mexp(QQ, xx, t, transpose)
#     return res[1:n], res[n+1:2*n], res[2*n+1:3*n], res[3*n+1:4*n]
# end
