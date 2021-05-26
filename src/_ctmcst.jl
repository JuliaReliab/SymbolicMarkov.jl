"""
Stationary Markov
"""

struct SymbolicCTMCExpression{Tv} <: AbstractSymbolic{Tv}
    params::Set{Symbol}
    op::Symbol
    args::Vector{<:AbstractSymbolic}
    options::Dict{Symbol,Any}
end

function gth(Q::SymbolicMatrix{Tv}) where Tv
    SymbolicCTMCExpression{Tv}(Q.params, :gth, [Q], Dict{Symbol,Any}())
end

function stgs(Q::SymbolicCSCMatrix{Tv}; maxiter=5000, steps=20, rtol::Tv=Tv(1.0e-6)) where Tv
    SymbolicCTMCExpression{Tv}(Q.params, :stgs, [Q], Dict{Symbol,Any}(:maxiter=>maxiter, :steps=>steps, :rtol=>rtol))
end

# function ctmcst(Q::AbstractSymbolicMatrix{Tv}, r::AbstractSymbolicVector{Tv}) where Tv
#     SymbolicExpression{Tv}(Q.params, :ctmcst2, [Q,r])
# end

"""
symboliceval(f, env, cache)
Return the value for expr f
"""

function _eval(::Val{:gth}, f::SymbolicCTMCExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    Q = symboliceval(f.args[1], env, cache)
    gth(Q)
end

function _eval(::Val{:stgs}, f::SymbolicCTMCExpression{Tv}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    Q = symboliceval(f.args[1], env, cache)
    x, conv, iter, rerror = stgs(Q, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

"""
symboliceval(f, dvar, env, cache)
Return the first derivative of expr f
"""

function _eval(::Val{:stgs}, f::SymbolicCTMCExpression{Tv}, dvar::Symbol, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = symboliceval(f, env, cache)
    Q = symboliceval(f.args[1], env, cache)
    dQ = symboliceval(f.args[1], dvar, env, cache)
    s, conv, iter, rerror = stsengs(Q, pis, dQ' * pis, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GSsen did not converge", iter, rerror
    end
    s
end

"""
symboliceval(f, dvar, env, cache)
Return the second derivative of expr f
"""

function _eval(::Val{:stgs}, f::SymbolicCTMCExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
    pis = symboliceval(f, env, cache)
    dpis_a = symboliceval(f, dvar[1], env, cache)
    dpis_b = symboliceval(f, dvar[1], env, cache)

    Q = symboliceval(f.args[1], env, cache)
    dQ_a = symboliceval(f.args[1], dvar[1], env, cache)
    dQ_b = symboliceval(f.args[1], dvar[2], env, cache)
    dQ_ab = symboliceval(f.args[1], dvar, env, cache)

    s, conv, iter, rerror = stsengs(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a, maxiter=f.options[:maxiter], steps=f.options[:steps], rtol=f.options[:rtol])
    if conv == false
        @warn "GSsen2 did not converge", iter, rerror
    end
    s
end

# function _eval(::Val{:ctmcst}, f::SymbolicExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Vector{Tv} where Tv
#     Q = symboliceval(f.args[1], env, cache)
#     dQ_a = symboliceval(f.args[1], dvar[1], env, cache)
#     dQ_b = symboliceval(f.args[1], dvar[2], env, cache)
#     dQ_ab = symboliceval(f.args[1], dvar, env, cache)

#     pis = _stsolve(Q)
#     dpis_a = _stsensolve(Q, pis, dQ_a' * pis)
#     dpis_b = _stsensolve(Q, pis, dQ_b' * pis)
#     dpis_ab = _stsensolve(Q, pis, dQ_ab' * pis + dQ_a' * dpis_b + dQ_b' * dpis_a)
#     dpis_ab
# end

# function _eval(::Val{:ctmcst2}, f::SymbolicExpression{Tv}, dvar::Tuple{Symbol,Symbol}, env::SymbolicEnv, cache::SymbolicCache)::Tv where Tv
#     Q = symboliceval(f.args[1], env, cache)
#     dQ_a = symboliceval(f.args[1], dvar[1], env, cache)
#     dQ_b = symboliceval(f.args[1], dvar[2], env, cache)
#     dQ_ab = symboliceval(f.args[1], dvar, env, cache)
#     r = symboliceval(f.args[2], env, cache)
#     dr_a = symboliceval(f.args[2], dvar[1], env, cache)
#     dr_b = symboliceval(f.args[2], dvar[2], env, cache)
#     dr_ab = symboliceval(f.args[2], dvar, env, cache)

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
