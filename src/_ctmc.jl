"""
ctmc type
"""

mutable struct CTMCModel{Tv}
    Q::AbstractMatrix{Tv}
    initv::Vector{Tv}
    reward::Dict{Symbol,Vector{Tv}}
    states::Vector{Symbol}
end

function Base.show(io::IO, x::CTMCModel{Tv}) where Tv
    Base.show(io, "CTMCModel($(objectid(x)))")
end

function ctmc(Q::AbstractMatrix{Tv}, initv::Vector{Tv}, r::Dict{Symbol,Vector{Tv}}, states::Vector{Symbol}) where Tv
    CTMCModel{Tv}(Q, initv, r, states)
end

function ctmc(m::Markov, modeltype=:SparseCTMC)
    Q, initv, r, states = generate(m, modeltype=modeltype)
    ctmc(Q, initv, r, states)
end

"""
prob
"""

function prob(m::CTMCModel{Tv}; states = m.states, maxiter=5000, steps=20, rtol=1.0e-6) where Tv
    s = [x in states for x = m.states]
    _prob(m.Q, maxiter, steps, rtol)[s]
end

function _prob(Q::Matrix{Tv}, maxiter, steps, rtol) where Tv
    gth(Q)
end

function _prob(Q::SparseCSC{Tv,Ti}, maxiter, steps, rtol) where {Tv<:Number,Ti}
    x, conv, iter, rerror = stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

function _prob(Q::SparseMatrixCSC{Tv,Ti}, maxiter, steps, rtol) where {Tv<:Number,Ti}
    x, conv, iter, rerror = stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    x
end

function _prob(Q::SparseCSC{<:AbstractSymbolic{Tv},Ti}, maxiter, steps, rtol) where {Tv,Ti}
    stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
end

function _prob(Q::SparseMatrixCSC{<:AbstractSymbolic{Tv},Ti}, maxiter, steps, rtol) where {Tv,Ti}
    stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
end

"""
exrss
"""

function exrss(m::CTMCModel{Tv}; reward, maxiter=5000, steps=20, rtol=1.0e-6) where {Tv}
    _exrss(m.Q, m.reward[reward], maxiter, steps, rtol)
end

function _exrss(Q::Matrix{Tv}, r::Vector{Tv}, maxiter, steps, rtol) where Tv
    dot(gth(Q), r)
end

function _exrss(Q::SparseCSC{Tv,Ti}, r::Vector{Tv}, maxiter, steps, rtol) where {Tv<:Number,Ti}
    x, conv, iter, rerror = stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    dot(x, r)
end

function _exrss(Q::SparseMatrixCSC{Tv,Ti}, r::Vector{Tv}, maxiter, steps, rtol) where {Tv<:Number,Ti}
    x, conv, iter, rerror = stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol)
    if conv == false
        @warn "GS did not converge", iter, rerror
    end
    dot(x, r)
end

function _exrss(Q::SparseCSC{<:AbstractSymbolic{Tv},Ti}, r::Vector{<:AbstractSymbolic{Tv}}, maxiter, steps, rtol) where {Tv,Ti}
    dot(stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol), r)
end

function _exrss(Q::SparseMatrixCSC{<:AbstractSymbolic{Tv},Ti}, r::Vector{<:AbstractSymbolic{Tv}}, maxiter, steps, rtol) where {Tv,Ti}
    dot(stgs(Q, maxiter=maxiter, steps=steps, rtol=rtol), r)
end

"""
exrt, cexrt
"""

function exrt(ts::AbstractVector{<:Number}, m::CTMCModel{Tv}; reward, forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv}
    _exrt(m.Q, m.initv, m.reward[reward], ts, forward, ufact, eps, rmax)
end

function cexrt(ts::AbstractVector{<:Number}, m::CTMCModel{Tv}; reward, forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv}
    _cexrt(m.Q, m.initv, m.reward[reward], ts, forward, ufact, eps, rmax)
end

function exrt(ts::Number, m::CTMCModel{Tv}; reward, forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv}
    _exrt(m.Q, m.initv, m.reward[reward], [ts], forward, ufact, eps, rmax)[1]
end

function cexrt(ts::Number, m::CTMCModel{Tv}; reward, forward=:T, ufact=1.01, eps=1.0e-8, rmax=500) where {Tv}
    _cexrt(m.Q, m.initv, m.reward[reward], [ts], forward, ufact, eps, rmax)[1]
end

function _exrt(Q::AbstractMatrix{Tv}, x::AbstractVector{Tv}, r::AbstractVector{Tv}, ts::AbstractVector{<:Number}, forward, ufact, eps, rmax) where {Tv<:Number}
    result, _, = tran(Q, x, r, ts, forward=forward, ufact=ufact, eps=eps, rmax=rmax)
    result
end

function _cexrt(Q::AbstractMatrix{Tv}, x::AbstractVector{Tv}, r::AbstractVector{Tv}, ts::AbstractVector{<:Number}, forward, ufact, eps, rmax) where {Tv<:Number}
    _, cresult, = tran(Q, x, r, ts, forward=forward, ufact=ufact, eps=eps, rmax=rmax)
    cresult
end

function _exrt(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, r::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{<:Number}, forward, ufact, eps, rmax) where {Tv}
    tran(Q, x, r, ts, forward=forward, ufact=ufact, eps=eps, rmax=rmax, cumulative=false)
end

function _cexrt(Q::AbstractMatrix{<:AbstractSymbolic{Tv}}, x::AbstractVector{<:AbstractSymbolic{Tv}}, r::AbstractVector{<:AbstractSymbolic{Tv}}, ts::AbstractVector{<:Number}, forward, ufact, eps, rmax) where {Tv}
    tran(Q, x, r, ts, forward=forward, ufact=ufact, eps=eps, rmax=rmax, cumulative=true)
end

