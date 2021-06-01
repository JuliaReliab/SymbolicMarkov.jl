"""
ctmcst
"""

# struct CTMCModel{MatrixT,Tv}
#     Q::MatrixT{Tv}
#     initv::Vector{Tv}
#     reward::Dict{Symbol,Vector{Tv}}
#     states::Vector{Symbol}
#     index::Dict{Symbol,Ti}
# end

# function exrss(m::CTMCModel{MatrixT,Tv}; reward = :default) where {MatrixT,Tv}
# end


"""
macro
"""

mutable struct Markov
    Tv::DataType
    state::Set{Symbol}
    initial::Dict{Symbol,Any}
    trans::Dict{Tuple{Symbol,Symbol},Any}
    reward::Dict{Symbol,Dict{Symbol,Any}}
    ireward::Dict{Symbol,Dict{Tuple{Symbol,Symbol},Any}}
end

function Markov()
    Markov(Float64, Set{Symbol}(), Dict{Symbol,Any}(), Dict{Tuple{Symbol,Symbol},Any}(),
        Dict{Symbol,Dict{Symbol,Any}}(), Dict{Symbol,Dict{Tuple{Symbol,Symbol},Any}}())
end

function trans!(m::Markov, src::Symbol, dest::Symbol, tr)
    push!(m.state, src)
    push!(m.state, dest)
    m.Tv = promote_type(m.Tv, typeof(tr))
    m.trans[(src,dest)] = tr
end

function reward!(m::Markov, label::Symbol, s::Symbol, r)
    push!(m.state, s)
    d = get(m.reward, label) do
        m.reward[label] = Dict{Symbol,Any}()
    end
    m.Tv = promote_type(m.Tv, typeof(r))
    d[s] = r
end

function reward!(m::Markov, label, src, dest, r)
    push!(m.state, src)
    push!(m.state, dest)
    d = get(m.ireward, label) do
        m.ireward[label] = Dict{Tuple{Symbol,Symbol},Any}()
    end
    m.Tv = promote_type(m.Tv, typeof(r))
    d[(src,dest)] = r
end

function initial!(m::Markov, s::Symbol, p)
    push!(m.state, s)
    m.Tv = promote_type(m.Tv, typeof(p))
    m.initial[s] = p
end

function generate(m; modeltype::Symbol = :CTMC)
    _generate(Val(modeltype), m)
end

function _generate(::Val{:CTMC}, m::Markov)
    states = [x for x = m.state]
    index = Dict([states[i] => i for i = 1:length(states)]...)
    Q = spzeros(m.Tv, length(states), length(states))
    for ((src,dest),t) = m.trans
        Q[index[src],index[dest]] = t
        Q[index[src],index[src]] -= t
    end
    initv = zeros(m.Tv, length(states))
    for (s,p) = m.initial
        initv[index[s]] = p
    end
    rwd = Dict{Symbol,Vector{m.Tv}}()
    for (k,v) = m.reward
        rwdv = zeros(m.Tv, length(states))
        for (s,r) = v
            rwdv[index[s]] = r
        end
        rwd[k] = rwdv
    end
    Q, initv, rwd, states
end

macro tr(m, block)
    if Meta.isexpr(block, :block)
        body = [_gentrans(x, m) for x = block.args]
        esc(Expr(:block, body...))
    else
        esc(_gentrans(block, m))
    end
end

function _gentrans(x::Any, m)
    x
end

function _gentrans(x::Expr, m)
    if Meta.isexpr(x, :tuple) && Meta.isexpr(x.args[1], :call) && x.args[1].args[1] == :(=>) && length(x.args) == 2
        src = x.args[1].args[2]
        dest = x.args[1].args[3]
        t = x.args[2]
        :(trans!($m, $src, $dest, $t))
    else
        throw(TypeError(x, "Invalid format for the transition"))
    end
end

macro init(m, block)
    if Meta.isexpr(block, :block)
        body = [_geninitial(x, m) for x = block.args]
        esc(Expr(:block, body...))
    else
        esc(_geninitial(block, m))
    end
end

function _geninitial(x::Any, m)
    x
end

function _geninitial(x::Expr, m)
    if Meta.isexpr(x, :tuple) && length(x.args) == 2
        s = x.args[1]
        p = x.args[2]
        :(initial!($m, $s, $p))
    else
        throw(TypeError(x, "Invalid format for the initial probability"))
    end
end

macro reward(m, label, block)
    if Meta.isexpr(block, :block)
        body = [_genreward(x, label, m) for x = block.args]
        esc(Expr(:block, body...))
    else
        esc(_genreward(block, label, m))
    end
end

function _genreward(x::Any, label, m)
    x
end

function _genreward(x::Expr, label, m)
    if Meta.isexpr(x, :tuple) && length(x.args) == 2
        s = x.args[1]
        r = x.args[2]
        :(reward!($m, $label, $s, $r))
    elseif Meta.isexpr(x, :tuple) && Meta.isexpr(x.args[1], :call) && x.args[1].args[1] == :(=>) && length(x.args) == 2
        src = x.args[1].args[2]
        dest = x.args[1].args[3]
        r = x.args[2]
        :(reward!($m, $label, $src, $dest, $r))
    else
        throw(TypeError(x, "Invalid format for the reward vector"))
    end
end

###

macro markov(f, block)
    body = []
    push!(body, Expr(:(=), :tmp, Expr(:call, :Markov)))
    if Meta.isexpr(block, :block)
        for x = block.args
            push!(body, _replace_macro(x))
        end
    end
    push!(body, :tmp)
    esc(Expr(:function, f, Expr(:block, body...)))
end

function _replace_macro(x::Any)
    x
end

function _replace_macro(x::Expr)
    if Meta.isexpr(x, :macrocall) && (x.args[1] == Symbol("@tr") || x.args[1] == Symbol("@init") || x.args[1] == Symbol("@reward"))
        Expr(:macrocall, x.args[1], x.args[2], :tmp, [_replace_macro(u) for u = x.args[3:end]]...)
    else
        Expr(x.head, [_replace_macro(u) for u = x.args]...)
    end
end

