
struct Markov{Tv}
    state::Set{Symbol}
    initial::Dict{Symbol,Tv}
    trans::Dict{Tuple{Symbol,Symbol},Tv}
    reward::Dict{Symbol,Tv}
    ireward::Dict{Tuple{Symbol,Symbol},Tv}
end

function Markov(::Type{Tv} = Float64) where Tv
    Markov{Tv}(Set{Symbol}(), Dict{Symbol,Tv}(), Dict{Tuple{Symbol,Symbol},Tv}(), Dict{Symbol,Tv}(), Dict{Tuple{Symbol,Symbol},Tv}())
end

function trans!(m::Markov{Tv}, src::Symbol, dest::Symbol, tr::Tv) where Tv
    push!(m.state, src)
    push!(m.state, dest)
    m.trans[(src,dest)] = tr
end

function trans!(m::Markov{Tv}, src::Symbol, dest::Symbol, tr::Tx) where {Tv,Tx}
    push!(m.state, src)
    push!(m.state, dest)
    m.trans[(src,dest)] = convert(Tv, tr)
end

function reward!(m::Markov{Tv}, s::Symbol, r::Tv) where Tv
    push!(m.state, s)
    m.reward[s] = r
end

function reward!(m::Markov{Tv}, s::Symbol, r::Tx) where {Tv,Tx}
    push!(m.state, s)
    m.reward[s] = convert(Tv, r)
end

function reward!(m::Markov{Tv}, src::Symbol, dest::Symbol, r::Tv) where Tv
    push!(m.state, src)
    push!(m.state, dest)
    m.ireward[(src,dest)] = r
end

function reward!(m::Markov{Tv}, src::Symbol, dest::Symbol, r::Tx) where {Tv,Tx}
    push!(m.state, src)
    push!(m.state, dest)
    m.ireward[(src,dest)] = convert(Tv, r)
end

function initial!(m::Markov{Tv}, s::Symbol, p::Tv) where Tv
    push!(m.state, s)
    m.initial[s] = p
end

function initial!(m::Markov{Tv}, s::Symbol, p::Tx) where {Tv,Tx}
    push!(m.state, s)
    m.initial[s] = convert(Tv, p)
end

function generate(m::Markov{Tv}) where Tv
    states = [x for x = m.state]
    index = Dict([states[i] => i for i = 1:length(states)]...)
    initv = Tv[0 for _ = states]
    rwd = Tv[0 for _ = states]
    Q = spzeros(Tv, length(states), length(states))
    for ((src,dest),t) = m.trans
        Q[index[src],index[dest]] = t
        Q[index[src],index[src]] -= t
    end
    for (s,p) = m.initial
        initv[index[s]] = p
    end
    for (s,r) = m.reward
        rwd[index[s]] = r
    end
    initv, Q, rwd, states, index
end

macro parameters(params...)
    body = [Expr(:(=), esc(x), esc(Expr(:call, :symbolic, Expr(:quote, x)))) for x = params]
    push!(body, Expr(:tuple, [esc(x) for x = params]...))
    Expr(:block, body...)
end

macro transition(m, block)
    @assert Meta.isexpr(block, :block) "@transition should take a block (begin ... end)"
    body = []
    for x = block.args
        push!(body, _gentrans(x, m))
    end
    Expr(:block, body...)
end

function _gentrans(x::Any, m)
    x
end

function _gentrans(x::Expr, m)
    if Meta.isexpr(x, :tuple) && Meta.isexpr(x.args[1], :call) && x.args[1].args[1] == :(=>) && length(x.args) == 2
        src = x.args[1].args[2]
        dest = x.args[1].args[3]
        t = x.args[2]
        esc(:(trans!($m, $(Expr(:quote, src)), $(Expr(:quote, dest)), $t)))
    else
        x
    end
end

macro initial(m, block)
    @assert Meta.isexpr(block, :block) "@initial should take a block (begin ... end)"
    body = []
    for x = block.args
        push!(body, _geninitial(x, m))
    end
    Expr(:block, body...)
end

function _geninitial(x::Any, m)
    x
end

function _geninitial(x::Expr, m)
    if Meta.isexpr(x, :tuple) && typeof(x.args[1]) == Symbol && length(x.args) == 2
        s = x.args[1]
        p = x.args[2]
        esc(:(initial!($m, $(Expr(:quote, s)), $p)))
    else
        x
    end
end

macro reward(m, block)
    @assert Meta.isexpr(block, :block) "@reward should take a block (begin ... end)"
    body = []
    for x = block.args
        push!(body, _genreward(x, m))
    end
    Expr(:block, body...)
end

function _genreward(x::Any, m)
    x
end

function _genreward(x::Expr, m)
    if Meta.isexpr(x, :tuple) && typeof(x.args[1]) == Symbol && length(x.args) == 2
        s = x.args[1]
        r = x.args[2]
        esc(:(reward!($m, $(Expr(:quote, s)), $r)))
    elseif Meta.isexpr(x, :tuple) && Meta.isexpr(x.args[1], :call) && x.args[1].args[1] == :(=>) && length(x.args) == 2
        src = x.args[1].args[2]
        dest = x.args[1].args[3]
        r = x.args[2]
        esc(:(reward!($m, $(Expr(:quote, src)), $(Expr(:quote, dest)), $r)))
    else
        x
    end
end
