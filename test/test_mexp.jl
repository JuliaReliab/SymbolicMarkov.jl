@testset "mexp1" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @init begin
            :up, 1
        end
    end

    begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    Q, x, _, states = generate(midplane(lam, mu))
    println(tprob(1.0, x, Q))
end

@testset "mexp2" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @init begin
            :up, 1
        end
    end

    @bind begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    Q, x, _, states = generate(midplane(lam, mu))
    println(seval(tprob(1.0, x, Q)))
end

@testset "mexp3" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @init begin
            :up, 1
        end
    end

    @bind begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    Q, x, _, states = generate(midplane(lam, mu))
    println(seval(tprob(1.0, x, Q), :lam))
end

@testset "mexp3" begin
    @markov midplane(lam, mu, p) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @init begin
            :up, p
            :down, 1-p
        end
    end

    @bind begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
        p = 1.0
    end

    Q, x, _, states = generate(midplane(lam, mu, p))
    println(seval(tprob(1.0, x, Q), :p))
    println(seval(tprob(1.0, x, Q), :lam))
    println(seval(tprob(1.0, x, Q), (:lam, :p)))
    println(seval(ctprob(100.0, x, Q), :p))
    println(seval(ctprob(100.0, x, Q), :lam))
    println(seval(ctprob(100.0, x, Q), (:lam, :p)))
end
