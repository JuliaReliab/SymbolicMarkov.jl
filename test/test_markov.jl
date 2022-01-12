@testset "Markov1" begin
    m = Markov()
    @tr m begin
        :up => :down, 1.0
        :down => :up, 100.0
    end
    @init m begin
        :up, 1.0
    end
    @reward m :avail begin
        :up, 1.0
    end
    Q, initv, rwd, = generate(m)
    println(initv)
    println(Q)
    println(rwd)
end

@testset "Markov1_1" begin
    m = Markov()
    @tr m begin
        :up => :down, 1.0
        :down => :up, 100.0
    end
    @init m begin
        :up, 1.0
    end
    @reward m :avail begin
        :up, 1.0
    end
    Q, initv, rwd, = generate(m, modeltype=:DenseCTMC)
    println(initv)
    println(Q)
    println(rwd)
end

@testset "Markov2" begin
    @vars lam1 lam2
    m = Markov()
    @tr m begin
        :up => :down, lam1
        :down => :up, lam2
    end
    @init m begin
        :up, 1.0
    end
    @reward m :avail begin
        :up, 1
    end
    Q, initv, rwd, = generate(m)
    println(initv)
    println(Q)
    println(rwd)

    avail = dot(prob(Q), rwd[:avail])

    @bind begin
        lam1 = 1.0
        lam2 = 100.0
    end

    a = seval(avail)
    println(a)
    da1 = seval(avail, :lam1)
    println(da1)
    da2 = seval(avail, :lam2)
    println(da2)
    da12 = seval(avail, (:lam1,:lam2))
    println(da12)
end

@testset "Markov3" begin
    @markov reliab(lam1, lam2) begin
        @tr begin
            :up => :down, lam1
            :down => :up, lam2
        end
        @init begin
            :up, 1.0
        end
        @reward :avail begin
            :up, 1
        end
    end

    @bind begin
        lam1 = 1.0
        lam2 = 100.0
    end
    m = reliab(lam1, lam2)
    println(ctmc)

    model = ctmc(m)
    avail = exrss(model, reward=:avail)

    a = seval(avail)
    println(a)
    da1 = seval(avail, :lam1)
    println(da1)
    da2 = seval(avail, :lam2)
    println(da2)
    da12 = seval(avail, (:lam1,:lam2))
    println(da12)
end

@testset "Markov4" begin
    @markov reliab(lam1, lam2) begin
        @tr :up => :down, lam1
        @tr :down => :up, lam2
        @init :up, 1.0
        @reward :avail :up, 1
    end
    m = reliab(1, 100)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
end

@testset "Markov5" begin
    @markov reliab(lam1, lam2) begin
        @tr :up => :down, lam1
        @tr :down => :up, lam2
        @init :up, 1.0
        @reward :avail :up, 1
    end
    m = reliab(1, 100)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
    
    tavail = exrt(LinRange(0, 1, 10), ctmc(m), reward=:avail)
    println(tavail)
end

@testset "Markov6" begin
    @markov reliab(lam1, lam2) begin
        @tr :up => :down, lam1
        @tr :down => :up, lam2
        @init :up, 1.0
        @reward :avail :up, 1
    end
    m = reliab(1, 100)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
    
    tavail = cexrt(LinRange(0, 1, 10), ctmc(m), reward=:avail)
    println(tavail)
end

@testset "Markov7" begin
    @markov reliab(lam1, lam2) begin
        @tr :up => :down, lam1
        @tr :down => :up, lam2
        @init :up, 1.0
        @reward :avail :up, 1
    end
    m = reliab(1, 100)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
    
    tavail = cexrt(0.5, ctmc(m), reward=:avail)
    println(tavail)
end

@testset "Markov8" begin
    @markov reliab(lam1, lam2) begin
        @tr :up => :down, lam1
        @tr :down => :up, lam2
        @init :up, 1.0
        @reward :avail :up, 1
    end

    @bind begin
        lam1 = 1.0
        lam2 = 100.0
    end

    m = reliab(lam1, lam2)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
    
    tavail = exrt(1.0, ctmc(m), reward=:avail)
    println(tavail)
    
    a = seval(tavail)
    println(a)
    da1 = seval(tavail, :lam1)
    println(da1)
    da2 = seval(tavail, :lam2)
    println(da2)
    da12 = seval(tavail, (:lam1,:lam2))
    println(da12)
end

@testset "Markov9" begin
    e = @macroexpand @markov mm1k(lam,mu,k) begin
        for i = 0:k-1
            @tr Symbol(i) => Symbol(i+1), lam
        end
        for i = 1:k
            @tr Symbol(i) => Symbol(i-1), mu
        end
        @init @s(0), 1.0
        for i = 0:k
            @reward :len Symbol(i), i
        end
    end
    println(e)
end

@testset "hybrid2" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @reward :r begin
            :up, 1
        end
    end

    @bind begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    m = ctmc(midplane(lam, mu))
    println(seval(prob(m.Q), lam))
    println(seval([lam, mu], delta))
    println(seval(prob(m.Q), delta))
    println(seval(prob(m.Q), (lam, delta)))
end