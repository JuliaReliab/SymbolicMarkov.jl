
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
    println(m)
    Q, initv, rwd, = generate(m)
    println(initv)
    println(Q)
    println(rwd)

    avail = dot(stgs(Q), rwd[:avail])

    lam1 => 1.0
    lam2 => 100.0

    a = symeval(avail, SymbolicCache())
    println(a)
    da1 = symeval(avail, :lam1, SymbolicCache())
    println(da1)
    da2 = symeval(avail, :lam2, SymbolicCache())
    println(da2)
    da12 = symeval(avail, (:lam1,:lam2), SymbolicCache())
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

    @vars lam1 lam2
    m = reliab(lam1, lam2)
    println(ctmc)

    model = ctmc(m)
    avail = exrss(model, reward=:avail)

    lam1 => 1.0
    lam2 => 100.0

    a = symeval(avail, SymbolicCache())
    println(a)
    da1 = symeval(avail, :lam1, SymbolicCache())
    println(da1)
    da2 = symeval(avail, :lam2, SymbolicCache())
    println(da2)
    da12 = symeval(avail, (:lam1,:lam2), SymbolicCache())
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
    @vars lam1 lam2
    m = reliab(lam1, lam2)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
    
    tavail = exrt(LinRange(0, 1, 10), ctmc(m), reward=:avail)
    println(tavail)
    
    lam1 => 1.0
    lam2 => 100.0

    a = symeval(tavail, SymbolicCache())
    println(a)
    da1 = symeval(tavail, :lam1, SymbolicCache())
    println(da1)
    da2 = symeval(tavail, :lam2, SymbolicCache())
    println(da2)
    da12 = symeval(tavail, (:lam1,:lam2), SymbolicCache())
    println(da12)
end

@testset "Markov9" begin
    e = @macroexpand @markov mm1k(lam,mu,k) begin
        for i = 0:k-1
            @tr Symbol(:s, i) => Symbol(:s, i+1), lam
        end
        for i = 1:k
            @tr Symbol(:s, i) => Symbol(:s, i-1), mu
        end
        @init Symbol(:s, 0), 1.0
        for i = 0:k
            @reward len Symbol(:s, i), i
        end
    end
    println(e)
    eval(e)
end
