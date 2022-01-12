@testset "prob1" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @reward :r begin
            :up, 1
        end
    end

    begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    m = ctmc(midplane(lam, mu))
    println(sprob(m))
end

@testset "sprob2" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @reward :r begin
            :up, 1
        end
    end

    begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    m = ctmc(midplane(lam, mu), :DenseCTMC)
    println(sprob(m))
end

@testset "sprob2" begin
    @markov midplane(lam, mu) begin
        @tr begin
            :up => :down, lam
            :down => :up, mu
        end
        @reward :r begin
            :up, 1
        end
    end

    begin
        lam = 0.4
        mu = 1.5
        delta = 1.5
    end

    m = ctmc(midplane(lam, mu), :DenseCTMC)
    println(sprob(m, states=[:up]))
end

@testset "sprob4" begin
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

    m = ctmc(midplane(lam, mu), :DenseCTMC)
    println(seval(sprob(m)))
    println(seval(sprob(m), :lam))
    println(seval(sprob(m), (:lam, :lam)))
end

@testset "sprob5" begin
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

    m = ctmc(midplane(lam, mu), :DenseCTMC)
    println(seval(sprob(m, states=[:up])))
    println(seval(sprob(m, states=[:up]), :lam))
    println(seval(sprob(m, states=[:up]), (:lam, :lam)))
end

