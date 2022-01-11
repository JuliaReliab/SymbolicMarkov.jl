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
    println(prob(m))
end

@testset "prob2" begin
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
    println(prob(m))
end

@testset "prob2" begin
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
    println(prob(m, states=[:up]))
end

@testset "prob4" begin
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
    println(seval(prob(m)))
    println(seval(prob(m), :lam))
    println(seval(prob(m), (:lam, :lam)))
end

@testset "prob5" begin
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
    println(seval(prob(m, states=[:up])))
    println(seval(prob(m, states=[:up]), :lam))
    println(seval(prob(m, states=[:up]), (:lam, :lam)))
end

