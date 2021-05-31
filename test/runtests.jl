using SymbolicMarkov
using SymbolicDiff
using SparseMatrix
using NMarkov
using LinearAlgebra
using Test

@testset "CTMCSt1" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    
    ma = gth(Q)
    
    x = 2.0
    @env test begin
        x = x
    end
    res = symboliceval(ma, test, SymbolicCache())
    ex = gth(Float64[-x x 0; 1 -2 1; 0 1 -1])
    @test isapprox(ex, res)
end

@testset "CTMCSt2" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = stgs(csc)
    
    x = 2.0
    @env test begin
        x = x
    end
    res = symboliceval(ma, test, SymbolicCache())
    ex = gth(Float64[-x x 0; 1 -2 1; 0 1 -1])
    @test isapprox(ex, res)
end

@testset "CTMCSt3" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    
    ma = dot(gth(Q), v)
    # println(ma)
    x = 2.0
    @env test begin
        x = x
    end
    res = symboliceval(ma, test, SymbolicCache())
    ex = sum(gth(Float64[-x x 0; 1 -2 1; 0 1 -1]) .* [1,0,0])
    println(res, ex)
    @test isapprox(ex, res)
end

@testset "CTMCStsen1" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = stgs(csc)
    
    x = 2.0
    @env test begin
        x = x
    end
    h = 0.0001
    @env test0 begin
        x = x + h
    end
    @env test1 begin
        x = x - h
    end
    
    vv0 = symboliceval(ma, test0, SymbolicCache())
    vv1 = symboliceval(ma, test1, SymbolicCache())
    ex = (vv0 - vv1) / (2*h)
    @test isapprox(symboliceval(ma, :x, test, SymbolicCache()), ex)
end

@testset "CTMCStsen2" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = dot(stgs(csc), v)
    
    x = 2.0
    @env test begin
        x = x
    end
    h = 0.0001
    @env test0 begin
        x = x + h
    end
    @env test1 begin
        x = x - h
    end
    
    vv0 = symboliceval(ma, test0, SymbolicCache())
    vv1 = symboliceval(ma, test1, SymbolicCache())
    ex = (vv0 - vv1) / (2*h)
    @test isapprox(symboliceval(ma, :x, test, SymbolicCache()), ex)
end

@testset "CTMCsensen1" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = stgs(csc)

    x = 2.0
    @env test begin
        x = x
    end
    h = 0.00001
    @env test0 begin
        x = x + h
    end
    @env test1 begin
        x = x - h
    end
    
    vv0 = symboliceval(ma, test0, SymbolicCache())
    vv1 = symboliceval(ma, test, SymbolicCache())
    vv2 = symboliceval(ma, test1, SymbolicCache())
    ex = (vv0 -2*vv1 + vv2) / (h^2)
    @test isapprox(symboliceval(ma, (:x, :x), test, SymbolicCache()), ex, atol = 1.0e-5)
end

@testset "CTMCsensen2" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = dot(stgs(csc), v)

    x = 2.0
    @env test begin
        x = x
    end
    h = 0.00001
    @env test0 begin
        x = x + h
    end
    @env test1 begin
        x = x - h
    end
    
    vv0 = symboliceval(ma, test0, SymbolicCache())
    vv1 = symboliceval(ma, test, SymbolicCache())
    vv2 = symboliceval(ma, test1, SymbolicCache())
    ex = (vv0 -2*vv1 + vv2) / (h^2)
    @test isapprox(symboliceval(ma, (:x, :x), test, SymbolicCache()), ex, atol = 1.0e-5)
end

@testset "CTMCTran1" begin
    Q = @expr [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x = 2.0
    y = 1.0
    z = 5.0
    @env test begin
        x = x
        y = y
        z = z
    end
    result = symboliceval(ma, test, SymbolicCache())
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "CTMCTran2" begin
    Q = @expr [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x = 2.0
    y = 1.0
    z = 5.0
    @env test begin
        x = x
        y = y
        z = z
    end
    result = symboliceval(ma, :x, test, SymbolicCache())
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "CTMCTran3" begin
    Q = @expr [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x = 2.0
    y = 1.0
    z = 5.0
    @env test begin
        x = x
        y = y
        z = z
    end
    result = symboliceval(ma, (:x,:y), test, SymbolicCache())
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "Markov1" begin
    m = Markov()
    @tr m begin
        up => down, 1.0
        down => up, 100.0
    end
    @init m begin
        up, 1.0
    end
    @reward m avail begin
        up, 1.0
    end
    Q, initv, rwd, = generate(m)
    println(initv)
    println(Q)
    println(rwd)
end

@testset "Markov2" begin
    @parameters lam1 lam2
    m = Markov()
    @tr m begin
        up => down, lam1
        down => up, lam2
    end
    @init m begin
        up, 1.0
    end
    @reward m avail begin
        up, 1
    end
    Q, initv, rwd, = generate(m)
    println(initv)
    println(Q)
    println(rwd)

    avail = dot(stgs(Q), rwd[:avail])

    @env test begin
        lam1 = 1.0
        lam2 = 100.0
    end

    a = symboliceval(avail, test, SymbolicCache())
    println(a)
    da1 = symboliceval(avail, :lam1, test, SymbolicCache())
    println(da1)
    da2 = symboliceval(avail, :lam2, test, SymbolicCache())
    println(da2)
    da12 = symboliceval(avail, (:lam1,:lam2), test, SymbolicCache())
    println(da12)
end

@testset "Markov3" begin
    @markov reliab(lam1, lam2) begin
        @tr begin
            up => down, lam1
            down => up, lam2
        end
        @init begin
            up, 1.0
        end
        @reward avail begin
            up, 1
        end
    end

    @parameters lam1 lam2
    m = reliab(lam1, lam2)
    println(ctmc)

    model = ctmc(m)
    avail = exrss(model, reward=:avail)

    @env test begin
        lam1 = 1.0
        lam2 = 100.0
    end

    a = symboliceval(avail, test, SymbolicCache())
    println(a)
    da1 = symboliceval(avail, :lam1, test, SymbolicCache())
    println(da1)
    da2 = symboliceval(avail, :lam2, test, SymbolicCache())
    println(da2)
    da12 = symboliceval(avail, (:lam1,:lam2), test, SymbolicCache())
    println(da12)
end

@testset "Markov4" begin
    @markov reliab(lam1, lam2) begin
        @tr up => down, lam1
        @tr down => up, lam2
        @init up, 1.0
        @reward avail up, 1
    end
    m = reliab(1, 100)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
end

@testset "Markov5" begin
    @markov reliab(lam1, lam2) begin
        @tr up => down, lam1
        @tr down => up, lam2
        @init up, 1.0
        @reward avail up, 1
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
        @tr up => down, lam1
        @tr down => up, lam2
        @init up, 1.0
        @reward avail up, 1
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
        @tr up => down, lam1
        @tr down => up, lam2
        @init up, 1.0
        @reward avail up, 1
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
        @tr up => down, lam1
        @tr down => up, lam2
        @init up, 1.0
        @reward avail up, 1
    end
    @parameters lam1 lam2
    m = reliab(lam1, lam2)
    println(m)

    avail = exrss(ctmc(m), reward=:avail)
    println(avail)
    
    tavail = exrt(LinRange(0, 1, 10), ctmc(m), reward=:avail)
    println(tavail)
    
    @env test begin
        lam1 = 1.0
        lam2 = 100.0
    end

    a = symboliceval(tavail, test, SymbolicCache())
    println(a)
    da1 = symboliceval(tavail, :lam1, test, SymbolicCache())
    println(da1)
    da2 = symboliceval(tavail, :lam2, test, SymbolicCache())
    println(da2)
    da12 = symboliceval(tavail, (:lam1,:lam2), test, SymbolicCache())
    println(da12)
end

@testset "Markov9" begin
    e = @macroexpand @markov mm1k(lam,mu,k) begin
        for i = 0:k-1
            @tr $(Symbol(i)) => $(Symbol(i+1)), lam
        end
        for i = 1:k
            @tr $(Symbol(i)) => $(Symbol(i-1)), mu
        end
        @init $(Symbol(0)), 1.0
        for i = 0:k
            @reward len $(Symbol(i)), i
        end
    end
    println(e)
end
