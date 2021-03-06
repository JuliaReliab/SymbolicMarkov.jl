@testset "CTMCSt1" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    
    ma = prob(Q)
    
    x = 2.0
    ex = prob(Float64[-x x 0; 1 -2 1; 0 1 -1])
    @bind x = x
    res = seval(ma)
    @test isapprox(ex, res)
end

@testset "CTMCSt2" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = prob(csc)
    
    x = 2.0
    ex = prob(Float64[-x x 0; 1 -2 1; 0 1 -1])
    @bind x = x
    res = seval(ma)
    @test isapprox(ex, res)
end

@testset "CTMCSt3" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    
    ma = dot(prob(Q), v)
    println(ma)
    x = 2.0
    ex = sum(prob(Float64[-x x 0; 1 -2 1; 0 1 -1]) .* [1,0,0])
    @bind x = x
    res = seval(ma)
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
    ma = prob(csc)
    
    x0 = 2.0
    test = SymbolicEnv()
    @bind test x = x0
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 x = x0 + h
    test1 = SymbolicEnv()
    @bind test1 x = x0 - h
    
    vv0 = seval(ma, test0)
    vv1 = seval(ma, test1)
    ex = (vv0 - vv1) / (2*h)
    @test isapprox(seval(ma, :x, test), ex)
end

@testset "CTMCStsen2" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = dot(prob(csc), v)
    
    x0 = 2.0
    test = SymbolicEnv()
    @bind test x = x0
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 x = x0 + h
    test1 = SymbolicEnv()
    @bind test1 x = x0 - h
    
    vv0 = seval(ma, test0)
    vv1 = seval(ma, test1)
    ex = (vv0 - vv1) / (2*h)
    @test isapprox(seval(ma, :x, test), ex)
end

@testset "CTMCsensen1" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = prob(csc)

    x0 = 2.0
    test = SymbolicEnv()
    @bind test x = x0
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 x = x0 + h
    test1 = SymbolicEnv()
    @bind test1 x = x0 - h
    
    vv0 = seval(ma, test0)
    vv1 = seval(ma, test)
    vv2 = seval(ma, test1)
    ex = (vv0 -2*vv1 + vv2) / (h^2)
    @test isapprox(seval(ma, (:x, :x), test), ex, atol = 1.0e-5)
end

@testset "CTMCsensen2" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = dot(prob(csc), v)

    x0 = 2.0
    test = SymbolicEnv()
    @bind test x = x0
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 x = x0 + h
    test1 = SymbolicEnv()
    @bind test1 x = x0 - h
    
    vv0 = seval(ma, test0)
    vv1 = seval(ma, test)
    vv2 = seval(ma, test1)
    ex = (vv0 -2*vv1 + vv2) / (h^2)
    @test isapprox(seval(ma, (:x, :x), test), ex, atol = 1.0e-5)
end

@testset "CTMCStsen3" begin
    Q = @expr [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = Q
    ma = dot(prob(csc), v)
    
    x0 = 2.0
    test = SymbolicEnv()
    @bind test x = x0
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 x = x0 + h
    test1 = SymbolicEnv()
    @bind test1 x = x0 - h
    
    vv0 = seval(ma, test0)
    vv1 = seval(ma, test1)
    ex = (vv0 - vv1) / (2*h)
    @test isapprox(seval(ma, :x, test), ex)
end