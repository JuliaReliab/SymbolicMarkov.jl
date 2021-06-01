@testset "CTMCSt1" begin
    Q = @expr [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    
    ma = gth(Q)
    
    x = 2.0
    @bind :x => x
    res = seval(ma)
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
    @bind :x => x
    res = seval(ma)
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
    @bind :x => x
    res = seval(ma)
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
    test = SymbolicEnv()
    @bind test :x => x
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 :x => x + h
    test1 = SymbolicEnv()
    @bind test1 :x => x - h
    
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
    ma = dot(stgs(csc), v)
    
    x = 2.0
    test = SymbolicEnv()
    @bind test :x => x
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 :x => x + h
    test1 = SymbolicEnv()
    @bind test1 :x => x - h
    
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
    ma = stgs(csc)

    x = 2.0
    test = SymbolicEnv()
    @bind test :x => x
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 :x => x + h
    test1 = SymbolicEnv()
    @bind test1 :x => x - h
    
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
    ma = dot(stgs(csc), v)

    x = 2.0
    test = SymbolicEnv()
    @bind test :x => x
    h = 0.0001
    
    test0 = SymbolicEnv()
    @bind test0 :x => x + h
    test1 = SymbolicEnv()
    @bind test1 :x => x - h
    
    vv0 = seval(ma, test0)
    vv1 = seval(ma, test)
    vv2 = seval(ma, test1)
    ex = (vv0 -2*vv1 + vv2) / (h^2)
    @test isapprox(seval(ma, (:x, :x), test), ex, atol = 1.0e-5)
end
