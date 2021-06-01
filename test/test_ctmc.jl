
@testset "CTMCSt1" begin
    @vars x
    Q = [-x x 0;
        1 -2 1;
        0 1 -1]
    v = [1, 0, 0]
    
    ma = gth(Q)
    
    x => 2.0
    res = symeval(ma, SymbolicCache())
    ex = gth(symeval([-x x 0; 1 -2 1; 0 1 -1]))
    @test isapprox(ex, res)
end

@testset "CTMCSt2" begin
    @vars x
    Q = [-x x 0;
        1 -2 1;
        0 1 -1]
    v = [1, 0, 0]
    csc = SparseCSC(Q)
    ma = stgs(csc)
    
    x => 2.0
    res = symeval(ma, SymbolicCache())
    ex = gth(symeval([-x x 0; 1 -2 1; 0 1 -1]))
    @test isapprox(ex, res)
end

@testset "CTMCSt3" begin
    @vars x
    Q = [-x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    
    ma = dot(gth(Q), v) # TODO: v should be promoted
    # println(ma)
    x => 2.0
    res = symeval(ma, SymbolicCache())
    ex = sum(gth(symeval([-x x 0; 1 -2 1; 0 1 -1])) .* [1,0,0])
    println(res, ex)
    @test isapprox(ex, res)
end

@testset "CTMCStsen1" begin
    @vars x
    Q = [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = stgs(csc)
    
    xvalue = 2.0
    h = 0.0001
    x => xvalue + h
    vv0 = symeval(ma, SymbolicCache())
    x => xvalue - h
    vv1 = symeval(ma, SymbolicCache())
    ex = (vv0 - vv1) / (2*h)

    x => xvalue
    @test isapprox(symeval(ma, :x, SymbolicCache()), ex)
end

@testset "CTMCStsen2" begin
    @vars x
    Q = [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = dot(stgs(csc), v)
    
    xvalue = 2.0
    h = 0.0001
    x => xvalue + h
    vv0 = symeval(ma, SymbolicCache())
    x => xvalue - h
    vv1 = symeval(ma, SymbolicCache())
    ex = (vv0 - vv1) / (2*h)

    x => xvalue
    @test isapprox(symeval(ma, :x, SymbolicCache()), ex)
end

@testset "CTMCsensen3" begin
    @vars x
    Q = [
        -x x 0;
        1 -2 1;
        0 1 -1]
    v = @expr [1, 0, 0]
    csc = SparseCSC(Q)
    ma = stgs(csc)

    xvalue = 2.0
    h = 0.00001

    x => xvalue + h    
    vv0 = symeval(ma, SymbolicCache())
    x => xvalue
    vv1 = symeval(ma, SymbolicCache())
    x => xvalue - h
    vv2 = symeval(ma, SymbolicCache())
    ex = (vv0 -2*vv1 + vv2) / (h^2)
    @test isapprox(symeval(ma, (:x, :x), SymbolicCache()), ex, atol = 1.0e-5)
end

@testset "CTMCTran1" begin
    @vars x y z
    Q = [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    xv = @expr [1.0, 0, 0]
    rv = @expr [1.0, 1, 0]
    ma = tran(Q, xv, rv, LinRange(0, 1, 10))
    
    x => 2.0
    y => 1.0
    z => 5.0
    result = symeval(ma, SymbolicCache())
    println(result)
    # ex = exp(Matrix(symeval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "CTMCTran2" begin
    @vars x y z
    Q = [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x => 2.0
    y => 1.0
    z => 5.0

    result = symeval(ma, :x, SymbolicCache())
    println(result)
    # ex = exp(Matrix(symeval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "CTMCTran3" begin
    @vars x y z
    Q = [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x => 2.0
    y => 1.0
    z => 5.0
    result = symeval(ma, (:x,:y), SymbolicCache())
    println(result)
    # ex = exp(Matrix(symeval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end
