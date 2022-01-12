@testset "CTMCTran1" begin
    Q = @expr [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = exrt(LinRange(0, 1, 10), Q, x, r)
    @bind begin
        x = 2.0
        y = 1.0
        z = 5.0
    end
    result = seval(ma)
    println(result)
    # ex = exp(Matrix(seval(Q, test, SymbolicCache())))' * [1,0,0]
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
    ma = exrt(LinRange(0, 1, 10), Q, x, r)
    
    @bind begin
        x = 2.0
        y = 1.0
        z = 5.0
    end
    result = seval(ma, :x)
    println(result)
    # ex = exp(Matrix(seval(Q, test, SymbolicCache())))' * [1,0,0]
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
    ma = exrt(LinRange(0, 1, 10), Q, x, r)
    
    @bind begin
        x = 2.0
        y = 1.0
        z = 5.0
    end
    result = seval(ma, (:x,:y))
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "CTMCTran4" begin
    Q = @expr [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = exrt(1.0, Q, x, r)
    
    @bind begin
        x = 2.0
        y = 1.0
        z = 5.0
    end
    result = seval(ma, (:x,:y))
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end

@testset "CTMCTran4" begin
    Q = @expr [
        -x x 0;
        y -(y+z) z;
        0 1 -1
    ]
    x = @expr [1.0, 0, 0]
    r = @expr [1.0, 1, 0]
    ma = cexrt(LinRange(0, 1, 10), Q, x, r)
    
    @bind begin
        x = 2.0
        y = 1.0
        z = 5.0
    end
    result = seval(ma, (:x,:y))
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end