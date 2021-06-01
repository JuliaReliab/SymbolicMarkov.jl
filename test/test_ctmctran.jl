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
    @bind begin
        :x => x
        :y => y
        :z => z
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
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x = 2.0
    y = 1.0
    z = 5.0
    @bind begin
        :x => x
        :y => y
        :z => z
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
    ma = tran(Q, x, r, LinRange(0, 1, 10))
    
    x = 2.0
    y = 1.0
    z = 5.0
    @bind begin
        :x => x
        :y => y
        :z => z
    end
    result = seval(ma, (:x,:y))
    println(result)
    # ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
    # @test isapprox(ex, result)
end
