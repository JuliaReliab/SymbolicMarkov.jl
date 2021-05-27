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
    csc = SymbolicCSCMatrix(Q)
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
    csc = SymbolicCSCMatrix(Q)
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
    csc = SymbolicCSCMatrix(Q)
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
    csc = SymbolicCSCMatrix(Q)
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
    csc = SymbolicCSCMatrix(Q)
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

@testset "Markov1" begin
    m = Markov()
    @transition m begin
        up => down, 1.0
        down => up, 100.0
    end
    @initial m begin
        up, 1.0
    end
    @reward m begin
        up, 1.0
    end
    initv, Q, rwd, = generate(m)
    println(initv)
    println(Q)
    println(rwd)
end

# TODO
# @testset "Markov2" begin
#     @parameters lam1 lam2
#     m = Markov(AbstractSymbolic{Float64})
#     @transition m begin
#         up => down, lam1
#         down => up, lam2
#     end
#     @initial m begin
#         up, @expr 1.0
#     end
#     @reward m begin
#         up, @expr 1.0
#     end
#     initv, Q, rwd, = generate(m)
#     println(initv)
#     println(Q)
#     println(rwd)
# end

# @testset "CTMCTran1" begin
#     Q = @expr [
#         -x x 0;
#         1 -1 0;
#         0 1 -1
#     ]
#     Q = SymbolicCSCMatrix(Q)
#     v = @expr [1, 0, 0]
#     ma = ctmctran(Q, v, @expr(t))
    
#     x = 2.0
#     @env test begin
#         x = x
#         t = 1.0
#     end
#     result = symboliceval(ma, test, SymbolicCache())
#     ex = exp(Matrix(symboliceval(Q, test, SymbolicCache())))' * [1,0,0]
#     @test isapprox(ex, result)
# end

# @testset "CTMCTran2" begin
#     Q = @expr [
#         -x x 0;
#         1 -1 0;
#         0 1 -1
#     ]
#     Q = SymbolicCSCMatrix(Q)
#     v = @expr [1, 0, 0]
#     ma = ctmctran(Q, v, @expr(t))
    
#     x = 2.0
#     @env test begin
#         x = x
#         t = 1.0
#     end

#     h = 0.0001
#     @env test1 begin
#         x = x + h
#         t = 1.0
#     end
#     @env test2 begin
#         x = x - h
#         t = 1.0
#     end
#     result = symboliceval(ma, :x, test, SymbolicCache())
#     ex = (symboliceval(ma, test1, SymbolicCache()) - symboliceval(ma, test2, SymbolicCache())) / (2*h)
#     @test isapprox(ex, result)
# end

# @testset "CTMCTran3" begin
#     Q = @expr [
#         -x x 0;
#         1 -1 0;
#         0 1 -1
#     ]
#     Q = SymbolicCSCMatrix(Q)
#     v = @expr [1, 0, 0]
#     ma = ctmctran(Q, v, @expr(t))
    
#     x = 2.0
#     @env test begin
#         x = x
#         t = 1.0
#     end

#     h = 0.0001
#     @env test1 begin
#         x = x + h
#         t = 1.0
#     end
#     @env test2 begin
#         x = x - h
#         t = 1.0
#     end
#     result = symboliceval(ma, (:x,:x), test, SymbolicCache())
#     ex = (symboliceval(ma, test1, SymbolicCache()) - 2*symboliceval(ma, test, SymbolicCache()) + symboliceval(ma, test2, SymbolicCache())) / (h^2)
#     @test isapprox(ex, result, atol=1.0e-5)
# end

# @testset "CTMCTran4" begin
#     Q = @expr [
#         -x x 0;
#         1 -1 0;
#         0 1 -1
#     ]
#     Q = SymbolicCSCMatrix(Q)
#     v = @expr [1, 0, 0]
#     r = @expr [0, x, 2]
#     ma = dot(ctmctran(Q, v, @expr(t)), r)
#     ma2 = ctmctran(Q, v, r, @expr(t))
    
#     x = 2.0
#     @env test begin
#         x = x
#         t = 1.0
#     end
#     result = symboliceval(ma, test, SymbolicCache())
#     ex = symboliceval(ma2, test, SymbolicCache())
#     @test isapprox(ex, result)
# end

# @testset "CTMCTran5" begin
#     Q = @expr [
#         -x x 0;
#         1 -1 0;
#         0 1 -1
#     ]
#     Q = SymbolicCSCMatrix(Q)
#     v = @expr [1, 0, 0]
#     r = @expr [0, x, 2]
#     ma = dot(ctmctran(Q, v, @expr(t)), r)
#     ma2 = ctmctran(Q, v, r, @expr(t))
    
#     x = 2.0
#     @env test begin
#         x = x
#         t = 1.0
#     end
#     result = symboliceval(ma, :x, test, SymbolicCache())
#     ex = symboliceval(ma2, :x, test, SymbolicCache())
#     @test isapprox(ex, result)
# end

# @testset "CTMCTran6" begin
#     Q = @expr [
#         -x x 0;
#         1 -1 0;
#         0 1 -1
#     ]
#     Q = SymbolicCSCMatrix(Q)
#     v = @expr [1, 0, 0]
#     r = @expr [0, x, 2]
#     ma = dot(ctmctran(Q, v, @expr(t)), r)
#     ma2 = ctmctran(Q, v, r, @expr(t))
    
#     x = 2.0
#     @env test begin
#         x = x
#         t = 1.0
#     end
#     result = symboliceval(ma, (:x,:x), test, SymbolicCache())
#     ex = symboliceval(ma2, (:x,:x), test, SymbolicCache())
#     @test isapprox(ex, result)
# end
