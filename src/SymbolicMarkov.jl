module SymbolicMarkov

# export SymbolicMarkovConfig
# export ctmctran
# export ctmcst, CTMCST_CONFIG

#export gth
export Markov, trans!, reward!, initial!, generate, @transition, @reward, @initial, @parameters

using SymbolicDiff: AbstractSymbolic, SymbolicEnv, SymbolicCache, SymbolicExpression
import NMarkov: gth, stgs, stsengs, tran
import SymbolicDiff: _toexpr, _eval, symboliceval, @expr, symbolic
using SparseArrays: SparseMatrixCSC, spzeros
using SparseMatrix: SparseCSC, BlockCOO
import Base
import LinearAlgebra: dot


include("_markov.jl")
include("_ctmcst.jl")
include("_ctmctran.jl")

end
