module SymbolicMarkov

# export SymbolicMarkovConfig
# export ctmctran
# export ctmcst, CTMCST_CONFIG

#export gth
export Markov, trans!, reward!, initial!, generate, @transition, @reward, @initial, @parameters

# using SparseMatrix: AbstractSparseM, SparseCSR, SparseCSC, SparseCOO, BlockCOO
# using NMarkov: gth, stgs, stsengs, @dot, AbstractTranspose, NoTrans, Trans, mexp
using SymbolicDiff: AbstractSymbolic, SymbolicMatrix, SymbolicCSCMatrix, SymbolicEnv, SymbolicCache #, SymbolicExpression, AbstractSymbolicMatrix, AbstractSymbolicVector, SymbolicEnv, SymbolicCache, symboliceval
import NMarkov: gth, stgs, stsengs
import SymbolicDiff: _eval, symboliceval, @expr, symbolic
using SparseArrays: spzeros
import Base

include("_markov.jl")
include("_ctmcst.jl")

end
