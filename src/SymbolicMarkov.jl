module SymbolicMarkov

# export SymbolicMarkovConfig
# export ctmctran
# export ctmcst, CTMCST_CONFIG

export gth

# using SparseMatrix: AbstractSparseM, SparseCSR, SparseCSC, SparseCOO, BlockCOO
# using NMarkov: gth, stgs, stsengs, @dot, AbstractTranspose, NoTrans, Trans, mexp
using SymbolicDiff: AbstractSymbolic, SymbolicMatrix, SymbolicCSCMatrix, SymbolicEnv, SymbolicCache #, SymbolicExpression, AbstractSymbolicMatrix, AbstractSymbolicVector, SymbolicEnv, SymbolicCache, symboliceval
import NMarkov: gth, stgs, stsengs
import SymbolicDiff: _eval, symboliceval

# const SymbolicMarkovConfig = Dict{Symbol,Union{Int,Float64}}(
#     :maxiter => 5000,
#     :steps => 20,
#     :reltol => 1.0e-6,
#     :ufact => 1.01,
#     :eps => 1.0e-8,
#     :rmax => 500,
# )

include("_ctmcst.jl")
# include("_ctmctran.jl")

end
