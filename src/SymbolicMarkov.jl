module SymbolicMarkov

export Markov, trans!, reward!, initial!, generate, @tr, @reward, @init, @markov, @s
export CTMCModel, ctmc, exrss, exrt, cexrt
export prob, tprob, ctprob

using SymbolicDiff: AbstractSymbolic, AbstractVectorSymbolic, AbstractMatrixSymbolic, SymbolicEnv, SymbolicCache, SymbolicExpression, @expr, @bind, @vars
import SymbolicDiff: _toexpr, _eval, seval
import NMarkov: gth, stsen, stgs, stsengs, tran, mexp, mexpc
using SparseArrays: SparseMatrixCSC, spzeros
using SparseMatrix: SparseCSC, BlockCOO
import Base
import LinearAlgebra: dot

include("_markov.jl")
include("_ctmc.jl")

include("_ctmc_prob.jl")
include("_ctmc_tprob.jl")
include("_ctmc_exr.jl")

end
