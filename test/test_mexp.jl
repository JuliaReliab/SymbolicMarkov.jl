# @testset "mexp1" begin
#     @markov midplane(lam, mu) begin
#         @tr begin
#             :up => :down, lam
#             :down => :up, mu
#         end
#         @init begin
#             :up, 1
#         end
#     end

#     begin
#         lam = 0.4
#         mu = 1.5
#         delta = 1.5
#     end

#     Q, x, _, states = generate(midplane(lam, mu))
#     println(mexp(Q, x, 1.0))
# end

# @testset "mexp2" begin
#     @markov midplane(lam, mu) begin
#         @tr begin
#             :up => :down, lam
#             :down => :up, mu
#         end
#         @init begin
#             :up, 1
#         end
#     end

#     @bind begin
#         lam = 0.4
#         mu = 1.5
#         delta = 1.5
#     end

#     Q, x, _, states = generate(midplane(lam, mu))
#     println(seval(mexp(Q, x, 1.0)))
# end

# @testset "mexp3" begin
#     @markov midplane(lam, mu) begin
#         @tr begin
#             :up => :down, lam
#             :down => :up, mu
#         end
#         @init begin
#             :up, 1
#         end
#     end

#     @bind begin
#         lam = 0.4
#         mu = 1.5
#         delta = 1.5
#     end

#     Q, x, _, states = generate(midplane(lam, mu))
#     println(seval(mexp(Q, x, 1.0, transpose=:T), :lam))
# end

# @testset "mexp3" begin
#     @markov midplane(lam, mu, p) begin
#         @tr begin
#             :up => :down, lam
#             :down => :up, mu
#         end
#         @init begin
#             :up, p
#             :down, 1-p
#         end
#     end

#     @bind begin
#         lam = 0.4
#         mu = 1.5
#         delta = 1.5
#         p = 1.0
#     end

#     Q, x, _, states = generate(midplane(lam, mu, p))
#     println(seval(mexpc(Q, x, 100.0, transpose=:T), :p))
#     println(seval(mexpc(Q, x, 100.0, transpose=:T), :lam))
#     println(seval(mexpc(Q, x, 100.0, transpose=:T), (:lam, :p)))
# end
