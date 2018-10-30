using Test
using ECOS, SCS, Mosek
using MathProgBase
import JuMP
using SparseArrays
using LinearAlgebra
using Hypatia
using ConicBenchmarkUtilities

@testset "example1.cbf" begin
    dat = readcbfdata("test/example1.cbf")
    # SOC constraint, PSD variables

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)

    mpb_m = MathProgBase.ConicModel(MosekSolver())
    MathProgBase.loadproblem!(mpb_m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(mpb_m)

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true)
    (c1, A1, b1, G, h, hypatia_cone) = ConicBenchmarkUtilities.mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset, dense = false)
    Hypatia.check_data(c1, A1, b1, G, h, hypatia_cone)
    (c2, A2, b2, G2, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c1, A1, b1, G, useQR=true)
    L = Hypatia.QRSymmCache(c2, A2, b2, G2, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=100, verbose=true)
    Hypatia.load_data!(mdl, c2, A2, b2, G2, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    MathProgBase.getsolution(mpb_m)
    mdl.x

    @test isapprox(MathProgBase.getobjval(mpb_m), Hypatia.get_pobj(mdl), atol=1e-4)
end

@testset "example1c.cbf" begin
    dat = readcbfdata("test/example1c.cbf")
    # Like example 1, but includes a fixed variable

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)

    mpb_m = MathProgBase.ConicModel(MosekSolver())
    MathProgBase.loadproblem!(mpb_m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(mpb_m)

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true)
    (c1, A1, b1, G, h, hypatia_cone) = ConicBenchmarkUtilities.mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset, dense = false)
    @test isapprox(A1, sparse([0.0 -1.0 0.0 0.0 -1.0 0.0 -1.0 0.0 0.0 -1.0; -1.0 0.0 -1.0 0.0 -1.0 -1.41421 -1.0 -1.41421 -1.41421 -1.0; 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0]), atol=1e-4)
    @test isapprox(b1, [-1.0  -0.5  0.0]', atol=1e-4)
    Hypatia.check_data(c1, A1, b1, G, h, hypatia_cone)
    (c2, A2, b2, G2, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c1, A1, b1, G, useQR=true)
    L = Hypatia.QRSymmCache(c2, A2, b2, G2, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=100, verbose=true)
    Hypatia.load_data!(mdl, c2, A2, b2, G2, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    MathProgBase.getsolution(mpb_m)
    mdl.x

    @test isapprox(MathProgBase.getobjval(mpb_m), Hypatia.get_pobj(mdl), atol=1e-4)
end

@testset "example1d.cbf" begin
    dat = readcbfdata("test/example1d.cbf")
    # RSOC constraint, PSD variables

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)

    mpb_m = MathProgBase.ConicModel(MosekSolver())
    MathProgBase.loadproblem!(mpb_m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(mpb_m)

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true)
    (c1, A1, b1, G, h, hypatia_cone) = ConicBenchmarkUtilities.mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset, dense = false)
    Hypatia.check_data(c1, A1, b1, G, h, hypatia_cone)
    (c2, A2, b2, G2, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c1, A1, b1, G, useQR=true)
    L = Hypatia.QRSymmCache(c2, A2, b2, G2, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=200, verbose=true)
    Hypatia.load_data!(mdl, c2, A2, b2, G2, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    @test isapprox(MathProgBase.getobjval(mpb_m), Hypatia.get_pobj(mdl), atol=1e-4)
end

@testset "example3.cbf" begin
    dat = readcbfdata("test/example3.cbf")

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)

    mpb_m = MathProgBase.ConicModel(SCSSolver())
    MathProgBase.loadproblem!(mpb_m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(mpb_m)

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true)
    (c1, A1, b1, G, h, hypatia_cone) = ConicBenchmarkUtilities.mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset)
    Hypatia.check_data(c1, A1, b1, G, h, hypatia_cone)
    (c2, A2, b2, G2, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c1, A1, b1, G, useQR=true)
    L = Hypatia.QRSymmCache(c2, A2, b2, G2, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=100, verbose=true)
    Hypatia.load_data!(mdl, c2, A2, b2, G2, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    MathProgBase.getsolution(mpb_m)
    mdl.x

    @test isapprox(MathProgBase.getobjval(mpb_m), Hypatia.get_pobj(mdl), atol=1e-4)
end

@testset "example4.cbf to mpb" begin

    dat = readcbfdata("test/example4.cbf")

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)

    @test c ≈ [1.0, 0.64]
    @test A ≈ [-50.0 -31; -3.0 2.0]
    @test b ≈ [-250.0, 4.0]
    @test vartypes == [:Cont, :Cont]
    @test dat.sense == :Max
    @test dat.objoffset == 0.0
    @test con_cones == [(:NonPos,[1]),(:NonNeg,[2])]

    m = MathProgBase.ConicModel(MosekSolver())
    MathProgBase.loadproblem!(m, -c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(m)

    x_sol = MathProgBase.getsolution(m)
    objval = MathProgBase.getobjval(m)

    @test x_sol ≈  [1.9482; 4.9222] atol=1e-4
    @test objval ≈ -5.0984 atol=1e-4

    # test CBF writer
    newdat = mpbtocbf("example", c, A, b, con_cones, var_cones, vartypes, dat.sense)
    writecbfdata("example_out.cbf",newdat,"# Example C.4 from the CBF documentation version 2")
    @test strip(read("test/example4.cbf", String)) == strip(read("example_out.cbf", String))
    rm("example_out.cbf")

end

@testset "example5 power cone.cbf" begin

    dat = readcbfdata("test/example5.cbf")
    # includes a power cone

    c, A, b, G, h, hypatia_cone, dat.objoffset = cbftohypatia(dat, dense=true)

    # note order of x1 and x3 swaps for Hypatia power cone definition
    @test c ≈ [-1.0;  0.0;  0.0]
    @test G ≈ [0.0  -1.0  -1.0
              0.0   0.0   0.0
              0.0  -1.0   0.0
              0.0  -1.0  -1.0
              0.0   0.0   0.0
              0.0   0.0  -1.0
              0.0  -1.0   0.0
              0.0   0.0  -1.0
             -1.0   0.0   0.0]
    @test h ≈ [0.0  1.0  0.0  0.0  1.0  0.0  0.0  0.0  0.0]'
    @test size(A, 1) == 0
    @test size(b, 1) == 0
    @test hypatia_cone.prmtvs[1].alpha ≈ [8.0/9.0; 1.0 /9.0]
    @test hypatia_cone.prmtvs[2].alpha ≈ [8.0/9.0; 1.0 /9.0]
    @test hypatia_cone.prmtvs[3].alpha ≈ [0.5; 0.5]

    Hypatia.check_data(c, A, b, G, h, hypatia_cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
    L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=100, verbose=false)
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    @test mdl.x[1] ≈ 1.0
    @test Hypatia.get_pobj(mdl) ≈ -1.0
end

@testset "example4.cbf to Hypatia" begin

    dat = readcbfdata("test/example4.cbf")
    c, A, b, G, h, hypatia_cone, dat.objoffset = cbftohypatia(dat)

    @test c ≈ [-1.0, -0.64]
    @test G ≈ [-50.0 -31; -3.0 2.0; -1.0 0.0; 0.0 -1.0]
    @test h ≈ [-250.0, 4.0, 0.0, 0.0]
    @test size(A) == (0, 2)
    @test size(b) == (0,)
    @test dat.objoffset == 0.0
    @test typeof.(hypatia_cone.prmtvs) == [Hypatia.Nonpositive; Hypatia.Nonnegative; Hypatia.Nonnegative]
    @test hypatia_cone.idxs == [1:1, 2:2, 3:4]

    Hypatia.check_data(c, A, b, G, h, hypatia_cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
    L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=100, verbose=false)
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    @test mdl.x ≈ [1.9482; 4.9222] atol=1e-4
    @test Hypatia.get_pobj(mdl) ≈ -5.0984 atol=1e-4

    # # test CBF writer
    # newdat = mpbtocbf("example", c, A, b, con_cones, var_cones, vartypes, dat.sense)
    # writecbfdata("example_out.cbf",newdat,"# Example C.4 from the CBF documentation version 2")
    # @test strip(read("example4.cbf", String)) == strip(read("example_out.cbf", String))
    # rm("example_out.cbf")

end

# test transformation utilities

@testset "dualize" begin
    # max  y + z
    # st   x <= 1
    #     (x,y,z) in SOC
    #      x in {0,1}
    c = [0.0, -1.0, -1.0]
    A = [1.0  0.0  0.0;
        -1.0  0.0  0.0;
         0.0 -1.0  0.0;
         0.0  0.0 -1.0]
    b = [1.0, 0.0, 0.0, 0.0]
    con_cones = [(:NonNeg,1:1), (:SOC,2:4)]
    var_cones = [(:Free,1:3)]

    (c, A, b, con_cones, var_cones) = dualize(c, A, b, con_cones, var_cones)

    @test c == [1.0, 0.0, 0.0, 0.0]
    @test A == [-1.0  1.0  0.0  0.0;
                 0.0  0.0  1.0  0.0;
                 0.0  0.0  0.0  1.0]
    @test b == [0.0, -1.0, -1.0]
    @test con_cones == [(:Zero,1:3)]
    @test var_cones == [(:NonNeg,1:1), (:SOC,2:4)]

end

@testset "socrotated_to_soc" begin
    # SOCRotated1 from MathProgBase conic tests
    c = [ 0.0, 0.0, -1.0, -1.0]
    A = [ 1.0  0.0   0.0   0.0
          0.0  1.0   0.0   0.0]
    b = [ 0.5, 1.0]
    con_cones = [(:Zero,1:2)]
    var_cones = [(:SOCRotated,1:4)]
    vartypes = fill(:Cont,4)

    (c, A, b, con_cones, var_cones, vartypes) = socrotated_to_soc(c, A, b, con_cones, var_cones, vartypes)

    @test c == [0.0,0.0,-1.0,-1.0]
    @test b == [0.5,1.0,0.0,0.0,0.0,0.0]
    @test A ≈ [1.0 0.0 0.0 0.0
                       0.0 1.0 0.0 0.0
                      -1.0 -1.0 0.0 0.0
                      -1.0 1.0 0.0 0.0
                       0.0 0.0 -1.4142135623730951 0.0
                       0.0 0.0 0.0 -1.4142135623730951]
    @test var_cones == [(:Free,1:4)]
    @test con_cones == [(:Zero,1:2),(:SOC,3:6)]

    c = [-1.0,-1.0]
    A = [0.0 0.0; 0.0 0.0; -1.0 0.0; 0.0 -1.0]
    b = [0.5, 1.0, 0.0, 0.0]
    con_cones = [(:SOCRotated,1:4)]
    var_cones = [(:Free,1:2)]
    vartypes = fill(:Cont,2)

    (c, A, b, con_cones, var_cones, vartypes) = socrotated_to_soc(c, A, b, con_cones, var_cones, vartypes)

    @test c == [-1.0,-1.0,0.0,0.0,0.0,0.0]
    @test b == [0.5,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
    @test A == [0.0 0.0 1.0 0.0 0.0 0.0
     0.0 0.0 0.0 1.0 0.0 0.0
     -1.0 0.0 0.0 0.0 1.0 0.0
     0.0 -1.0 0.0 0.0 0.0 1.0
     0.0 0.0 -1.0 -1.0 0.0 0.0
     0.0 0.0 -1.0 1.0 0.0 0.0
     0.0 0.0 0.0 0.0 -1.4142135623730951 0.0
     0.0 0.0 0.0 0.0 0.0 -1.4142135623730951]
    @test var_cones == [(:Free,1:2),(:Free,3:6)]
    @test con_cones == [(:Zero,1:4),(:SOC,5:8)]
end

@testset "remove_ints_in_nonlinear_cones" begin
    # SOCINT1
    c = [ 0.0, -2.0, -1.0]
    A = sparse([ 1.0   0.0   0.0])
    b = [ 1.0]
    con_cones = [(:Zero,1)]
    var_cones = [(:SOC,1:3)]
    vartypes = [:Cont,:Bin,:Bin]

    (c, A, b, con_cones, var_cones, vartypes) = remove_ints_in_nonlinear_cones(c, A, b, con_cones, var_cones, vartypes)

    @test c == [0.0,-2.0,-1.0,0.0,0.0]
    @test b == [1.0,0.0,0.0]
    @test A == [1.0 0.0 0.0 0.0 0.0
     0.0 1.0 0.0 -1.0 0.0
     0.0 0.0 1.0 0.0 -1.0]
    @test var_cones == [(:SOC,[1,4,5]),(:Free,[2,3])]
    @test con_cones == [(:Zero,[1]),(:Zero,[2,3])]
end

@testset "exponential cone" begin

    # TODO test save output from JuMP/hypatia as exptest.cbf

    (c, A, b, con_cones, var_cones, vartypes, sense, objoffset) = cbftompb(readcbfdata("test/exptest.cbf"))

    @test sense == :Min
    @test objoffset == 0.0
    @test all(vartypes .== :Cont)
    md = MathProgBase.ConicModel(ECOSSolver(verbose=0))
    MathProgBase.loadproblem!(md, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(md)
    @test MathProgBase.status(md) == :Optimal
    x_sol = MathProgBase.getsolution(md)
    @test x_sol ≈ [1.0,exp(1),exp(1)] atol=1e-5
    @test MathProgBase.getobjval(md) ≈ exp(1) atol=1e-5

    dat = readcbfdata("test/exptest.cbf")
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true)
    (c1, A1, b1, G, h, hypatia_cone) = ConicBenchmarkUtilities.mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.con_power_refs, dat.var_power_refs, dat.power_cone_alphas, dat.objoffset)
    Hypatia.check_data(c1, A1, b1, G, h, hypatia_cone)
    (c2, A2, b2, G2, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c1, A1, b1, G, useQR=true)
    L = Hypatia.QRSymmCache(c2, A2, b2, G2, h, hypatia_cone, Q2, RiQ1)
    mdl = Hypatia.Model(maxiter=100, verbose=true)
    Hypatia.load_data!(mdl, c2, A2, b2, G2, h, hypatia_cone, L)
    Hypatia.solve!(mdl)

    @test Hypatia.get_pobj(mdl) ≈ exp(1) atol=1e-5

    # rm("exptest.cbf")
end

# SDP tests

@testset "roundtrip read/write" begin
    dat = readcbfdata("test/example1.cbf")
    @test dat.sense == :Min
    @test dat.objoffset == 0.0
    @test isempty(dat.intlist)
    writecbfdata("example_out.cbf",dat,"# Example C.1 from the CBF documentation version 2")
    @test strip(read("test/example1.cbf", String)) == strip(read("example_out.cbf", String))
    rm("example_out.cbf")
end

@testset "roundtrip through MPB format" begin
    dat = readcbfdata("test/example1.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)
    newdat = mpbtocbf("example", c, A, b, con_cones, var_cones, vartypes, dat.sense)
    writecbfdata("example_out.cbf",newdat,"# Example C.1 from the CBF documentation version 2")
    output = """
    # Example C.1 from the CBF documentation version 2
    VER
    2

    OBJSENSE
    MIN

    PSDVAR
    1
    3

    VAR
    3 1
    F 3

    CON
    5 2
    L= 2
    Q 3

    OBJFCOORD
    5
    0 0 0 2.0
    0 0 1 1.0
    0 1 1 2.0
    0 1 2 1.0
    0 2 2 2.0

    OBJACOORD
    1
    1 1.0

    FCOORD
    9
    0 0 0 0 1.0
    1 0 0 0 1.0
    1 0 0 1 1.0
    1 0 0 2 1.0
    0 0 1 1 1.0
    1 0 1 1 1.0
    1 0 1 2 1.0
    0 0 2 2 1.0
    1 0 2 2 1.0

    ACOORD
    6
    1 0 1.0
    3 0 1.0
    0 1 1.0
    2 1 1.0
    1 2 1.0
    4 2 1.0

    BCOORD
    2
    0 -1.0
    1 -0.5

    """
    @test read("example_out.cbf", String) == output
    rm("example_out.cbf")
end

@testset "Instance with only PSD variables" begin
    dat = readcbfdata("test/psd_var_only.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)
    @test var_cones == [(:SDP, [1, 2, 3])]
end

@testset "Instance with only PSD variables hypatia" begin
    dat = readcbfdata("test/psd_var_only.cbf")
    (c, A, b, G, h, hypatia_cone, dat.objoffset) = cbftohypatia(dat)
    @test isa(hypatia_cone.prmtvs[1], Hypatia.PosSemidef)
    @test hypatia_cone.prmtvs[1].dim == 3
    @test hypatia_cone.idxs == [1:3]
end

SCSSOLVER = SCSSolver(eps=1e-6, verbose=0)

@testset "roundtrip through MPB solver" begin
    dat = readcbfdata("test/example1.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)
    m = MathProgBase.ConicModel(SCSSOLVER)
    MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(m)
    @test MathProgBase.status(m) == :Optimal

    (scalar_solution, psdvar_solution) = ConicBenchmarkUtilities.mpb_sol_to_cbf(dat,MathProgBase.getsolution(m))

    jm = JuMP.Model(solver=SCSSOLVER)
    @JuMP.variable(jm, x[1:3])
    @JuMP.variable(jm, X[1:3,1:3], SDP)

    @JuMP.objective(jm, Min, dot([2 1 0; 1 2 1; 0 1 2],X) + x[2])
    @JuMP.constraint(jm, X[1,1]+X[2,2]+X[3,3]+x[2] == 1.0)
    @JuMP.constraint(jm, dot(ones(3,3),X) + x[1] + x[3] == 0.5)
    @JuMP.constraint(jm, norm([x[1],x[3]]) <= x[2])
    @test JuMP.solve(jm) == :Optimal
    @test JuMP.getobjectivevalue(jm) ≈ MathProgBase.getobjval(m) atol=1e-4
    for i in 1:3
        @test JuMP.getvalue(x[i]) ≈ scalar_solution[i] atol=1e-4
    end
    for i in 1:3, j in 1:3
        @test JuMP.getvalue(X[i,j]) ≈ psdvar_solution[1][i,j] atol=1e-4
    end
end

@testset "example3.cbf" begin
    dat = readcbfdata("test/example3.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)

    @test dat.sense == :Min
    @test dat.objoffset == 1.0
    @test all(vartypes .== :Cont)

    m = MathProgBase.ConicModel(SCSSOLVER)
    MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(m)
    @test MathProgBase.status(m) == :Optimal

    scalar_solution, psdvar_solution = ConicBenchmarkUtilities.mpb_sol_to_cbf(dat,MathProgBase.getsolution(m))

    jm = JuMP.Model(solver=SCSSOLVER)
    @JuMP.variable(jm, x[1:2])
    @JuMP.variable(jm, X[1:2,1:2], SDP)

    @JuMP.objective(jm, Min, X[1,1] + X[2,2] + x[1] + x[2] + 1)
    @JuMP.constraint(jm, X[1,2] + X[2,1] - x[1] - x[2] ≥ 0.0)
    @JuMP.SDconstraint(jm, [0 1; 1 3]*x[1] + [3 1; 1 0]*x[2] - [1 0; 0 1] >= 0)

    @test JuMP.solve(jm) == :Optimal
    @test JuMP.getobjectivevalue(jm) ≈ MathProgBase.getobjval(m)+dat.objoffset atol=1e-4
    for i in 1:2
        @test JuMP.getvalue(x[i]) ≈ scalar_solution[i] atol=1e-4
    end
    for i in 1:2, j in 1:2
        @test JuMP.getvalue(X[i,j]) ≈ psdvar_solution[1][i,j] atol=1e-4
    end

    # should match example3 modulo irrelevant changes
    ConicBenchmarkUtilities.jump_to_cbf(jm, "example3", "sdptest.cbf")

    output = """
    # Generated by ConicBenchmarkUtilities.jl
    VER
    2

    OBJSENSE
    MIN

    PSDVAR
    1
    2

    VAR
    2 1
    F 2

    PSDCON
    1
    2

    CON
    1 1
    L- 1

    OBJFCOORD
    2
    0 0 0 1.0
    0 1 1 1.0

    OBJACOORD
    2
    0 1.0
    1 1.0

    FCOORD
    1
    0 0 0 1 -0.9999999999999999

    ACOORD
    2
    0 0 1.0
    0 1 1.0

    HCOORD
    4
    0 0 0 1 0.9999999999999999
    0 0 1 1 3.0
    0 1 0 0 3.0
    0 1 0 1 0.9999999999999999

    DCOORD
    2
    0 0 0 -1.0
    0 1 1 -1.0

    """

    @test read("sdptest.cbf", String) == output
    rm("sdptest.cbf")
end
