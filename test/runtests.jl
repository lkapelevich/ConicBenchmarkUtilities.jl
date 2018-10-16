using Test
using ECOS, SCS
using MathProgBase
using MathOptInterface
using JuMP
using SparseArrays
using LinearAlgebra
using ConicBenchmarkUtilities

const MOI = MathOptInterface

@testset "example4.cbf" begin

    dat = readcbfdata("example4.cbf")

    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)

    @test c ≈ [1.0, 0.64]
    @test A ≈ [-50.0 -31; -3.0 2.0]
    @test b ≈ [-250.0, 4.0]
    @test vartypes == [:Cont, :Cont]
    @test dat.sense == :Max
    @test dat.objoffset == 0.0
    @test con_cones == [(:NonPos,[1]),(:NonNeg,[2])]

    m = MathProgBase.ConicModel(ECOSSolver(verbose=0))
    MathProgBase.loadproblem!(m, -c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(m)

    x_sol = MathProgBase.getsolution(m)
    objval = MathProgBase.getobjval(m)

    @test x_sol ≈  [1.9482; 4.9222] atol=1e-4
    @test objval ≈ -5.0984 atol=1e-4

    # test CBF writer
    newdat = mpbtocbf("example", c, A, b, con_cones, var_cones, vartypes, dat.sense)
    writecbfdata("example_out.cbf",newdat,"# Example C.4 from the CBF documentation version 2")
    @test strip(read("example4.cbf", String)) == strip(read("example_out.cbf", String))
    rm("example_out.cbf")

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

# SDP tests

@testset "roundtrip read/write" begin
    dat = readcbfdata("example1.cbf")
    @test dat.sense == :Min
    @test dat.objoffset == 0.0
    @test isempty(dat.intlist)
    writecbfdata("example_out.cbf",dat,"# Example C.1 from the CBF documentation version 2")
    @test strip(read("example1.cbf", String)) == strip(read("example_out.cbf", String))
    rm("example_out.cbf")
end

@testset "roundtrip through MPB format" begin
    dat = readcbfdata("example1.cbf")
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
    dat = readcbfdata("psd_var_only.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)
    @test var_cones == [(:SDP, [1, 2, 3])]
end

@testset "roundtrip through MPB solver" begin
    dat = readcbfdata("example1.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)
    m = MathProgBase.ConicModel(SCSSolver(eps=1e-6, verbose=0))
    MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(m)
    @test MathProgBase.status(m) == :Optimal

    (scalar_solution, psdvar_solution) = ConicBenchmarkUtilities.mpb_sol_to_cbf(dat,MathProgBase.getsolution(m))

    jm = JuMP.Model(with_optimizer(SCS.Optimizer))
    @JuMP.variable(jm, x[1:3])
    @JuMP.variable(jm, X[1:3,1:3], PSD)

    @JuMP.objective(jm, Min, dot([2 1 0; 1 2 1; 0 1 2],X) + x[2])
    @JuMP.constraint(jm, X[1,1]+X[2,2]+X[3,3]+x[2] == 1.0)
    @JuMP.constraint(jm, dot(ones(3,3),X) + x[1] + x[3] == 0.5)
    @JuMP.constraint(jm, [x[2], x[1], x[3]] in MOI.SecondOrderCone(3))
    JuMP.optimize!(jm)
    @test JuMP.termination_status(jm) == MOI.Success
    @test JuMP.objective_value(jm) ≈ MathProgBase.getobjval(m) atol=1e-4
    for i in 1:3
        @test JuMP.result_value(x[i]) ≈ scalar_solution[i] atol=1e-4
    end
    for i in 1:3, j in 1:3
        @test JuMP.result_value(X[i,j]) ≈ psdvar_solution[1][i,j] atol=1e-4
    end
end

@testset "example3.cbf" begin
    dat = readcbfdata("example3.cbf")
    (c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset) = cbftompb(dat)

    @test dat.sense == :Min
    @test dat.objoffset == 1.0
    @test all(vartypes .== :Cont)

    m = MathProgBase.ConicModel(SCSSolver(eps=1e-6, verbose=0))
    MathProgBase.loadproblem!(m, c, A, b, con_cones, var_cones)
    MathProgBase.optimize!(m)
    @test MathProgBase.status(m) == :Optimal

    scalar_solution, psdvar_solution = ConicBenchmarkUtilities.mpb_sol_to_cbf(dat,MathProgBase.getsolution(m))

    jm = JuMP.Model(with_optimizer(SCS.Optimizer))
    @JuMP.variable(jm, x[1:2])
    @JuMP.variable(jm, X[1:2,1:2], PSD)

    @JuMP.objective(jm, Min, X[1,1] + X[2,2] + x[1] + x[2] + 1)
    @JuMP.constraint(jm, X[1,2] + X[2,1] - x[1] - x[2] ≥ 0.0)
    @JuMP.SDconstraint(jm, [0 x[1]; x[1] 3x[1]] + [3x[2] x[2]; x[2] 0] ⪰ [1 0; 0 1])
#    @JuMP.SDconstraint(jm, [0 1; 1 3]*x[1] + [3 1; 1 0]*x[2] - [1 0; 0 1] >= 0)

    JuMP.optimize!(jm)
    @test JuMP.termination_status(jm) == MOI.Success
    @test JuMP.objective_value(jm) ≈ MathProgBase.getobjval(m)+dat.objoffset atol=1e-4
    for i in 1:2
        @test JuMP.result_value(x[i]) ≈ scalar_solution[i] atol=1e-4
    end
    for i in 1:2, j in 1:2
        @test JuMP.result_value(X[i,j]) ≈ psdvar_solution[1][i,j] atol=1e-4
    end

    # should match example3 modulo irrelevant changes
    # ConicBenchmarkUtilities.jump_to_cbf(jm, "example3", "sdptest.cbf")
    #
    # output = """
    # # Generated by ConicBenchmarkUtilities.jl
    # VER
    # 2
    #
    # OBJSENSE
    # MIN
    #
    # PSDVAR
    # 1
    # 2
    #
    # VAR
    # 2 1
    # F 2
    #
    # PSDCON
    # 1
    # 2
    #
    # CON
    # 1 1
    # L- 1
    #
    # OBJFCOORD
    # 2
    # 0 0 0 1.0
    # 0 1 1 1.0
    #
    # OBJACOORD
    # 2
    # 0 1.0
    # 1 1.0
    #
    # FCOORD
    # 1
    # 0 0 0 1 -0.9999999999999999
    #
    # ACOORD
    # 2
    # 0 0 1.0
    # 0 1 1.0
    #
    # HCOORD
    # 4
    # 0 0 0 1 0.9999999999999999
    # 0 0 1 1 3.0
    # 0 1 0 0 3.0
    # 0 1 0 1 0.9999999999999999
    #
    # DCOORD
    # 2
    # 0 0 0 -1.0
    # 0 1 1 -1.0
    #
    # """
    #
    # @test readstring("sdptest.cbf") == output
    # rm("sdptest.cbf")
end
