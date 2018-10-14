
const conemap_mpb_to_hypatia = Dict(
    :NonPos => Hypatia.NonpositiveCone,
    :NonNeg =>  Hypatia.NonnegativeCone,
    :SOC => Hypatia.SecondOrderCone,
    :SOCRotated => Hypatia.RotatedSecondOrderCone,
    :ExpPrimal => Hypatia.ExponentialCone
    # :ExpDual => "EXP*"
)

const DimCones = Union{Type{Hypatia.NonpositiveCone}, Type{Hypatia.NonnegativeCone}, Type{Hypatia.SecondOrderCone}, Type{Hypatia.SecondOrderCone}}

function get_hypatia_cone(t::T, dim::Int) where T <: DimCones
    t(dim)
end
function get_hypatia_cone(t::T, ::Int) where T <: Type{Hypatia.PrimitiveCone}
    t()
end
# function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::UnitRange{Int})
function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::Vector{Int})
    @show conesym
    conetype = conemap_mpb_to_hypatia[conesym]
    conedim = length(idxs)
    push!(hypatia_cone.prms, get_hypatia_cone(conetype, conedim))
    push!(hypatia_cone.idxs, UnitRange{Int}(idxs[1], idxs[end]))
    @show hypatia_cone
    hypatia_cone
end

add_offset(v::Vector{Int}, offset::Int) = (v .+= offset)
add_offset(v::UnitRange{Int}, offset::Int) = UnitRange{Int}(v.start+offset:v.stop+offset)

function cbfcones_to_mpbcones!(hypatia_cone::Hypatia.Cone, mpb_cones::Vector{Tuple{Symbol,Vector{Int}}}, offset::Int=0)
    for c in mpb_cones
        if offset > 0
            output_idxs = add_offset(c[2], offset)
        else
            output_idxs = c[2]
        end
        add_hypatia_cone!(hypatia_cone, c[1], output_idxs)
    end
    hypatia_cone
end

function mbgtohypatia(c::Vector{Float64},
    A_in::AbstractMatrix,
    b_in::Vector{Float64},
    con_cones::Vector{Tuple{Symbol,Vector{Int}}},
    var_cones::Vector{Tuple{Symbol,Vector{Int}}},
    vartypes::Vector{Symbol},
    sense::Symbol,
    objoffset::Float64;
    dense::Bool=true
    )

    # cannot do integer variables yet
    for v in vartypes
        if v != :Cont
            error("We cannot handle binary or integer variables yet.")
        end
    end

    # dimension of x
    n = length(c)

    # count the number of "zero" constraints
    zero_constrs = 0
    cone_constrs = 0
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            zero_constrs += length(inds)
        else
            cone_constrs += length(inds)
        end
    end

    # count the number of cone variables
    cone_vars = 0
    cone_var_inds = Int[]
    for (cone_type, inds) in var_cones
        if cone_type != :Free
            cone_vars += length(inds)
            push!(cone_var_inds, inds...)
        end
    end
    @assert length(cone_var_inds) == cone_vars

    h = zeros(cone_constrs + cone_vars)
    b = zeros(zero_constrs)
    A = zeros(zero_constrs, n)
    G = zeros(cone_constrs + cone_vars, n)

    i = 0
    j = 0
    # constraints are split among A and G
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            i += 1
            out_inds = i:i+length(inds)-1
            A[out_inds, :] .= A_in[inds, :]
            b[out_inds] = b_in[inds]
        else
            j += 1
            out_inds = j:j+length(inds)-1
            G[out_inds, :] .= A_in[inds, :]
            h[out_inds] .= b_in[inds]
        end
    end
    # append G
    G[cone_constrs+1:end, :] .= Matrix(-1I, n, n)[cone_var_inds, :]

    # prepare cones
    hypatia_cone = Hypatia.Cone()
    cbfcones_to_mpbcones!(hypatia_cone, con_cones)
    cbfcones_to_mpbcones!(hypatia_cone, var_cones, cone_constrs)

    (c, A, b, G, h, hypatia_cone)

end

function cbftohypatia(dat::CBFData)
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat)
    (dat.sense == :Max) && (c .*= -1.0)
    (c, A, b, G, h, hypatia_cone) = mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset)
    (c, A, b, G, h, hypatia_cone, dat.objoffset)
end
