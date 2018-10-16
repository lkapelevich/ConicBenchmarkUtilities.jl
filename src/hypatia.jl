
const conemap_mpb_to_hypatia = Dict(
    :NonPos => Hypatia.NonpositiveCone,
    :NonNeg =>  Hypatia.NonnegativeCone,
    :SOC => Hypatia.SecondOrderCone,
    :SOCRotated => Hypatia.RotatedSecondOrderCone,
    :ExpPrimal => Hypatia.ExponentialCone,
    # :ExpDual => "EXP*"
    :SDP => Hypatia.PositiveSemidefiniteCone
)

const DimCones = Union{Type{Hypatia.NonpositiveCone}, Type{Hypatia.NonnegativeCone}, Type{Hypatia.SecondOrderCone}, Type{Hypatia.RotatedSecondOrderCone}, Type{Hypatia.PositiveSemidefiniteCone}}

function get_hypatia_cone(t::T, dim::Int) where T <: DimCones
    t(dim)
end
function get_hypatia_cone(t::T, ::Int) where T <: Type{Hypatia.PrimitiveCone}
    t()
end
# function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::UnitRange{Int})
function add_hypatia_cone!(hypatia_cone::Hypatia.Cone, conesym::Symbol, idxs::Vector{Int})
    conetype = conemap_mpb_to_hypatia[conesym]
    conedim = length(idxs)
    push!(hypatia_cone.prmtvs, get_hypatia_cone(conetype, conedim))
    push!(hypatia_cone.idxs, UnitRange{Int}(idxs[1], idxs[end]))
    push!(hypatia_cone.useduals, false)
    hypatia_cone
end

function cbfcones_to_mpbcones!(hypatia_cone::Hypatia.Cone, mpb_cones::Vector{Tuple{Symbol,Vector{Int}}}, offset::Int=0)
    for c in mpb_cones
        c[1] in (:Zero, :Free) && continue
        smallest_ind = minimum(c[2])
        output_idxs = (offset- smallest_ind + 1) .+ c[2]
        offset += length(c[2])
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
    if dense
        A = zeros(zero_constrs, n)
        G = zeros(cone_constrs + cone_vars, n)
    else
        A = spzeros(zero_constrs, n)
        G = spzeros(cone_constrs + cone_vars, n)
    end

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

function cbftohypatia(dat::CBFData, remove_ints::Bool=false)
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true)
    (dat.sense == :Max) && (c .*= -1.0)
    if remove_ints
        (c, A, b, con_cones, var_cones, vartypes) = remove_ints_in_nonlinear_cones(c, A, b, con_cones, var_cones, vartypes)
    end
    (c, A, b, G, h, hypatia_cone) = mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset)
    (c, A, b, G, h, hypatia_cone, dat.objoffset)
end
