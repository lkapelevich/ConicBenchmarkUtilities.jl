
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
function get_hypatia_cone(t::Type{T}, ::Int) where T <: Hypatia.PrimitiveCone
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

function mbgtohypatia(c_in::Vector{Float64},
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
    c = copy(c_in)
    n = length(c_in)

    # brute force correct order for any exponential cones
    exp_con_indices = Vector{Int}[]

    # count the number of "zero" constraints
    zero_constrs = 0
    cone_constrs = 0
    for (cone_type, inds) in con_cones
        if cone_type == :Zero
            zero_constrs += length(inds)
        elseif cone_type == :ExpPrimal
            # expect exponential indices to be in reverse order
            cone_constrs += 3
            push!(exp_con_indices, inds)
        else
            cone_constrs += length(inds)
        end
    end

    # TODO when going CBF -> hypatia directly this will definitely not need to happen
    for ind_collection in exp_con_indices
        A_in[ind_collection, :] .= A_in[reverse(ind_collection), :]
        b_in[ind_collection, :] .= b_in[reverse(ind_collection), :]
    end

    # count the number of cone variables
    cone_vars = 0
    cone_var_inds = Int[]
    zero_var_inds = Int[]
    for (cone_type, inds) in var_cones
        # TODO treat fixed variables better
        if cone_type == :Zero
            zero_constrs += length(inds)
            push!(zero_var_inds, inds...)
        elseif cone_type == :ExpPrimal
            # take out if this ever happens
            error("We didn't know CBF allows variables in exponential cones.")
        elseif cone_type != :Free
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
            # i += 1
            nexti = i + length(inds)
            out_inds = i+1:nexti
            # out_inds = i:i+length(inds)-1
            A[out_inds, :] .= A_in[inds, :]
            b[out_inds] = b_in[inds]
            i = nexti
        else
            if cone_type == :ExpPrimal
                inds .= reverse(inds)
            end
            nextj = j + length(inds)
            out_inds = j+1:nextj
            # j += 1
            # out_inds = j:j+length(inds)-1
            G[out_inds, :] .= A_in[inds, :]
            @show h, cone_constrs
            h[out_inds] .= b_in[inds]
            j = nextj
        end
    end
    @show A, length(zero_var_inds), zero_var_inds, i
    # corner case, add variables fixed at zero as constraints TODO treat fixed variables better
    b[zero_var_inds] .= 0.0
    for (_, inds) in var_cones[zero_var_inds]
        l = length(inds)
        nexti = i + l
        out_inds = i+1:nexti
        A[out_inds, inds] .=  Matrix{Float64}(I, l, l)
        i = nexti
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
    c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset = cbftompb(dat, col_major=true, roundints=true)
    (dat.sense == :Max) && (c .*= -1.0)
    if remove_ints
        (c, A, b, con_cones, var_cones, vartypes) = remove_ints_in_nonlinear_cones(c, A, b, con_cones, var_cones, vartypes)
    end
    (c, A, b, G, h, hypatia_cone) = mbgtohypatia(c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset)
    (c, A, b, G, h, hypatia_cone, dat.objoffset)
end

# function cbfcones_to_hypatiacones(c::Vector{Tuple{String,Int}},total)
#     i = 1
#     mpb_cones = Tuple{Symbol,Vector{Int}}[]
#
#     for (cname,count) in c
#         conesymbol = conemap[cname]
#         if conesymbol == :ExpPrimal
#             @assert count == 3
#             indices = i+2:-1:i
#         else
#             indices = i:(i+count-1)
#         end
#         push!(mpb_cones, (conesymbol, collect(indices)))
#         i += count
#     end
#     @assert i == total + 1
#     return mpb_cones
# end

# function cbftohypatia(dat::CBFData; roundints::Bool=true)
#     @assert dat.nvar == (isempty(dat.var) ? 0 : sum(c->c[2],dat.var))
#     @assert dat.nconstr == (isempty(dat.con) ? 0 : sum(c->c[2],dat.con))
#
#     c = zeros(dat.nvar)
#     for (i,v) in dat.objacoord
#         c[i] = v
#     end
#
#     var_cones = cbfcones_to_mpbcones(dat.var, dat.nvar)
#     con_cones = cbfcones_to_mpbcones(dat.con, dat.nconstr)
#
#     I_A, J_A, V_A = unzip(dat.acoord)
#     b = zeros(dat.nconstr)
#     for (i,v) in dat.bcoord
#         b[i] = v
#     end
#
#     psdvarstartidx = Int[]
#     for i in 1:length(dat.psdvar)
#         if i == 1
#             push!(psdvarstartidx,dat.nvar+1)
#         else
#             push!(psdvarstartidx,psdvarstartidx[i-1] + psd_len(dat.psdvar[i-1]))
#         end
#         push!(var_cones,(:SDP,psdvarstartidx[i]:psdvarstartidx[i]+psd_len(dat.psdvar[i])-1))
#     end
#     nvar = (length(dat.psdvar) > 0) ? psdvarstartidx[end] + psd_len(dat.psdvar[end]) - 1 : dat.nvar
#
#     psdconstartidx = Int[]
#     for i in 1:length(dat.psdcon)
#         if i == 1
#             push!(psdconstartidx,dat.nconstr+1)
#         else
#             push!(psdconstartidx,psdconstartidx[i-1] + psd_len(dat.psdcon[i-1]))
#         end
#         push!(con_cones,(:SDP,psdconstartidx[i]:psdconstartidx[i]+psd_len(dat.psdcon[i])-1))
#     end
#     nconstr = (length(dat.psdcon) > 0) ? psdconstartidx[end] + psd_len(dat.psdcon[end]) - 1 : dat.nconstr
#
#     c = [c;zeros(nvar-dat.nvar)]
#     for (matidx,i,j,v) in dat.objfcoord
#         ix = psdvarstartidx[matidx] + idx_to_offset(dat.psdvar[matidx],i,j,col_major)
#         @assert c[ix] == 0.0
#         scale = (i == j) ? 1.0 : sqrt(2)
#         c[ix] = scale*v
#     end
#
#     for (conidx,matidx,i,j,v) in dat.fcoord
#         ix = psdvarstartidx[matidx] + idx_to_offset(dat.psdvar[matidx],i,j,col_major)
#         push!(I_A,conidx)
#         push!(J_A,ix)
#         scale = (i == j) ? 1.0 : sqrt(2)
#         push!(V_A,scale*v)
#     end
#
#     for (conidx,varidx,i,j,v) in dat.hcoord
#         ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
#         push!(I_A,ix)
#         push!(J_A,varidx)
#         scale = (i == j) ? 1.0 : sqrt(2)
#         push!(V_A,scale*v)
#     end
#
#     b = [b;zeros(nconstr-dat.nconstr)]
#     for (conidx,i,j,v) in dat.dcoord
#         ix = psdconstartidx[conidx] + idx_to_offset(dat.psdcon[conidx],i,j,col_major)
#         @assert b[ix] == 0.0
#         scale = (i == j) ? 1.0 : sqrt(2)
#         b[ix] = scale*v
#     end
#
#     A = sparse(I_A,J_A,-V_A,nconstr,nvar)
#
#     vartypes = fill(:Cont, nvar)
#     if !roundints
#         vartypes[dat.intlist] .= :Int
#     end
#
#     return c, A, b, con_cones, var_cones, vartypes, dat.sense, dat.objoffset
# end
