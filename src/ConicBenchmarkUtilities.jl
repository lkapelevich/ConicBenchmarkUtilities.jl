__precompile__()

module ConicBenchmarkUtilities

using GZip
using SparseArrays
using Hypatia
using LinearAlgebra

export readcbfdata, cbftompb, mpbtocbf, writecbfdata, cbftohypatia
export remove_zero_varcones, socrotated_to_soc, remove_ints_in_nonlinear_cones, dualize

include("cbf_input.jl")
include("cbf_output.jl")
include("mpb.jl")
include("hypatia.jl")
include("preprocess_mpb.jl")
include("jump_to_cbf.jl")

end # module
