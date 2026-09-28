# Usage: julia build_sysimage.jl <project> <statements_file> <output.so>
#
# PackageCompiler lives in its own environment so it never enters the
# project that juliapkg manages.
using Pkg

project, statements_file, output = ARGS

build_env = joinpath(@__DIR__, ".build-env")
Pkg.activate(build_env; io=devnull)
if Base.find_package("PackageCompiler") === nothing
    Pkg.add("PackageCompiler")
end
using PackageCompiler

# PythonCall must be in the image: loading it on top of the image invalidates
# much of the image's code (~40 s of recompiling). relocate_pythoncall.jl stops
# it from remembering the build machine's install path.
packages = ["Finch", "HDF5", "NPZ", "TensorMarket", "PythonCall"]

create_sysimage(
    packages;
    project=project,
    sysimage_path=output,
    precompile_statements_file=statements_file,
    script=joinpath(@__DIR__, "relocate_pythoncall.jl"),
    # a generic target so the image runs on any x86_64 CPU, not just this one
    cpu_target=PackageCompiler.default_app_cpu_target(),
    include_transitive_dependencies=true,
)
