# Run by PackageCompiler in the process that writes the sysimage (the `script`
# argument), so this redefinition is what the image contains.
#
# PythonCall's `init_stdlib` finds juliacall's `init.jl` through
# `const ROOT_DIR = dirname(dirname(@__DIR__))`. In a sysimage that constant is
# frozen to the build machine's path, so juliacall fails to start anywhere else.
# This copy of the function finds `init.jl` in the Python `juliacall` package
# instead, which is already in `sys.modules` when Python embeds Julia and holds
# an identical copy of the file. It is otherwise PythonCall 0.9.36's function.
#
# Leaving PythonCall out of the image avoids this but costs ~40 s on the first
# call: loading it on top of the image invalidates much of the image's code.
using PythonCall

pkgversion(PythonCall) == v"0.9.36" || error(
    "PythonCall is $(pkgversion(PythonCall)), not 0.9.36: compare its " *
    "src/Core/stdlib.jl with scripts/sysimage/relocate_pythoncall.jl and update it",
)

@eval PythonCall.Core function init_stdlib()

    # check word size
    pywordsize = pygt(Bool, pysysmodule.maxsize, Int64(2)^32) ? 64 : 32
    pywordsize == Sys.WORD_SIZE ||
        error("Julia is $(Sys.WORD_SIZE)-bit but Python is $(pywordsize)-bit")

    if C.CTX.is_embedded

        # This uses some internals, but Base._start() gets the state more like Julia
        # is if you call the executable directly, in particular it creates workers when
        # the --procs argument is given.
        juliacall_dir = pystr_asstring(
            pyosmodule.path.dirname(pysysmodule.modules["juliacall"].__file__),
        )
        push!(Base.Core.ARGS, joinpath(juliacall_dir, "init.jl"))
        Base._start()
        Base.eval(:(PROGRAM_FILE = ""))

        # if Python is interactive, ensure Julia is too
        if pyhasattr(pysysmodule, "ps1")
            Base.eval(:(is_interactive = true))
            Base.load_InteractiveUtils()
        end

    else

        # set sys.argv
        pysysmodule.argv = pylist([""; ARGS])

        # some modules test for interactivity by checking if sys.ps1 exists
        if isinteractive() && !pyhasattr(pysysmodule, "ps1")
            pysysmodule.ps1 = ">>> "
        end

        # add hook to perform certain actions when certain modules are loaded
        g = pydict()
        pyexec(
            """
     import sys
     class JuliaCompatHooks:
         def __init__(self):
             self.hooks = {}
         def find_spec(self, name, path=None, target=None):
             hs = self.hooks.get(name)
             if hs is not None:
                 for h in hs:
                     h()
         def add_hook(self, name, h):
             if name not in self.hooks:
                 self.hooks[name] = [h]
             else:
                 self.hooks[name].append(h)
             if name in sys.modules:
                 h()
     JULIA_COMPAT_HOOKS = JuliaCompatHooks()
     sys.meta_path.insert(0, JULIA_COMPAT_HOOKS)
     """,
            g,
        )
        pycopy!(pymodulehooks, g["JULIA_COMPAT_HOOKS"])
    end

end
