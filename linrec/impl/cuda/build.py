from pathlib import Path
from nvidia.cuda_runtime.include import __path__ as CUDA_RUNTIME_INCLUDES # .../site-packages/nvidia/cuda_runtime/include/

SRC_DIR = Path(__file__).parent
PKG_DIR = Path(__file__).parents[3]
BUILD_DIR = PKG_DIR / ".build"

### CUDA/C++ Compilation Arguments
LIB_SOURCES = [str(SRC_DIR / "linrec_ref.cu"), str(SRC_DIR / "linrec_tile.cu"), str(SRC_DIR / "linrec_pipe.cu")]
EXT_SOURCES = [str(SRC_DIR / "extension.cpp")]
EXE_SOURCES = [str(SRC_DIR / "executable.cpp")]
INCLUDES = [str(SRC_DIR)] + CUDA_RUNTIME_INCLUDES


CUDA_FLAGS = [
    # Options for specifying behavior of compiler/linker.
    # "--profile",                              # Instrument generated code/executable for use by gprof (Linux only).
    #"--device-debug",                          # Generate debug information for device code.
    "--generate-line-info",                     # Generate line-number information for device code.
    "-O3",                                      # Specify optimization level for host code.
    #'--ftemplate-depth 199',                   # Set the maximum instantiation depth for template classes to <limit> (default 199).
    "-std=c++20",                               # Select a particular C++ dialect
    #'--expt-relaxed-constexpr',                # Experimental flag: Allow host code to invoke __device__ constexpr functions

    # Options for passing specific phase options
    # -Xptxas: options for ptxas, the PTX optimizing assembler.
    #"-Xptxas", "-dlcm=cs",                        # Default cache modifier on global/generic load
    #"-Xptxas", "-dscm=cs",                        # Default cache modifier on global/generic store
    "-Xptxas", "-v",                               # Enable verbose mode which prints code generation statistics.
    #"-Xptxas", "-regUsageLevel=5",                # Lower values inhibit optimizations that aggressively increase register usage (BETA).
    "-Xptxas", "-warn-spills",                     # Warning if registers are spilled to local memory.
    "-Xptxas", "-warn-lmem-usage",                 # Warning if local memory is used.
    #"-Xptxas", "-Werror",                         # Make all warnings into errors.

    # Miscellaneous options for guiding the compiler driver
    "--keep",                                   # Keep all intermediate files that are generated during internal compilation steps.

    # Options for steering GPU code generation.
    "--use_fast_math",                          # Make use of fast math library.
    #'--maxrregcount=64',                       # Specify the maximum amount of registers that GPU functions can use.
    #'--fmad false',                            # This option disables the contraction of into floating-point multiply-add operations. '--use_fast_math' implies '--fmad=true'.
    
    # Generic tool options.
    "--source-in-ptx",                          # Interleave source in PTX. May only be used in conjunction with --device-debug or --generate-line-info.
    "--resource-usage",                         # Show resource usage such as registers and memory of the GPU code. Implies '--ptxas-options --verbose'.
    #'--fira-loop-pressure',
]

CPP_FLAGS = [
    "-std=c++20",
]




def extension(extra_cflags=[], extra_cuda_cflags=[], verbose=False, clean=False):
    import time
    from torch.utils.cpp_extension import load
    make_build_dir(clean=clean)
    start = time.perf_counter()

    try: 
        ext = load(
            name="pylinrec",
            sources=LIB_SOURCES + EXT_SOURCES,
            extra_include_paths=INCLUDES,
            extra_cflags=CPP_FLAGS + extra_cflags,
            extra_cuda_cflags=CUDA_FLAGS + extra_cuda_cflags,
            build_directory=str(BUILD_DIR),
            verbose=verbose,
        )
    except Exception as e: # create_compile even if build failed
        create_compile_commands(which="ext", verbose=verbose)
        raise e
    
    if verbose:
        duration = round(time.perf_counter() - start)
        print(f'Took {duration // 60}:{duration % 60:02d} min to build.')
    create_compile_commands(which="ext", verbose=verbose)
    return ext
    

def executable(extra_cflags=[], extra_cuda_cflags=[], verbose=False, clean=False):
    import time
    from torch.utils.cpp_extension import load
    make_build_dir(clean=clean)
    start = time.perf_counter()

    try:
        exe = load(
            name="cpplinrec",
            sources=LIB_SOURCES + EXE_SOURCES,
            extra_include_paths=INCLUDES,
            extra_cflags=CPP_FLAGS + extra_cflags,
            extra_cuda_cflags=CUDA_FLAGS + extra_cuda_cflags,
            build_directory=str(BUILD_DIR),
            is_python_module=False,
            is_standalone=True,
            verbose=verbose,
        )
    except Exception as e: # create_compile even if build failed
        create_compile_commands(which="exe", verbose=verbose)
        raise e
    
    if verbose:
        duration = round(time.perf_counter() - start)
        print(f'Took {duration // 60}:{duration % 60:02d} min to build.')
    create_compile_commands(which="exe", verbose=verbose)
    return exe


def make_build_dir(clean=False):
    import os, shutil
    if clean:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    return

def create_compile_commands(which="", verbose=True):
    import os, json, subprocess
    assert which in ["exe", "ext", ""]

    if which in ["exe", "ext"]:
        path = BUILD_DIR / f"compile_commands_{which}.json"
        with open(path, "w") as f:
            subprocess.run(
                ["ninja", "-t", "compdb"],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=BUILD_DIR,
                check=True,
                env=os.environ.copy(),
            )

    combined = []
    for which in ["exe", "ext"]:
        path = BUILD_DIR / f"compile_commands_{which}.json"
        if not path.is_file():
            continue
        with open(path, "r") as f:
            combined.extend(json.load(f))

    path = BUILD_DIR / "compile_commands.json"
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)

    if verbose:
        print("Generated '.build/compile_commands.json' for IDE code integrations.")
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "which", choices=["exe", "executable", "ext", "extension", "both"], nargs="?", default="both"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-c", "--clean", action="store_true")
    args = parser.parse_args()

    print("Compile liblinrec into")
    make_build_dir(clean=args.clean)
    if args.which in ["exe", "executable", "both"]:
        exe = executable(verbose=args.verbose)
        print("  - a C++ executable: .build/cpplinrec")
    
    if args.which in ["ext", "extension", "both"]:
        ext = extension(verbose=args.verbose)
        print("  - a Python extension: .build/pylinrec.so")
