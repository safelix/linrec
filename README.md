A CUDA extension for PyTorch to compute the [linear recurrence](https://en.wikipedia.org/wiki/Linear_recurrence_with_constant_coefficients) $y_l =  y_{l-1} \cdot c_l + x_l$ as efficiently as a binary operator:
```python
import torch, linrec
kwargs = dict(size=(512, 2**16), device='cuda')
inputs, coeffs = torch.randn(**kwargs), torch.rand(**kwargs)
outputs = linrec.linrec(inputs, coeffs, reverse=False)
```

## Repository Structure
```
├── eval
│   ├── add_linrec_to_path.py                           # add root to path if not installed
│   ├── bench.py                                        # benchmark implementations
│   ├── debug.py                                        # call/debug kernels
│   ├── test.py                                         # test implementations
│   ├── tune.py                                         # tune configurations
│   └── utils.py                                        # eval utilities
├── linrec
│   ├── impl
│   │   ├── cuda
│   │   │   ├── build.py                                # build system
│   │   │   ├── cuhelpers.cuh                           # various cuda helpers
│   │   │   ├── dispatch.h                              # dispatch features
│   │   │   ├── executable.cpp                          # standalone executable
│   │   │   ├── extension.cpp                           # pytorch extension
│   │   │   ├── linrec.h                                # header files
│   │   │   ├── linrec_pipe.cu                          # pipe host-side function
│   │   │   ├── linrec_pipe.cuh                         # pipe kernel implementation
│   │   │   ├── linrec_ref.cu                           # reference host-side function
│   │   │   ├── linrec_ref.cuh                          # reference kernel implementation
│   │   │   ├── linrec_tile.cu                          # tile host-side function 
│   │   │   ├── linrec_tile.cuh                         # tile kernel implementation
│   │   │   ├── memio.cuh                               # memory loading 
│   │   │   └── ops.py                                  # cuda ops interface
│   │   └── python
│   │       └── ops.py                                  # python ops interface
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.py
```

## Setup

For an installation without CUDA implementation, no build environment is required. 

### Build Tools

Installing the CUDA Toolkit can be a bit tricky and is explained in the [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux). For this specific project we have the following requirements:
- CUDA requires `gxx<=13.2`, as explained [here](https://deocs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy) and [here](https://gist.github.com/ax3l/9489132).
- C++20 with `static constexpr` requires `gxx>=12.1`.
- triton/torch.compile requires `python>=3.9,<3.13`.
- the latest [PyTorch](https://pytorch.org/) is built with `cuda-runtime==12.4` (can be ignored)

We follow the [Conda installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#conda-installation) to setup a build environment:
```
conda create -n CUDA12.4 -c conda-forge gxx==13.2 python==3.12 nvidia::cuda==12.4
```
If you see the error `InvalidSpec: The package "cuda==XX" is not available for the specified platform`, install `gxx` first and then `cuda`. The exact compiler requirements for a given CUDA installation are in `.../include/crt/host_config.h`. Note that [`nvidia::cuda`](https://anaconda.org/nvidia/cuda) includes packages from [`nvidia::cuda-runtime`](https://anaconda.org/nvidia/cuda-runtime), which are additionally installed through `pip` as dependencies of `torch` (see [Section on Pip Wheels in the Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pip-wheels)).


### User Installation
Once the build environment is ready, you can simply build and install `linrec` with 
```
pip install git+https://github.com/safelix/linrec.git
```

### Developer Installation
For evals and lightweight development, a quick editable install is available (clone then `pip install -e linrec[eval]`). For C++/CUDA development, accessing the code directly without installation provides more finegrained control over the compilation process (make sure to `pip uninstall linrec`). Let's get started and run the eval suite:
``` 
git clone git@github.com:safelix/linrec.git
pip install -r linrec/requirements.txt
python linrec/eval/test.py
python linrec/eval/bench.py
``` 

This automatically compiles the extension and stores all the build files in the `.build` directory. The C++/CUDA build process is controled from `linrec/impl/cuda/build.py`. For example, calling `_C = linrec.impl.cuda.build.extension()` dynamically loads the extension into the Python runtime and triggers recompilation if the C++/CUDA source files changed. Its commandline interface `python -m linrec.impl.cuda.build` allows to force clean re-builds, show compilation outputs and compile a light-weight standalone executable for C++/CUDA debugging or profiling.

**Tipps:** Link the `.build/compile_commands.json` to your IDE for C++/CUDA linting and code integrations. Set the environment variable `MAX_JOBS` to enable parallel compliation. Demangle function names in the compilation outputs with `[build command] | cu++filt`.


### Profiling

To use NSight Compute, install it on your local machine and use its `Remote Launch` feature to start a profiling activity via `SSH`. You can also install it on a server using `conda install nvidia::nsight-compute`, but this requires a workaround. For this, open obtain the paths `which ncu` and `which ncu-ui`, open the bash scripts and insert in line 47:
```
# WORKAROUND
# If installed with cuda, nsight-compute tools will be under nsight-compute/<version> folder. e.g nsight-compute/2019.4.0
for nsight_compute_tool_dir_path in "$CUDA_TOOLKIT_BIN_DIR"/../nsight-compute/*; do
    if [ ! -e "$nsight_compute_tool_dir_path" ]; then
        # Glob didn't match anything. Let's skip this single iteration.
        continue
    fi
    setLatestNsightComputeToolDir "$(basename "$nsight_compute_tool_dir_path")" "$nsight_compute_tool_dir_path"
done
```



## Other Projects:
- Associative Scan Interfaces in CUB: [Device-Level Scan](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html), [Block-Level Scan](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockScan.html), [Warp-Level Scan](https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockScan.html)
- Warp-level associative scan implementation in triton: [github.com/triton-lang/triton](https://github.com/triton-lang/triton/blob/7480ef5028b724cb434b7841b016c6d6debf3b84/lib/Conversion/TritonGPUToLLVM/ScanOpToLLVM.cpp#L77)
- Device and warp-level associative scan operations in pytorch: [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch/blob/main/torch/_higher_order_ops/associative_scan.py)
- Warp-level associative scan implementation by Volodymyr Kyrylov: [github.com/proger/accelerated-scan](https://github.com/proger/accelerated-scan)
- Device-level associative scan implementation by Alexandre TL: [github.com/alxndrTL/mamba.py](https://github.com/alxndrTL/mamba.py/blob/main/mambapy/pscan.py)
- Associative scan implementations by John Ryan: [github.com/johnryan465/pscan](https://github.com/johnryan465/pscan)

