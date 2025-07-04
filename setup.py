# Template from: https://github.com/pytorch/extension-cpp/blob/master/setup.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys, warnings
from pathlib import Path
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

library_name = "linrec"


def get_extensions():
    if CUDA_HOME is None:
        warnings.warn("NVCC not found, linrec.impl.cuda will not be installed.")
        return []

    # build.py is also a module: python -m linrec.impl.cuda.build
    sys.path.insert(0, str(Path(__file__).parent / library_name / "impl" / "cuda"))
    from build import LIB_SOURCES, EXT_SOURCES, INCLUDES, CPP_FLAGS, CUDA_FLAGS

    ext = CUDAExtension(
        name=f"{library_name}._C",
        sources=LIB_SOURCES + EXT_SOURCES,
        include_dirs=INCLUDES,
        extra_compile_args=dict(nvcc=CUDA_FLAGS, cxx=CPP_FLAGS),
    )
    return [ext]


# NVIDIA provides Python Wheels for installing CUDA through pip, primarily for using CUDA
# with Python. These packages are intended for runtime use through distributed with torch.
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pip-wheels
install_requires = ["torch", "nvidia-cuda-runtime-cu12"]
extras_require = dict(build=["ninja"], eval=["triton", "numpy", "pandas", "matplotlib"])
extras_require["all"] = [dep for group in extras_require.values() for dep in group]

setup(
    name=library_name,
    description="Example of PyTorch cpp and CUDA extensions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/safelix/linrec",
    author="Felix Sarnthein",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9,<3.13", # for pytorch/triton
    install_requires=install_requires,
    extras_require=extras_require,
)
