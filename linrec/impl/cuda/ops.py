import torch
from typing import Tuple
from types import NoneType
from torch.autograd.function import FunctionCtx
try:
    from ... import _C # CUDAExtension via setup.py
except ImportError:
   from .build import extension
   _C = extension()
   
__all__ = ["linrec_ref", "linrec_tile", "linrec_pipe"]
_C.config_names = _C.config_names                              # works with from .ops import _C
_C.config_list = list({tuple(c):None for c in _C.config_list}) # make config_list unique using ordered dict

# Requirements for setup_context signature are descriped here:
# https://pytorch.org/docs/stable/library.html#torch.library.register_autograd
def setup_context(ctx:FunctionCtx, inputs, output):
    inputs, coeffs, reverse = inputs
    outputs = output # named argument must be 'output'
    ctx.save_for_backward(coeffs, outputs)
    ctx.reverse = reverse

def fake_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
    assert inputs.size() == coeffs.size()       # same dimensions
    assert inputs.stride() == coeffs.stride()   # same strides
    assert inputs.device == coeffs.device       # same device
    assert inputs.is_cuda == coeffs.is_cuda     # both cuda
    assert inputs.dtype == coeffs.dtype         # same dtype
    assert inputs.stride(-1) == 1               # inner most dimension is last 
    return torch.empty_like(inputs)

def fake_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
    assert d_outputs.size() == coeffs.size() and coeffs.size() == outputs.size()            # same dimensions
    assert d_outputs.stride() == coeffs.stride() and coeffs.stride() == outputs.stride()    # same strides
    assert d_outputs.device == coeffs.device and coeffs.device == outputs.device            # same device
    assert d_outputs.is_cuda and coeffs.is_cuda and outputs.is_cuda                         # both cuda
    assert d_outputs.dtype == coeffs.dtype and coeffs.dtype == outputs.dtype                # same dtype
    assert d_outputs.stride(-1) == 1                                                        # inner most dimension is last 
    return torch.empty_like(d_outputs), torch.empty_like(coeffs)



# CUDA Reference Implementations
@torch.library.custom_op("linrec::curef_fwd", mutates_args=(), device_types='cuda')
def linrec_ref_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
    return _C.linrec_ref_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)

@torch.library.custom_op("linrec::curef_bwd", mutates_args=(), device_types='cuda')
def linrec_ref_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    return _C.linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=reverse)

linrec_ref_fwd.register_fake(fake_fwd)
linrec_ref_bwd.register_fake(fake_bwd)
def backward(ctx:FunctionCtx, d_outputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, NoneType]:
    d_outputs = d_outputs if d_outputs.stride(-1) == 1 else d_outputs.contiguous()  # user has no control
    coeffs, outputs = ctx.saved_tensors
    d_inputs, d_coeffs = linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
    return d_inputs, d_coeffs, None
torch.library.register_autograd("linrec::curef_fwd", backward, setup_context=setup_context)
linrec_ref = linrec_ref_fwd



# CUDA Tiled Implementations
@torch.library.custom_op("linrec::cutile_fwd", mutates_args=(), device_types='cuda')
def linrec_tile_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
    return _C.linrec_tile_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)

@torch.library.custom_op("linrec::cutile_bwd", mutates_args=(), device_types='cuda')
def linrec_tile_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    return _C.linrec_tile_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=reverse)

linrec_tile_fwd.register_fake(fake_fwd)
linrec_tile_bwd.register_fake(fake_bwd)
def backward(ctx:FunctionCtx, d_outputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, NoneType]:
    d_outputs = d_outputs if d_outputs.stride(-1) == 1 else d_outputs.contiguous()  # user has no control
    coeffs, outputs = ctx.saved_tensors
    d_inputs, d_coeffs = linrec_tile_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
    return d_inputs, d_coeffs, None
torch.library.register_autograd("linrec::cutile_fwd", backward, setup_context=setup_context)
linrec_tile = linrec_tile_fwd


# CUDA Piped Implementations
@torch.library.custom_op("linrec::cupipe_fwd", mutates_args=(), device_types='cuda')
def linrec_pipe_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
    return _C.linrec_pipe_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)

@torch.library.custom_op("linrec::cupipe_bwd", mutates_args=(), device_types='cuda')
def linrec_pipe_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False) -> Tuple[torch.Tensor, torch.Tensor]:
    return _C.linrec_pipe_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=reverse)

linrec_pipe_fwd.register_fake(fake_fwd)
linrec_pipe_bwd.register_fake(fake_bwd)
def backward(ctx:FunctionCtx, d_outputs:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, NoneType]:
    d_outputs = d_outputs if d_outputs.stride(-1) == 1 else d_outputs.contiguous()  # user has no control
    coeffs, outputs = ctx.saved_tensors
    d_inputs, d_coeffs = linrec_pipe_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
    return d_inputs, d_coeffs, None
torch.library.register_autograd("linrec::cupipe_fwd", backward, setup_context=setup_context)
linrec_pipe = linrec_pipe_fwd



# https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html
#
# Q: When should I create a Custom Operator?
# If your operation is expressible as a composition of built-in PyTorch operators then please
# write it as a Python function and call it instead of creating a custom operator. Use the 
# operator registration APIs to create a custom operator if you are calling into some library
# that PyTorch doesn’t understand (e.g. custom C/C++ code, a custom CUDA kernel, or Python
# bindings to C/C++/CUDA extensions).
#
# Q: Why should I create a Custom Operator?
# A: It is possible to use a C/C++/CUDA kernel by grabbing a Tensor’s data pointer and passing 
# it to a pybind’ed kernel. However, this approach doesn’t compose with PyTorch subsystems 
# like autograd, torch.compile, vmap, and more. In order for an operation to compose with 
# PyTorch subsystems, it must be registered via the operator registration APIs.
