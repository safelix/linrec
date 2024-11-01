import torch
from torch.autograd.function import FunctionCtx
try:
    from ... import _C # via setup.py
except ImportError:
   from .build import extension
   _C = extension()
   
__all__ = ["linrec_ref", "linrec_tile", "linrec_pipe"]
config_names = _C.config_names
config_list = list({tuple(c):None for c in _C.config_list}) # make config_list unique using ordered dict


# CUDA Reference Implementations
def linrec_ref_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False):
    return _C.linrec_ref_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)

def linrec_ref_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False):
    return _C.linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=reverse)

class LinrecRefFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_ref_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_ref(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecRefFn.apply(inputs, coeffs, reverse)



# CUDA Tiled Implementations
def linrec_tile_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False, **config):
    return _C.linrec_tile_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse, **config)

def linrec_tile_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False, **config):
    return _C.linrec_tile_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=reverse, **config)

def linrec_tile_attrs(fwd:bool, **config):
    return _C.linrec_tile_attrs(fwd, **config)

class LinrecTileFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_tile_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_tile_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_tile(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecTileFn.apply(inputs, coeffs, reverse)



# CUDA Piped Implementations
def linrec_pipe_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False, **config):
    return _C.linrec_pipe_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse, **config)

def linrec_pipe_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse:bool=False, **config):
    return _C.linrec_pipe_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=reverse, **config)

def linrec_pipe_attrs(fwd:bool, **config):
    return _C.linrec_pipe_attrs(fwd, **config)

class LinrecPipeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_pipe_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_pipe_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_pipe(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecPipeFn.apply(inputs, coeffs, reverse)



# TODO: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#why-should-i-create-a-custom-operator
# Q: Why should I create a Custom Operator?
# A: It is possible to use a C/C++/CUDA kernel by grabbing a Tensor’s data pointer and passing 
# it to a pybind’ed kernel. However, this approach doesn’t compose with PyTorch subsystems 
# like autograd, torch.compile, vmap, and more. In order for an operation to compose with 
# PyTorch subsystems, it must be registered via the operator registration APIs.
