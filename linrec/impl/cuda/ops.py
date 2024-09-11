import torch
from torch.autograd.function import FunctionCtx
try:
    from ... import _C # via setup.py
except ImportError:
   from .build import extension
   _C = extension()
   
__all__ = ["linrec_ref", "linrec_tile"]

class LinrecRefFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = _C.linrec_ref_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = _C.linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_ref(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecRefFn.apply(inputs, coeffs, reverse)


class LinrecTileFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = _C.linrec_tile_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = _C.linrec_tile_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_tile(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecTileFn.apply(inputs, coeffs, reverse)



# TODO: https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#why-should-i-create-a-custom-operator
# Q: Why should I create a Custom Operator?
# A: It is possible to use a C/C++/CUDA kernel by grabbing a Tensor’s data pointer and passing 
# it to a pybind’ed kernel. However, this approach doesn’t compose with PyTorch subsystems 
# like autograd, torch.compile, vmap, and more. In order for an operation to compose with 
# PyTorch subsystems, it must be registered via the operator registration APIs.
