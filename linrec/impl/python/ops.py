import torch
from torch.autograd.function import Function, FunctionCtx
from torch._higher_order_ops.associative_scan import associative_scan

__all__ = ["linrec_ref, linrec_hop"]

### Eager Reference Implementations
def linrec_ref_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    outputs = torch.zeros_like(inputs)
    prev = torch.zeros_like(outputs[..., 0])

    for i in range(0, inputs.shape[-1])[::-1 if reverse else 1]:
        outputs[..., i] = prev * coeffs[..., i] + inputs[..., i]
        prev = outputs[..., i].clone()
        
    return outputs

def shift(input, shifts, fillval=0):
    # torch.roll without the copy of the wrap-around section
    if shifts > 0:
        output = torch.cat([torch.full_like(input[..., :shifts], fillval), input[..., :-shifts]], dim=-1)
    if shifts < 0:
        output = torch.cat([input[..., -shifts:], torch.full_like(input[..., shifts:], fillval)], dim=-1)
    return output

def linrec_ref_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    coeffs = shift(coeffs, -1 if not reverse else 1, fillval=0)
    d_inputs = linrec_ref_fwd(inputs=d_outputs, coeffs=coeffs, reverse=(not reverse))
    d_coeffs =  d_inputs * shift(outputs, shifts=1 if not reverse else -1, fillval=0)
    return d_inputs, d_coeffs


class LinrecRefFn(Function):
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



### Eager Higher-Order Op Implementations
def scan_op(acc:tuple, curr:tuple):
    accOut, accCoeff = acc
    currInp, currCoeff = curr
    return accOut * currCoeff + currInp, accCoeff * currCoeff

def linrec_hop_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    outputs, _ = associative_scan(scan_op, (inputs, coeffs), dim=-1, reverse=reverse)
    return outputs

def linrec_hop_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    coeffs = shift(coeffs, -1 if not reverse else 1, fillval=0)
    d_inputs = linrec_hop_fwd(inputs=d_outputs, coeffs=coeffs, reverse=(not reverse))
    d_coeffs =  d_inputs * shift(outputs, shifts=1 if not reverse else -1, fillval=0)
    return d_inputs, d_coeffs


class LinrecHopFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_hop_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_hop_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None


def linrec_hop(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecHopFn.apply(inputs, coeffs, reverse)

