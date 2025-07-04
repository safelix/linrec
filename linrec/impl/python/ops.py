import torch
from torch.autograd.function import Function, FunctionCtx
from torch._higher_order_ops.associative_scan import associative_scan

__all__ = ["linrec_ref, linrec_hop"]

### Eager Reference Implementations
def linrec_ref_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, logc=False, dim=-1, reverse=False):
    inputs, coeffs = inputs.transpose(dim, -1), coeffs.transpose(dim, -1)
    outputs = torch.zeros_like(inputs)
    prev = torch.zeros_like(outputs[..., 0])

    if logc:
        coeffs = torch.exp(coeffs)

    for i in range(0, inputs.shape[-1])[::-1 if reverse else 1]:
        outputs[..., i] = prev * coeffs[..., i] + inputs[..., i]
        prev = outputs[..., i].clone()
        
    return outputs.transpose(-1, dim)

def shift(input:torch.Tensor, shift, dim=0, fillval=0):
    # torch.roll without the copy of the wrap-around section
    size = input.size(dim)
    fill = torch.full_like(input.narrow(dim, 0, abs(shift)), fillval)
    if shift > 0:
        output = torch.cat([fill, input.narrow(dim, 0, size-shift)], dim=dim)
    if shift < 0:
        output = torch.cat([input.narrow(dim, -shift, size+shift), fill], dim=dim)
    return output

def linrec_ref_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, logc=False, dim=-1, reverse=False):
    coeffs = shift(coeffs, shift=(-1 if not reverse else 1), dim=dim, fillval=0)
    d_inputs = linrec_ref_fwd(inputs=d_outputs, coeffs=coeffs, logc=logc, dim=dim, reverse=(not reverse))
    d_coeffs =  d_inputs * shift(outputs, shift=(1 if not reverse else -1), dim=dim, fillval=0)

    if logc:
        d_coeffs *= torch.exp(coeffs)

    return d_inputs, d_coeffs


class LinrecRefFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, logc=False, dim=-1, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_ref_fwd(inputs=inputs, coeffs=coeffs, logc=logc, dim=dim, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.logc = logc
        ctx.dim = dim
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, logc=ctx.logc, dim=ctx.dim, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None, None, None


def linrec_ref(inputs:torch.Tensor, coeffs:torch.Tensor, logc=False, dim=-1, reverse=False):
    return LinrecRefFn.apply(inputs, coeffs, logc, dim, reverse)



### Eager Higher-Order Op Implementations
def linrec_hop_fwd(inputs:torch.Tensor, coeffs:torch.Tensor, logc=False, dim=-1, reverse=False):

    def op(acc:dict, curr:dict):
        x = curr['c'] * acc['x']  + curr['x']
        c = acc['c'] * curr['c']
        return dict(x=x, c=c)
    
    def logop(acc:dict, curr:dict):
        x = torch.exp(curr['c']) * acc['x']  + curr['x']
        c = acc['c'] + curr['c']
        return dict(x=x, c=c)
    
    outputs = associative_scan(logop if logc else op, dict(x=inputs, c=coeffs), dim=dim, reverse=reverse)['x']
    return outputs

def linrec_hop_bwd(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, logc=False, dim=-1, reverse=False):
    coeffs = shift(coeffs, shift=(-1 if not reverse else 1), dim=dim, fillval=0)
    d_inputs = linrec_hop_fwd(inputs=d_outputs, coeffs=coeffs, logc=logc, dim=dim, reverse=(not reverse))
    d_coeffs =  d_inputs * shift(outputs, shift=(1 if not reverse else -1), dim=dim, fillval=0)

    if logc:
        d_coeffs = torch.exp(coeffs) * d_coeffs

    return d_inputs, d_coeffs


class LinrecHopFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, logc=False, dim=-1, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_hop_fwd(inputs=inputs, coeffs=coeffs, logc=logc, dim=dim, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.logc = logc
        ctx.dim = dim
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_hop_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, logc=ctx.logc, dim=ctx.dim, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None, None, None


def linrec_hop(inputs:torch.Tensor, coeffs:torch.Tensor, logc=False, dim=-1, reverse=False):
    return LinrecHopFn.apply(inputs, coeffs, logc, dim, reverse)