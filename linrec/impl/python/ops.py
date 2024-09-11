import torch
from torch.autograd.function import Function, FunctionCtx
from torch._higher_order_ops.associative_scan import associative_scan

__all__ = ["linrec_ref, linrec_hop"]

### Eager Reference Implementations
def linrec_fwd_ref(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    outputs = torch.zeros_like(inputs)
    if not reverse:
        outputs[..., 0] = inputs[..., 0]
        for i in range(1, inputs.shape[-1], 1):
            outputs[..., i] = outputs[..., i-1].clone() * coeffs[..., i] + inputs[..., i]
    else:
        outputs[..., -1] = inputs[..., -1]
        for i in range(inputs.shape[-1] - 1, 0, -1):
            outputs[..., i-1] = outputs[..., i].clone() * coeffs[..., i-1] + inputs[..., i-1]
    return outputs


def linrec_bwd_ref(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    d_inputs = linrec_fwd_ref(inputs=d_outputs, coeffs=coeffs, reverse=(not reverse))

    d_coeffs = torch.zeros_like(coeffs)
    if not reverse:
        d_coeffs[..., 1:] = outputs[..., :-1] * d_inputs[..., 1:]
    else:
        d_coeffs[..., :-1] = outputs[..., 1:] * d_inputs[..., :-1]
    
    return d_inputs, d_coeffs


class LinrecRefFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_fwd_ref(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_bwd_ref(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None


def linrec_ref(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecRefFn.apply(inputs, coeffs, reverse)



### Eager Higher-Order Op Implementations
def scan_op(acc:tuple, curr:tuple):
    accOut, accCoeff = acc
    currInp, currCoeff = curr
    return accOut * currCoeff + currInp, accCoeff * currCoeff

def linrec_fwd_hop(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    if reverse:
        inputs, coeffs = inputs.flip(dims=[-1]), coeffs.flip(dims=[-1])
    outputs, _ = associative_scan(scan_op, (inputs, coeffs), dim=-1)
    if reverse:
        outputs = outputs.flip(dims=[-1])
    return outputs

def linrec_bwd_hop(d_outputs:torch.Tensor, coeffs:torch.Tensor, outputs:torch.Tensor, reverse=False):
    d_inputs = linrec_fwd_hop(inputs=d_outputs, coeffs=coeffs, reverse=(not reverse))

    d_coeffs = torch.zeros_like(coeffs)
    if not reverse:
        d_coeffs[..., 1:] = outputs[..., :-1] * d_inputs[..., 1:]
    else:
        d_coeffs[..., :-1] = outputs[..., 1:] * d_inputs[..., :-1]
    
    return d_inputs, d_coeffs


class LinrecHopFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_fwd_hop(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_bwd_hop(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None


def linrec_hop(inputs:torch.Tensor, coeffs:torch.Tensor, reverse=False):
    return LinrecHopFn.apply(inputs, coeffs, reverse)

