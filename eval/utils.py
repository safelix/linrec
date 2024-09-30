import torch
import warnings
from triton.testing import get_dram_gbps


import add_linrec_to_path
from linrec.impl.cuda import build
_C = build.extension()


def execption2nan(warn=False):
    def decorator(func):
        def wrappedfunc(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if warn:
                    warnings.warn(RuntimeWarning(*e.args))
                return float('nan')
        return wrappedfunc
    return decorator


def meminit(seqlen, n_batches=1, n_channels=1, fwd=True, dtype=None, device=None, seed=None):
    size = (n_batches, n_channels, seqlen)

    device = torch.device('cuda') if device is None else device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    inputs = torch.randn(size=size, dtype=dtype, device=device, generator=generator)
    coeffs = 0.99 * torch.ones(size=size, dtype=dtype, device=device)

    if fwd:
        return inputs, coeffs

    outputs = _C.linrec_ref_fwd(inputs, coeffs)
    d_outputs = 0.99 * torch.ones(size=size, dtype=dtype, device=device, requires_grad=False)
    return d_outputs, coeffs, outputs

def memio(stmt, data):
    outputs = stmt(*data)
    data += (outputs, ) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    return sum(t.numel() * t.element_size() for t in data if isinstance(t, torch.Tensor))

def memio_limit(seqlen, n_batches=1, n_channels=1, fwd=True, dtype=None, device=None, seed=None):
    #size = (n_batches, n_channels, seqlen)
    #tensor = torch.randn(size=size, dtype=dtype, device=device)
    #bytes = (3 if fwd else 5) * tensor.numel() * tensor.element_size()
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.get_default_device() if device is None else device

    size = n_batches * n_channels * seqlen
    bytes = (3 if fwd else 5) * size * dtype.itemsize

    bandwidth = get_dram_gbps(device=device.index)  # GB/s
    ms = (bytes * 1e-9) / (bandwidth) * 1e3 
    return ms, bytes