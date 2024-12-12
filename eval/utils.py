import torch
import warnings
import triton.testing

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


def meminit(seqlen, n_batches=1, n_channels=1, dtype=None, device=None, seed=None, grad=None, fwd=None):
    size = (n_batches, n_channels, seqlen)

    device = torch.device('cuda') if device is None else device
    generator = torch.Generator(device=device)
    if seed is not None:
        generator = generator.manual_seed(seed)

    inputs = torch.randn(size=size, dtype=dtype, device=device, requires_grad=True, generator=generator)
    coeffs = torch.rand(size=size, dtype=dtype, device=device, requires_grad=True)

    if grad in [None, False]:
        return inputs, coeffs
    
    d_outputs = torch.randn(size=size, dtype=dtype, device=device, requires_grad=False)

    if grad == 'bwd' and callable(fwd):
        return d_outputs, coeffs, fwd(inputs, coeffs)               # arguments for _C.linrec_*_bwd()
    
    if grad == 'autograd':
        return (inputs, coeffs), d_outputs                          # arguments for torch.autograd.grad()
    
    raise ValueError('Specify grad as \'autograd\' or \'bwd\' to compute gradients of the function fwd.')


def memio(stmt, data):
    outputs = stmt(*data)
    data += (outputs, ) if isinstance(outputs, torch.Tensor) else tuple(outputs)
    return sum(t.numel() * t.element_size() for t in data if isinstance(t, torch.Tensor))

def memio_limit(seqlen, n_batches=1, n_channels=1, dtype=None, device=None, seed=None, grad=None, fwd=None):
    #size = (n_batches, n_channels, seqlen)
    #tensor = torch.randn(size=size, dtype=dtype, device=device)
    #bytes = (3 if fwd else 5) * tensor.numel() * tensor.element_size()
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.get_default_device() if device is None else device

    size = n_batches * n_channels * seqlen
    bytes = (3 if grad in [None, False] else 5) * size * dtype.itemsize

    bandwidth = triton.testing.get_dram_gbps(device=device.index)  # GB/s
    ms = (bytes * 1e-9) / (bandwidth) * 1e3 
    return ms, bytes


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, quantiles=None, return_mode="mean", device=None):
    """
    Benchmark the runtime of the provided function in miliseconds. By default, return the median runtime of 
    :code:`fn` along with the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", or "median". Default is "mean".
    :type return_mode: str
    """
    import numpy as np
    with torch.cuda.device(device):
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep, grad_to_none=grad_to_none, quantiles=quantiles, return_mode=return_mode)
    return np.array(ms)