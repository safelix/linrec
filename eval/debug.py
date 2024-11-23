import torch
from torch.autograd import grad
import matplotlib.pyplot as plt

import add_linrec_to_path
from linrec.impl.cuda.ops import _C as cuops

dtype = torch.float32
device = torch.device('cuda:0')
kwargs = dict(reverse=False, kMaxElemsPerThread=4, kMaxThreadsPerBlock=32, memcode=0, algocode=3)
print(cuops.linrec_pipe_attrs(fwd=False, **kwargs))

shape = (140*100, 16 * 1024)
inputs = 1 * torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
coeffs = 1 * torch.rand(shape, dtype=dtype, device=device, requires_grad=True)
d_outputs = 1 * torch.randn(shape, dtype=dtype, device=device, requires_grad=False)

outputs = cuops.linrec_pipe_fwd(inputs=inputs, coeffs=coeffs, **kwargs)
d_inputs, d_coeffs = cuops.linrec_pipe_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, **kwargs)

outputs_ref = cuops.linrec_ref_fwd(inputs=inputs, coeffs=coeffs, reverse=kwargs['reverse'])
d_inputs_ref, d_coeffs_ref =  cuops.linrec_ref_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs_ref, reverse=kwargs['reverse'])


print(f'err outputs: {(outputs - outputs_ref).abs().max().item():.5e}')
print(f'err d_inputs: {(d_inputs - d_inputs_ref).abs().max().item():.5e}')
print(f'err d_coeffs: {(d_coeffs - d_coeffs_ref).abs().max().item():.5e}')

err = (outputs - outputs_ref).abs().max(0).values
plt.plot(err.cpu(), label='outputs: err.max()')

err = (d_inputs - d_inputs_ref).abs().max(0).values
plt.plot(err.cpu(), label='d_inputs: err.max()')

err = (d_coeffs - d_coeffs_ref).abs().max(0).values
plt.plot(err.cpu(), label='d_coeffs: err.max()')

plt.gca().set_yscale('symlog', linthresh=1e-6)
plt.legend()
plt.show()