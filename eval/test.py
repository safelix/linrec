import torch
from functools import partial
from utils import execption2nan, meminit

import add_linrec_to_path
from linrec.impl.cuda import ops as cuops
from linrec.impl.triton import ops as ttops
from linrec.impl.python import ops as pyops

def test(stmt, ref, atol=None, **memargs):
        
    data = meminit(**memargs)
    if memargs['grad'] == 'autograd':
        data, grad = data
    
    out, sol = stmt(*data), ref(*data)
    if memargs['grad'] == 'autograd':
        out = torch.autograd.grad(out, data, grad)
        sol = torch.autograd.grad(sol, data, grad)

    out = out if isinstance(out, torch.Tensor) else torch.stack(out)
    sol = sol if isinstance(sol, torch.Tensor) else torch.stack(sol)

    diff = (sol - out).abs().max().item()
    if atol is not None:
        diff = torch.allclose(sol, out, atol=atol)

    # overwrite memory to mitigate risk of spurious leaks between tests
    out.fill_(torch.nan), sol.fill_(torch.nan) 
    return diff


if __name__ == '__main__':
    from argparse import ArgumentParser
    from triton.testing import Benchmark, Mark

    parser = ArgumentParser()
    parser.add_argument('--ref', choices=['pyref_fwd', 'pyref', 'curef'], default='pyref_fwd', help='Reference implementation to test against.')
    parser.add_argument('--grad', choices=['autograd'], default=None, help='If and how to test gradient computation.')
    parser.add_argument('--seqlen_min', type=int, default=4, help='Minimal sequence length to test with (log2).')
    parser.add_argument('--seqlen_max', type=int, default=10, help='Maximal sequence length to test with (log2).')
    parser.add_argument('--n_batches', type=int, default=-1, help='Number of batches to test with (-1 = Shared Multiprocessors).')
    parser.add_argument('--n_channels', type=int, default=100, help='Number of channels to test with.')
    parser.add_argument('--reverse', action='store_true', help='Use reverse scan to test with.')
    parser.add_argument('--seed', type=int, default=12334567890, help='Seed to generate data with.')
    parser.add_argument('--device', type=int, default=0, help='Device id to test on.')
    parser.add_argument('--csv', action='store_true', help='Store results as CSV.')
    parser.add_argument('--showplots', action='store_true', help='Whether to show plots.')
    args = parser.parse_args()

    # Prepare Data Arguments
    args.device = torch.device(f'cuda:{args.device}')
    sm = torch.cuda.get_device_properties(args.device).multi_processor_count

    seqlens = [2**i for i in range(args.seqlen_min, args.seqlen_max+1)]
    memargs = dict(n_batches=args.n_batches if (args.n_batches > 0) else sm,
                n_channels=args.n_channels,
                dtype=torch.float32,
                device=args.device,
                seed=args.seed, 
                grad=args.grad)
    
    # Prepare Statements
    stmts = {'pyref': pyops.linrec_ref,'pyhop': pyops.linrec_hop,  
             'tttile': ttops.linrec_tile, 'ttpipe': ttops.linrec_pipe,
             'cutile': cuops.linrec_tile, 'cupipe': cuops.linrec_pipe, 'curef': cuops.linrec_ref}
    stmts = {key:partial(stmt, reverse=args.reverse) for key, stmt in stmts.items()}

    ref = partial(pyops.linrec_ref_fwd, reverse=args.reverse) 
    ref = ref if (args.ref=='pyref_fwd') else stmts[args.ref]

    # Execute one statment
    # print(bench(stmts['cupipe'], ref=ref **dict(seqlen=seqlens[0], **memargs)))

    # Run Benchmark
    benchmark = Benchmark(
        x_names=["seqlen"],  # argument names to use as an x-axis for the plot
        x_vals=seqlens,
        line_arg="stmt",  # argument name whose value corresponds to a different line in the plot
        line_names=list(stmts.keys()),
        line_vals=list(stmts.values()),
        plot_name=f'Test {'forward' if not args.grad else 'backward'}',
        args=memargs,
        xlabel='sequence length',
        x_log=True,
        y_log=True,
    )

    test = execption2nan(warn=True)(test)
    Mark(test, benchmark).run(show_plots=args.showplots, print_data=True, ref=ref)