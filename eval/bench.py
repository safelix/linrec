import torch
from functools import partial
from utils import execption2nan, meminit, memio, memio_limit, do_bench

import add_linrec_to_path
from linrec.impl.cuda import ops as cuops
from linrec.impl.python import ops as pyops


def bench(stmt, throughput=False, **memargs):
    device = memargs.get('device', None)

    if stmt in [memio_limit, 'memio_limit']:
        ms, bytes = memio_limit(**memargs)
        return bytes / ms * 1e-6 if throughput else ms

    data = meminit(**memargs)
    if memargs['grad'] == 'autograd':
        data = (stmt(*data[0]), data[0], data[1])
        stmt = partial(torch.autograd.grad, retain_graph=True)

    ms = do_bench(lambda: stmt(*data), quantiles=[0, 0.5, 1], device=device)#, warmup=50, rep=1000)
    bytes = memio(stmt, data)

    if throughput: # throughput
        return (bytes * 1e-9) / (ms * 1e-3) # GB/s
    return ms



if __name__ == '__main__':
    from argparse import ArgumentParser
    from triton.testing import Benchmark, Mark

    parser = ArgumentParser()
    parser.add_argument('--grad', choices=['autograd', 'bwd'], default=None, help='If and how to benchmark gradient computation.')
    parser.add_argument('--seqlen_min', type=int, default=4, help='Minimal sequence length to benchmark with (log2).')
    parser.add_argument('--seqlen_max', type=int, default=16, help='Maximal sequence length to benchmark with (log2).')
    parser.add_argument('--n_batches', type=int, default=-1, help='Number of batches to benchmark with (-1 = Shared Multiprocessors).')
    parser.add_argument('--n_channels', type=int, default=100, help='Number of channels to benchmark with.')
    parser.add_argument('--reverse', action='store_true', help='Use reverse scan to benchmark with.')
    parser.add_argument('--throughput', action='store_true', help='Use throughput as a metric.')
    parser.add_argument('--seed', type=int, default=12334567890, help='Seed to generate data with.')
    parser.add_argument('--device', type=int, default=0, help='Device id to benchmark on.')
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
                grad=args.grad,
                fwd=cuops.linrec_ref_fwd if args.grad == 'bwd' else None)
    
    # Prepare Statements
    stmts = {'pyref': pyops.linrec_ref if (args.grad!='bwd') else pyops.linrec_ref_bwd,
             'pyhop': pyops.linrec_hop if (args.grad!='bwd') else pyops.linrec_hop_bwd,  
             'curef': cuops.linrec_ref if (args.grad!='bwd') else cuops.linrec_ref_bwd,  
             'cutile': cuops.linrec_tile if (args.grad!='bwd') else cuops.linrec_tile_bwd,  
             'cupipe': cuops.linrec_pipe if (args.grad!='bwd') else cuops.linrec_pipe_bwd}
    stmts = {key:partial(stmt, reverse=args.reverse) for key, stmt in stmts.items()}
    stmts['memio_limit'] = memio_limit
    
    # Execute one statment
    # print(bench(stmts['cupipe'], **dict(seqlen=seqlens[0], **memargs)))

    # Run Benchmark
    benchmark = Benchmark(
        x_names=["seqlen"],  # argument names to use as an x-axis for the plot
        x_vals=seqlens,
        line_arg="stmt",  # argument name whose value corresponds to a different line in the plot
        line_names=list(stmts.keys()),
        line_vals=list(stmts.values()),
        plot_name=f'Bench {'forward' if not args.grad else 'backward'}',
        args=memargs,
        xlabel='sequence length',
        x_log=True,
        y_log=True,
    )

    bench = execption2nan(warn=True)(bench)
    Mark(bench, benchmark).run(show_plots=args.showplots, print_data=True, throughput=args.throughput)