import torch
from triton.testing import Benchmark, Mark, do_bench
from utils import execption2nan, meminit, memio, memio_limit

import add_linrec_to_path
from linrec.impl.cuda import build
_C = build.extension()


@execption2nan(warn=True)
def bench(stmt, meminit, throughput=False, **kwargs):

    data = meminit(**kwargs)

    if stmt in [memio_limit, 'memio_limit']:
        ms, bytes = memio_limit(**kwargs)
    else:
        with torch.cuda.device(kwargs.get('device', None)):
            ms = do_bench(lambda: stmt(*data))#, warmup=50, rep=1000)
        bytes = memio(stmt, data)
    del data

    if throughput: # throughput
        return (bytes * 1e-9) / (ms * 1e-3) # GB/s
    return ms

@execption2nan()
def test(stmt, meminit, ref, **kwargs):
    data = meminit(**kwargs)

    out = stmt(*data)
    sol = ref(*data)
    del data

    out = out if isinstance(out, torch.Tensor) else torch.stack(out)
    sol = sol if isinstance(sol, torch.Tensor) else torch.stack(sol)

    return (sol - out).abs().max().item()


if __name__ == '__main__':
    import pandas as pd
    from argparse import ArgumentParser
    from functools import partial

    parser = ArgumentParser()
    impls = ['linrec_tile_fwd', 'linrec_tile_bwd']
    parser.add_argument('impl', choices=impls, help='Which implementation to tune.')
    parser.add_argument('--seqlen_min', type=int, default=4, help='Minimal sequence length to tune with.')
    parser.add_argument('--seqlen_max', type=int, default=14, help='Maximal sequence length to tune with.')
    parser.add_argument('--n_batches', type=int, default=-1, help='Number of batches to tune with (-1 = Shared Multiprocessors).')
    parser.add_argument('--n_channels', type=int, default=100, help='Number of channels to tune with.')
    parser.add_argument('--reverse', action='store_true', help='Use reverse scan to tune with.')
    parser.add_argument('--throughput', action='store_true', help='Use throughput as a metric.')
    parser.add_argument('--kMaxElemsPerThread', type=int, default=None, help='Value to filter kMaxElemsPerThread with.')
    parser.add_argument('--kMaxThreadsPerWarp', type=int, default=32, help='Value to filter kMaxThreadsPerWarp with.')
    parser.add_argument('--kMaxThreadsPerBlock', type=int, default=None, help='Value to filter kMaxThreadsPerBlock with.')
    parser.add_argument('--memcode', type=int, default=None, help='Value to filter memcode with.')
    parser.add_argument('--algocode', type=int, default=None, help='Value to filter algocode with.')
    parser.add_argument('--seed', type=int, default=12334567890, help='Seed to generate data with.')
    parser.add_argument('--device', type=int, default=0, help='Device id to tune on.')
    args = parser.parse_args()

    # Prepare Data Arguments
    args.fwd = (args.impl[-3:]=='fwd')
    args.device = torch.device(f'cuda:{args.device}')
    sm = torch.cuda.get_device_properties(args.device).multi_processor_count

    seqlens = [2**i for i in range(args.seqlen_min, args.seqlen_max+1)]
    memargs = dict(n_batches=args.n_batches if (args.n_batches > 0) else sm,
                n_channels=args.n_channels,
                fwd=args.fwd, 
                dtype=torch.float32,
                device=args.device,
                seed=args.seed)
    
    iolimit = [bench(memio_limit, meminit, throughput=args.throughput, **dict(seqlen=seqlen, **memargs)) for seqlen in seqlens]
    
    # Prepare Statements
    func = partial(getattr(_C, args.impl), reverse=args.reverse)
    ref =  partial(getattr(_C, f'linrec_ref_{args.impl[-3:]}'), reverse=args.reverse)
    names = ["kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"]

    index, stmts = [], []
    for p in _C.COMPILEPARAMS:
        kwargs = dict(zip(names, p))

        index_key, drop_stmt = {}, False
        for arg in kwargs:
            if getattr(args, arg) is None:
                index_key[arg] = kwargs[arg]
            elif getattr(args, arg) != kwargs[arg]:
                drop_stmt = drop_stmt or True
        if drop_stmt:
            continue

        stmts.append(partial(func, **kwargs))
        index.append(index_key)

    #index, stmts = index[:2], stmts[:2]
    index = pd.MultiIndex.from_frame(pd.DataFrame(index))
    stmts = pd.Series(stmts, index=index)

    # Execute one statment 
    #print(bench(stmts.iloc[0], meminit, throughput=args.throughput, **dict(seqlen=seqlens[0], **memargs)))
    #print(test(stmts.iloc[0], meminit, ref=ref, **dict(seqlen=seqlens[0], **memargs)))

    # Run Benchmark
    benchmark = Benchmark(
        x_names=["seqlen"],  # argument names to use as an x-axis for the plot
        x_vals=seqlens,
        line_arg="stmt",  # argument name whose value corresponds to a different line in the plot
        line_names=list(str(i) for i in stmts.index),
        line_vals=list(stmts),
        plot_name=f'Tune {args.impl} ({torch.cuda.get_device_name()})',
        args=dict(meminit=meminit, **memargs),
        xlabel='sequence length',
        x_log=True,
        y_log=True,
    )

    columns = pd.Index(seqlens, name='seqlen')
    diffs = Mark(test, benchmark).run(return_df=True, ref=ref)
    diffs = pd.DataFrame(diffs.values[:, 1:].T, columns=columns, index=index)
    diffs.to_csv(f'tune_{args.impl}{'_rev' if args.reverse else ''}_diffs.csv')

    times = Mark(bench, benchmark).run(return_df=True, throughput=args.throughput)
    times = pd.DataFrame(times.values[:, 1:].T, columns=columns, index=index)
    times.to_csv(f'tune_{args.impl}{'_rev' if args.reverse else ''}_times.csv')

    # Add iolimit to index and lmem column to data
    columns = pd.MultiIndex.from_arrays([seqlens, iolimit], names=['seqlen', 'iolimit'])
    times = pd.DataFrame(times.values, columns=columns, index=index)
    func_attrs = partial(getattr(_C, f'{args.impl[:-3]}attrs'), fwd=args.fwd)
    times.insert(0, ('lmem', None), stmts.map(lambda func: func_attrs(**func.keywords)['localSizeBytes']))
    times.insert(0, ('regs', None), stmts.map(lambda func: func_attrs(**func.keywords)['numRegs']))

    print('\n################## Testing ##################')
    print(diffs.to_string(float_format=lambda x: f'{x:.1e}', max_rows=diffs.shape[0], max_cols=diffs.shape[1]))
    print(f'Max difference: {diffs.max(axis=None, skipna=True):.1e}')

    print('\n################## Benchmark ##################')
    print(times.to_string(float_format=lambda x: f'{x:.1f}', max_rows=times.shape[0], max_cols=times.shape[1]))

