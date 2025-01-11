import torch
from functools import partial
from utils import execption2nan, memio_limit, meminit
from test import test
from bench import bench

import add_linrec_to_path
from linrec.impl.cuda.ops import _C as cuops


if __name__ == '__main__':
    import pandas as pd
    from argparse import ArgumentParser
    from triton.testing import Benchmark, Mark

    parser = ArgumentParser()
    impls = ['linrec_tile_fwd', 'linrec_tile_bwd', 'linrec_pipe_fwd', 'linrec_pipe_bwd']
    parser.add_argument('impl', choices=impls, help='Which implementation to tune.')
    parser.add_argument('--seqlen_min', type=int, default=4, help='Minimal sequence length to tune with (log2).')
    parser.add_argument('--seqlen_max', type=int, default=16, help='Maximal sequence length to tune with (log2).')
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
    parser.add_argument('--showplots', action='store_true', help='Whether to show plots.')
    parser.add_argument('--csv', action='store_true', help='Store results as CSV.')
    args = parser.parse_args()

    # Prepare Data Arguments
    args.fwd = (args.impl[-3:]=='fwd')
    args.device = torch.device(f'cuda:{args.device}')
    sm = torch.cuda.get_device_properties(args.device).multi_processor_count

    seqlens = [2**i for i in range(args.seqlen_min, args.seqlen_max+1)]
    memargs = dict(n_batches=args.n_batches if (args.n_batches > 0) else sm,
                n_channels=args.n_channels,
                dtype=torch.float32,
                device=args.device,
                seed=args.seed,
                grad=None if args.fwd else 'bwd', 
                fwd=cuops.linrec_ref_fwd, 
                meminit=meminit)
    
    iolimit = [bench(memio_limit, throughput=args.throughput, **dict(seqlen=seqlen, **memargs)) for seqlen in seqlens]
    
    # Prepare Statements
    func = partial(getattr(cuops, args.impl), reverse=args.reverse)
    #ref =  partial(getattr(cuops, f'linrec_ref_{args.impl[-3:]}'), reverse=args.reverse)
    ref =  partial(getattr(cuops, f'linrec_ref_{args.impl[-3:]}'), reverse=args.reverse)

    configkeys, stmts = [], [] # prepare config keys and statements
    for p in cuops.config_list:
        config = dict(zip(cuops.config_names, p))

        # filter configs from parsed arguments
        configkey, drop_stmt = {}, False
        for name in config.keys():
            if getattr(args, name) is None:
                configkey[name] = config[name] # drop key if name in args
            elif getattr(args, name) != config[name]:
                drop_stmt = drop_stmt or True # drop if 
        if drop_stmt:
            continue

        configkeys.append(configkey)
        stmts.append(partial(func, **config))

    assert len(configkeys) > 0
    index = pd.MultiIndex.from_frame(pd.DataFrame(configkeys))
    stmts = pd.Series(stmts, index=index)

    # Execute one statment 
    #print(bench(stmts.iloc[0], throughput=args.throughput, **dict(seqlen=seqlens[0], **memargs)))
    #print(test(stmts.iloc[0], ref=ref, **dict(seqlen=seqlens[0], **memargs)))

    # Run Benchmark
    benchmark = Benchmark(
        x_names=["seqlen"],  # argument names to use as an x-axis for the plot
        x_vals=seqlens,
        line_arg="stmt",  # argument name whose value corresponds to a different line in the plot
        line_names=list(str(i) for i in stmts.index),
        line_vals=list(stmts),
        plot_name=f'Tune {args.impl} ({torch.cuda.get_device_name()})',
        args=memargs,
        xlabel='sequence length',
        x_log=True,
        y_log=True,
    )

    columns = pd.Index(seqlens, name='seqlen')
    test = execption2nan()(test)
    diffs = Mark(test, benchmark).run(return_df=True, show_plots=args.showplots, ref=ref)
    diffs = pd.DataFrame(diffs.values[:, 1:].T, columns=columns, index=stmts.index)
    if args.csv:
        diffs.to_csv(f'tune_{args.impl}{'_rev' if args.reverse else ''}_diffs.csv')

    bench = execption2nan(warn=True)(bench)
    times = Mark(bench, benchmark).run(return_df=True, show_plots=args.showplots, throughput=args.throughput)
    times = pd.DataFrame(times.values[:, 1:].T, columns=columns, index=stmts.index)
    if args.csv:
        times.to_csv(f'tune_{args.impl}{'_rev' if args.reverse else ''}_times.csv')


    # Compute Max Perfomance
    maxperfidx = times.idxmax(axis=0, skipna=True) if args.throughput else times.idxmin(axis=0, skipna=True)
    maxperf = times.max(axis=0, skipna=True) if args.throughput else times.min(axis=0, skipna=True)
    maxperf = pd.concat([maxperfidx.apply(pd.Series).astype(int, errors='ignore'), maxperf], axis=1).T
    maxperf.index = stmts.index.names + ['throughput (GB/s)' if args.throughput else 'runtime (ms)']

    # Add iolimit to index and lmem column to data
    columns = pd.MultiIndex.from_arrays([seqlens, iolimit], names=['seqlen', 'iolimit'])
    times = pd.DataFrame(times.values, columns=columns, index=stmts.index)
    func_attrs = partial(getattr(cuops, f'{args.impl[:-3]}attrs'), fwd=args.fwd)
    times.insert(0, ('lmem', None), stmts.map(lambda func: func_attrs(**func.keywords)['localSizeBytes']))
    times.insert(0, ('regs', None), stmts.map(lambda func: func_attrs(**func.keywords)['numRegs']))

    print('\n################## Testing ##################')
    print(diffs.to_string(float_format=lambda x: f'{x:.1e}', max_rows=diffs.shape[0], max_cols=diffs.shape[1]))
    print(f'Max difference: {diffs.max(axis=None, skipna=True):.1e}')

    print('\n################## Benchmark ##################')
    float_format = (lambda x: f'{x:.1f}') if args.throughput else (lambda x: f'{x:.3f}')
    print(times.to_string(float_format=float_format, max_rows=times.shape[0], max_cols=times.shape[1]))

    print('Max performance:')
    maxperf = pd.concat([maxperf.iloc[:-1], maxperf.iloc[-1:].map(float_format)]) # float format only last row
    print(maxperf.to_string(float_format=lambda x: f'{x:.0f}', max_rows=maxperf.shape[0], max_cols=maxperf.shape[1]))        
