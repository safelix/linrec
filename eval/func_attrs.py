import torch
from functools import partial

from bench import bench

import add_linrec_to_path
from linrec.impl.cuda.ops import _C as cuops


if __name__ == '__main__':
    import pandas as pd
    from argparse import ArgumentParser

    parser = ArgumentParser()
    impls = ['linrec_tile_fwd', 'linrec_tile_bwd', 'linrec_pipe_fwd', 'linrec_pipe_bwd']
    parser.add_argument('--kMaxElemsPerThread', type=int, default=None, help='Value to filter kMaxElemsPerThread with.')
    parser.add_argument('--kMaxThreadsPerWarp', type=int, default=32, help='Value to filter kMaxThreadsPerWarp with.')
    parser.add_argument('--kMaxThreadsPerBlock', type=int, default=None, help='Value to filter kMaxThreadsPerBlock with.')
    parser.add_argument('--memcode', type=int, default=None, help='Value to filter memcode with.')
    parser.add_argument('--algocode', type=int, default=None, help='Value to filter algocode with.')
    parser.add_argument('--showplots', action='store_true', help='Whether to show plots.')
    parser.add_argument('--csv', action='store_true', help='Store results as CSV.')
    args = parser.parse_args()
    
    lmem, regs = [], []
    for impl in impls:
        # Prepare Statements
        func = partial(getattr(cuops, impl))

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

        #index, stmts = index[:2], stmts[:2]
        index = pd.MultiIndex.from_frame(pd.DataFrame(configkeys))
        stmts = pd.Series(stmts, index=index)
        func_attrs = partial(getattr(cuops, f'{impl[:-3]}attrs'), fwd=impl[-3:]=='fwd')
        regs += [stmts.map(lambda func: func_attrs(**func.keywords)['numRegs'])]
        lmem += [stmts.map(lambda func: func_attrs(**func.keywords)['localSizeBytes'])]

    print('Number of Registers used:')
    regs = pd.concat(regs, axis='columns').set_axis(impls, axis='columns')
    print(regs.to_string(max_rows=regs.shape[0], max_cols=regs.shape[1]))

    print('\nNumber of bytes spiled to local memory:')
    lmem = pd.concat(lmem, axis='columns').set_axis(impls, axis='columns')
    print(lmem.to_string(max_rows=lmem.shape[0], max_cols=lmem.shape[1]))
