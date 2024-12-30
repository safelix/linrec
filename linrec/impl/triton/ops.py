import torch
import triton
import triton.language as tl
from torch.autograd.function import Function, FunctionCtx

configs = [
    triton.Config(kwargs={'tileSize': kMaxElemsPerWarp * kMaxWarpsPerBlock}, num_warps=kMaxWarpsPerBlock)
    for kMaxWarpsPerBlock in [1, 2, 4, 8, 16, 32]
    for kMaxElemsPerWarp in [128, 256, 512]
]


@triton.jit
def op(accOut, accCoeff, currInp, currCoeff):
    return accOut * currCoeff + currInp, accCoeff * currCoeff

# Triton Tiled Implementations
@triton.jit
def linrec_tile_fwd_kernel(inputs, coeffs, outputs, seqLen, rev: tl.constexpr, tileSize: tl.constexpr):

    # Layout: dim=(X,L), strides=(L,1)
    seqBaseIdx = seqLen * tl.program_id(0)  # process sequences independently: inputs[seqBaseIdx+i]
    inputs += seqBaseIdx                    # get pointer to sequence
    coeffs += seqBaseIdx                    # get pointer to sequence
    outputs += seqBaseIdx                   # get pointer to sequence

    idx = tl.arange(0, tileSize)  # constexpr/static vector to index within tile
    idx = idx if not rev else (seqLen-1 - idx) # reverse indices if needed

    # Load inputs and coeffs of tile into static shared memory arrays
    tileInputs = tl.load(inputs + idx, (0<=idx and idx<seqLen), other=0, eviction_policy='evict_first')
    tileCoeffs = tl.load(coeffs + idx, (0<=idx and idx<seqLen), other=1, eviction_policy='evict_first')

    tileAccOutputs, _ = tl.associative_scan((tileInputs, tileCoeffs), axis=0, combine_fn=op)

    # store outputs
    tl.store(outputs + idx, tileAccOutputs, (0<=idx and idx<seqLen))

@triton.jit
def linrec_tile_bwd_kernel(d_outputs, coeffs, outputs, d_inputs, d_coeffs, seqLen, rev: tl.constexpr, tileSize: tl.constexpr):

    # Layout: dim=(X,L), strides=(L,1)
    seqBaseIdx = seqLen * tl.program_id(0)  # process sequences independently: inputs[seqBaseIdx+i]
    d_outputs += seqBaseIdx                 # get pointer to sequence
    coeffs += seqBaseIdx                    # get pointer to sequence
    outputs += seqBaseIdx                   # get pointer to sequence
    d_inputs += seqBaseIdx                  # get pointer to sequence
    d_coeffs += seqBaseIdx                  # get pointer to sequence

    idx = tl.arange(0, tileSize)  # constexpr/static vector to index within tile
    idx = (seqLen - 1 - idx) if not rev else idx # reverse indices for default backward
    shl, shr = (idx+1, idx-1) if not rev else (idx-1, idx+1) # left and right shifted indices

    # Load inputs and coeffs of tile into static shared memory arrays
    tileDOutputs = tl.load(d_outputs + idx, (0<=idx and idx<seqLen), other=0, eviction_policy='evict_first')
    tileCoeffs = tl.load(coeffs + shl, (0<=shl and shl<seqLen), other=1, eviction_policy='evict_first')

    tileAccDInputs, _ = tl.associative_scan((tileDOutputs, tileCoeffs), axis=0, combine_fn=op)

    # store outputs of back-propagation through time
    tl.store(d_inputs + idx, tileAccDInputs, (0<=idx and idx<seqLen))

    # Load outputs shifted to the right or if reverse shifted to the left
    tileOutputs = tl.load(outputs + shr, (0<=shr and shr<seqLen), other=0, eviction_policy='evict_first')
    tileDCoeffs = tileOutputs * tileAccDInputs  # compute shifted element-wise multiplication
    tl.store(d_coeffs + idx, tileDCoeffs, (0<=idx and idx<seqLen))  # derivative wrt coeffs


def linrec_tile_fwd(inputs, coeffs, reverse=False):
    outputs = torch.empty_like(inputs)
    seqlen = int(inputs.size(-1))
    numseq = int(inputs.numel() // seqlen)
    with torch.cuda.device(inputs.device):
        linrec_tile_fwd_kernel[(numseq,)](inputs, coeffs, outputs, seqlen, reverse, seqlen)
    return outputs

def linrec_tile_bwd(d_outputs, coeffs, outputs, reverse=False):
    d_inputs = torch.empty_like(d_outputs)
    d_coeffs = torch.empty_like(coeffs)
    seqlen = int(d_outputs.size(-1))
    numseq = int(d_outputs.numel() // seqlen)
    with torch.cuda.device(d_outputs.device):
        linrec_tile_bwd_kernel[(numseq,)](d_outputs, coeffs, outputs, d_inputs, d_coeffs, seqlen, reverse, seqlen)
    return d_inputs, d_coeffs

class LinrecTileFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_tile_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_tile_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_tile(inputs: torch.Tensor, coeffs: torch.Tensor, reverse=False):
    return LinrecTileFn.apply(inputs, coeffs, reverse)



# Triton Piped Implementations
@triton.autotune(key=['seqLen'], configs=configs)
@triton.jit
def linrec_pipe_fwd_kernel(inputs, coeffs, outputs, seqLen, rev: tl.constexpr, tileSize: tl.constexpr):
    # Layout: dim=(X,L), strides=(L,1)
    seqBaseIdx = seqLen * tl.program_id(0)  # process sequences independently: inputs[seqBaseIdx+i]
    inputs += seqBaseIdx                    # get pointer to sequence
    coeffs += seqBaseIdx                    # get pointer to sequence
    outputs += seqBaseIdx                   # get pointer to sequence

    seqAccOutput = 0.0
    first, last = (tl.arange(0, tileSize) == 0), (tl.arange(0, tileSize) == tileSize - 1)
    for tileIdx in tl.range(0, seqLen, tileSize):
        idx = tileIdx + tl.arange(0, tileSize)  # constexpr/static vector to index within tile
        idx = idx if not rev else (seqLen-1 - idx) # reverse indices if needed

        # Load inputs and coeffs of tile into static shared memory arrays
        tileInputs = tl.load(inputs + idx, (0<=idx and idx<seqLen), other=0, eviction_policy='evict_first')
        tileCoeffs = tl.load(coeffs + idx, (0<=idx and idx<seqLen), other=1, eviction_policy='evict_first')

        # Combine seqAccOutput with first tileAccInput, perform scan, and save seqAccOutput
        tileInputs = tl.where(first, seqAccOutput * tileCoeffs + tileInputs, tileInputs)
        tileAccOutputs, _ = tl.associative_scan((tileInputs, tileCoeffs), axis=0, combine_fn=op)
        seqAccOutput = tl.sum(tl.where(last, tileAccOutputs, 0))  # select tileAccOutputs[-1]

        # store outputs
        tl.store(outputs + idx, tileAccOutputs, (0<=idx and idx<seqLen))
    return

@triton.autotune(key=['seqLen'], configs=configs)
@triton.jit
def linrec_pipe_bwd_kernel(d_outputs, coeffs, outputs, d_inputs, d_coeffs, seqLen, rev: tl.constexpr, tileSize: tl.constexpr):
    #tl.device_assert(seqLen < tileSize and "Input sequence is longer than maximum tile size.")

    # Layout: dim=(X,L), strides=(L,1)
    seqBaseIdx = seqLen * tl.program_id(0)  # process sequences independently: inputs[seqBaseIdx+i]
    d_outputs += seqBaseIdx                 # get pointer to sequence
    coeffs += seqBaseIdx                    # get pointer to sequence
    outputs += seqBaseIdx                   # get pointer to sequence
    d_inputs += seqBaseIdx                  # get pointer to sequence
    d_coeffs += seqBaseIdx                  # get pointer to sequence

    seqAccDInputs = 0.0 # for sequential accumulation between tiles
    first, last = (tl.arange(0, tileSize) == 0), (tl.arange(0, tileSize) == tileSize - 1)
    for tileIdx in tl.range(0, seqLen, tileSize):
        idx = tileIdx + tl.arange(0, tileSize)  # constexpr/static vector to index within tile
        idx = (seqLen - 1 - idx) if not rev else idx # reverse indices for default backward
        shl, shr = (idx+1, idx-1) if not rev else (idx-1, idx+1) # left and right shifted indices

        # Load inputs and coeffs of tile into static shared memory arrays
        tileDOutputs = tl.load(d_outputs + idx, (0<=idx and idx<seqLen), other=0, eviction_policy='evict_first')
        tileCoeffs = tl.load(coeffs + shl, (0<=shl and shl<seqLen), other=1, eviction_policy='evict_first')

        # Combine seqAccDInputs with first tileAccDInput, perform scan, and save seqAccDInputs
        tileDOutputs = tl.where(first, seqAccDInputs * tileCoeffs + tileDOutputs, tileDOutputs)
        tileAccDInputs, _ = tl.associative_scan((tileDOutputs, tileCoeffs), axis=0, combine_fn=op)
        seqAccDInputs = tl.sum(tl.where(last, tileAccDInputs, 0))  # select tileAccDInputs[-1]

        # store outputs of back-propagation through time
        tl.store(d_inputs + idx, tileAccDInputs, (0<=idx and idx<seqLen))

        # Load outputs shifted to the right or if reverse shifted to the left
        tileOutputs = tl.load(outputs + shr, (0<=shr and shr<seqLen), other=0, eviction_policy='evict_first')
        tileDCoeffs = tileOutputs * tileAccDInputs  # compute shifted element-wise multiplication
        tl.store(d_coeffs + idx, tileDCoeffs, (0<=idx and idx<seqLen))  # derivative wrt coeffs


def linrec_pipe_fwd(inputs, coeffs, reverse=False):
    outputs = torch.empty_like(inputs)
    seqlen = int(inputs.size(-1))
    numseq = int(inputs.numel() // seqlen)
    with torch.cuda.device(inputs.device):
        linrec_pipe_fwd_kernel[(numseq,)](inputs, coeffs, outputs, seqlen, reverse)
    return outputs

def linrec_pipe_bwd(d_outputs, coeffs, outputs, reverse=False):
    d_inputs = torch.empty_like(d_outputs)
    d_coeffs = torch.empty_like(coeffs)
    seqlen = int(d_outputs.size(-1))
    numseq = int(d_outputs.numel() // seqlen)
    with torch.cuda.device(d_outputs.device):
        linrec_pipe_bwd_kernel[(numseq,)](d_outputs, coeffs, outputs, d_inputs, d_coeffs, seqlen, reverse)
    return d_inputs, d_coeffs

class LinrecPipeFn(Function):
    @staticmethod
    def forward(ctx:FunctionCtx, inputs:torch.Tensor, coeffs:torch.Tensor, reverse:bool=False) -> torch.Tensor:
        outputs = linrec_pipe_fwd(inputs=inputs, coeffs=coeffs, reverse=reverse)
        ctx.save_for_backward(coeffs, outputs)
        ctx.reverse = reverse
        return outputs
    
    @staticmethod
    def backward(ctx:FunctionCtx, d_outputs:torch.Tensor):
        coeffs, outputs = ctx.saved_tensors
        d_inputs, d_coeffs = linrec_pipe_bwd(d_outputs=d_outputs, coeffs=coeffs, outputs=outputs, reverse=ctx.reverse)
        return d_inputs, d_coeffs, None

def linrec_pipe(inputs: torch.Tensor, coeffs: torch.Tensor, reverse=False):
    return LinrecPipeFn.apply(inputs, coeffs, reverse)