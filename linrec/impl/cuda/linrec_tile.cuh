#pragma once
#include <cuda.h>
#include <memio.cuh>
#include <cuhelpers.cuh>

/* 
A parallel scan for a fixed-length linear recursion: `outputs[i] = outputs[i - 1] * coeffs[i] + inputs[i];`
Every thread is assumed to receive a different partition of the inputs/coeffs in `threadAccOutput` and
`threadAccCoeff` according to its threadId. The accumulation happens in three levels:
    1. Each thread accumulates threadSeqLen states/coeffs locally.
    2. Threads within each warp accumulate transition states/coeffs among each other.
    3. The first warp loads transition states/coeffs between warps accumulates them.

Finally, each thread distributes their accumulated `prevAccOutput` and `prevAccCoeff` into their local 
`threadAccOutput` and `threadAccCoeff`.

Template Arguments: this is a templated function, i.e. the kernels configuration is specified at compile time
    - `kT`:                                         the data type for which the kernel will specialize
    - `kMaxThreadsPerWarp`:                         the maximum number of threads per warp (=32 on CUDA)
    - `kMaxThreadsPerBlock`:                        the maximum number of threads per block 

Arguments:
    - `kT threadAccOutput[kMaxElemsPerThread]`:      the thread-local inputs which are accumulated into states
    - `kT threadAccCoeff[kMaxElemsPerThread]`:      the thread-local coeffs which accumulated into cumulative coeffs
    - `int threadSeqLen`:                           the length of the subsequence processed by the thread
    - `int laneId`:                                 the id of the thread within its warp
    - `int thisWarpSize`:                           the number of threads within the threads' warp
    - `int warpId`:                                 the id of the warps within the block
    - `int numWarps`:                               the number of warps within the warps' block

Outputs: accumulated states/coeffs are stored in `threadAccOutput` and `threadAccCoeff`.
*/
template <typename kT, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int algocode>
__forceinline__  __device__  void _linrec_scan_tile_parallel_(kT* threadAccOutput, kT* threadAccCoeff, const ushort threadSeqLen, const ushort numThreads) {
    kT warpAccOutput, warpAccCoeff;                                          // for level 2: each thread stores its own within-warp accumulates
    __shared__ kT blockAccOutput[kMaxThreadsPerBlock / kMaxThreadsPerWarp];  // for level 3: warp 0 computes within-block accumulates for all warps
    __shared__ kT blockAccCoeff[kMaxThreadsPerBlock / kMaxThreadsPerWarp];   // use constant definition since thisWarpSize not known at compile time

    // Determine Warp Configuration
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    assert((warpId==numWarps-1) ? (thisWarpSize == lastWarpSize) : true && "Error in thisWarpSize computation.");
    
    // for level 3: warp 0 needs to be big enough to accomodate transition elements between all warps
    assert((warpId == 0) ? thisWarpSize >= numWarps : true && "warp 0 needs to accomodate transition elements between all warps");

    //
    // Level 1: Accumulate elements within this thread
    for(ushort i = 1; i < threadSeqLen; i++) {
        threadAccOutput[i] = threadAccOutput[i-1] * threadAccCoeff[i] + threadAccOutput[i];
        threadAccCoeff[i]  = threadAccCoeff[i-1] * threadAccCoeff[i];
    }


    if (numThreads==1 || algocode==1)
        return;

    
    //
    // Level 2: Accumulate elements across threads within this warp
    warpAccOutput = __shfl_up_sync(0xffffffff, threadAccOutput[threadSeqLen-1], 1); // get transition elements between threads
    warpAccCoeff  = __shfl_up_sync(0xffffffff, threadAccCoeff[threadSeqLen-1], 1);  // get transition elements between threads
    warpAccOutput = (laneId == 0) ? 0 : warpAccOutput;  // set default 1 for first lane (=thread in warp)
    warpAccCoeff  = (laneId == 0) ? 1 : warpAccCoeff;   // set default 0 for first lane (=thread in warp)

    for (ushort delta = 1; delta < thisWarpSize; delta *= 2) { 
        kT prevAccOutput = __shfl_up_sync(0xffffffff, warpAccOutput, delta);
        kT prevAccCoeff  = __shfl_up_sync(0xffffffff, warpAccCoeff, delta);

        //if (laneId < delta) continue; // don't update warpAccOutput and warpAccCoeff in delta lower lanes
        warpAccOutput = (laneId < delta) ? warpAccOutput : prevAccOutput * warpAccCoeff + warpAccOutput;
        warpAccCoeff  = (laneId < delta) ? warpAccCoeff  : prevAccCoeff * warpAccCoeff;
    }

    for (ushort i = 0; i < threadSeqLen; i++) { // distribute accumulates into thread-local array
        threadAccOutput[i] = warpAccOutput * threadAccCoeff[i] + threadAccOutput[i];
        threadAccCoeff[i]  = warpAccCoeff * threadAccCoeff[i];
    }

    
    if (numWarps==1 || algocode==2)
        return;

    //
    // Level 3: Accumulate elements accross warps within thread block
    if (laneId == thisWarpSize-1) {
        blockAccOutput[warpId] = threadAccOutput[threadSeqLen-1]; // get transition elements between threads
        blockAccCoeff[warpId]  = threadAccCoeff[threadSeqLen-1];  // get transition elements between threads
    }
    __syncthreads(); // make sure that all threads see the updated shared memory

    for (ushort delta = 1; delta < numWarps; delta *= 2) {
        if (warpId > 0) break;          // compute only in warpId = 0
        if (laneId >= numWarps) break;  // compute only with laneId<numWarps

        unsigned mask = 0xffffffff >> (32-numWarps);
        kT prevAccOutput = __shfl_up_sync(mask, blockAccOutput[laneId], delta);
        kT prevAccCoeff = __shfl_up_sync(mask, blockAccCoeff[laneId], delta);

        //if (laneId < delta) continue; // don't update warpAccOutput and warpAccCoeff in delta lower lanes
        blockAccOutput[laneId] = (laneId < delta) ? blockAccOutput[laneId] : prevAccOutput * blockAccCoeff[laneId] + blockAccOutput[laneId];
        blockAccCoeff[laneId]  = (laneId < delta) ? blockAccCoeff[laneId]  : prevAccCoeff * blockAccCoeff[laneId];
    }
    __syncthreads(); // make sure that all threads see the updated shared memory

    for(ushort i = 0; i < threadSeqLen; i++) { // distribute accumulates into threads
        if (warpId == 0) break; // don't distribute into first warp
        threadAccOutput[i] = blockAccOutput[warpId-1] * threadAccCoeff[i] + threadAccOutput[i];
        threadAccCoeff[i] = blockAccCoeff[warpId-1] * threadAccCoeff[i]; // this could be optimized out
    }
}


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_tile_fwd_kernel_naive(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen) {

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[seqBaseIdx+i]
    inputs = &inputs[seqBaseIdx];               // get pointer to sequence
    coeffs = &coeffs[seqBaseIdx];               // get pointer to sequence
    outputs = &outputs[seqBaseIdx];             // get pointer to sequence

    // Determine Tile Layout
    const ushort numThreads = blockDim.x;
    const ushort threadId = threadIdx.x;                                                            // index of current thread
    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute seqLen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const ushort threadBaseIdx = threadId * elemsPerThread - max(threadTailId, 0);                  // base index to process by every thread


    //
    // Load inputs and coeffs of tile into thread-local arrays
    kT threadAccOutput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    for(ushort i = 0; i < kMaxElemsPerThread; i++) {
        threadAccOutput[i] = (i < threadSeqLen) ? inputs[threadBaseIdx + i] : 0;
        threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[threadBaseIdx + i] : 1;
    }

    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, numThreads);
    }

    // Store outputs
    for(ushort i = 0; i < threadSeqLen; i++) {
        outputs[threadBaseIdx + i] = threadAccOutput[i];
    }
}


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_tile_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen, const bool rev) {
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[seqBaseIdx+i]
    inputs = &inputs[seqBaseIdx];               // get pointer to sequence
    coeffs = &coeffs[seqBaseIdx];               // get pointer to sequence
    outputs = &outputs[seqBaseIdx];             // get pointer to sequence

    // Determine Tile Layout
    const ushort numThreads = blockDim.x;
    const ushort threadIdrev = !rev ? threadIdx.x : (numThreads - threadIdx.x - 1);                 // thread index, reversed to load reversed
    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute seqLen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = (int) threadIdrev - (numThreads - numTailThreads);                     // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const ushort threadBaseIdx = threadIdrev * elemsPerThread - max(threadTailId, 0);               // base index to process by every thread
    

    //
    // Load inputs and coeffs of tile into thread-local arrays
    kT threadAccOutput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    memio::load<kT, memcode>(threadAccOutput, inputs, seqLen, smem, threadBaseIdx, threadSeqLen, rev, kT(0), kMaxElemsPerThread);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, threadBaseIdx, threadSeqLen, rev, kT(1), kMaxElemsPerThread);

    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, numThreads);
    }

    // Store outputs
    memio::store<kT, memcode>(outputs, threadAccOutput, seqLen, smem, threadBaseIdx, threadSeqLen, rev);
}




template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_tile_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool rev) {
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[seqBaseIdx+i]
    d_outputs = &d_outputs[seqBaseIdx];         // get pointer to sequence        
    coeffs = &coeffs[seqBaseIdx];               // get pointer to sequence
    outputs = &outputs[seqBaseIdx];             // get pointer to sequence
    d_inputs = &d_inputs[seqBaseIdx];           // get pointer to sequence
    d_coeffs = &d_coeffs[seqBaseIdx];           // get pointer to sequence

    // Determine Tile Layout
    const ushort numThreads = blockDim.x;
    const ushort threadIdrev = !rev ? (numThreads - threadIdx.x - 1) : threadIdx.x;                 // reversed thread index to load reversed by default
    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute seqLen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = (int) threadIdrev - (numThreads - numTailThreads);                     // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const ushort threadBaseIdx = threadIdrev * elemsPerThread - max(threadTailId, 0);               // base index to process by every thread


    //
    // Load inputs and coeffs of tile into thread-local arrays
    kT threadAccDInput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    memio::load<kT, memcode>(threadAccDInput, d_outputs, seqLen, smem, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread, !rev ? -1 : 1);

    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, numThreads);
    }

    // Store outputs of Back Propoagation Through Time
    memio::store<kT, memcode>(d_inputs, threadAccDInput, seqLen, smem, threadBaseIdx, threadSeqLen, !rev);
    
    
    //
    // Load outputs shifted to the right or if reverse shifted to the left
    kT threadDCoeff[kMaxElemsPerThread];
    memio::load<kT, memcode>(threadDCoeff, outputs, seqLen, smem, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread, !rev ? 1 : -1);
    
    // Compute shifted element-wise multiplication
    for(ushort i = 0; 0 < algocode && i < kMaxElemsPerThread; i++) {
        threadDCoeff[i] *= threadAccDInput[i];
    }
    
    // Store outputs of shifted element-wise multiplication
    memio::store<kT, memcode>(d_coeffs, threadDCoeff, seqLen, smem, threadBaseIdx, threadSeqLen, !rev);
}
// new line to avoid ptx syntax error "Parsing error near ''"