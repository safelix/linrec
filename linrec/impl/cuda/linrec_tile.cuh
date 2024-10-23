#pragma once
#include <cuda.h>
#include <memio.cuh>

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
    - `kMaxElemsPerThread`:                         the maximum number of elements each thread processes
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
template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int algocode>
__forceinline__  __device__  void _linrec_scan_tile_parallel_(kT* threadAccOutput, kT* threadAccCoeff, const ushort threadSeqLen, const ushort laneId, const ushort thisWarpSize, const ushort warpId, const ushort numWarps, const ushort numThreads) {
    kT warpAccOutput, warpAccCoeff;                                          // for level 2: each thread stores its own within-warp accumulates
    __shared__ kT blockAccOutput[kMaxThreadsPerBlock / kMaxThreadsPerWarp];  // for level 3: warp 0 computes within-block accumulates for all warps
    __shared__ kT blockAccCoeff[kMaxThreadsPerBlock / kMaxThreadsPerWarp];   // use constant definition since thisWarpSize not known at compile time

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
    //__syncwarp(); // TODO: avoid divergence?

    for (ushort i = 0; i < threadSeqLen; i++) { // distribute accumulates into thread elements
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


// Integer ceiling devision 
template <typename T>
constexpr T ceildiv(T lhs, T rhs){
    return ((lhs - 1) / rhs) + 1;
}

template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_tile_fwd_kernel_naive(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen) {
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[seqBaseIdx+i]
    inputs = &inputs[seqBaseIdx];
    coeffs = &coeffs[seqBaseIdx];
    outputs = &outputs[seqBaseIdx];

    // Determine Warp Configuration (at run-time)
    const ushort numThreads = kMaxThreadsPerBlock; //blockDim.x;
    const ushort threadId = threadIdx.x;
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    
    assert((warpId==numWarps-1) ? (thisWarpSize == kMaxThreadsPerWarp) : true && "Error in thisWarpSize computation.");

    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute subseqlen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const ushort threadBaseIdx = threadId * elemsPerThread - max(threadTailId, 0);                     // base index to process by every thread


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
        _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
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
    inputs = &inputs[seqBaseIdx];
    coeffs = &coeffs[seqBaseIdx];
    outputs = &outputs[seqBaseIdx];

    // Determine Warp Configuration (at run-time)
    const ushort numThreads = kMaxThreadsPerBlock; //blockDim.x;
    const ushort threadId = threadIdx.x;
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    
    assert((warpId==numWarps-1) ? (thisWarpSize == kMaxThreadsPerWarp) : true && "Error in thisWarpSize computation.");

    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute subseqlen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const ushort threadBaseIdx = !rev ? (threadId * elemsPerThread - max(threadTailId, 0)) :                    // base index to process by every thread
                                    ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0));   // reverse ordered base index
    

    //
    // Load inputs and coeffs of tile into thread-local arrays
    kT threadAccOutput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadAccOutput[i] = (i < threadSeqLen) ? inputs[threadBaseIdx + (!rev ? i : (threadSeqLen-1-i))] : 0;
        threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[threadBaseIdx + (!rev ? i : (threadSeqLen-1-i))] : 1;
    }
    memio::load<kT, memcode>(threadAccOutput, inputs, seqLen, smem, threadBaseIdx, threadSeqLen, rev, kT(0), kMaxElemsPerThread);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, threadBaseIdx, threadSeqLen, rev, kT(1), kMaxElemsPerThread);

    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
    }

    // Store outputs
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        outputs[threadBaseIdx + (!rev ? i : (threadSeqLen-1-i))] = threadAccOutput[i];
    }
    memio::store<kT, memcode>(outputs, threadAccOutput, seqLen, smem, threadBaseIdx, threadSeqLen, rev);
}




template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_tile_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool rev) {
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently in reverse: inputs[seqBaseIdx-i]
    d_outputs = &d_outputs[seqBaseIdx];
    coeffs = &coeffs[seqBaseIdx];
    outputs = &outputs[seqBaseIdx];
    d_inputs = &d_inputs[seqBaseIdx];
    d_coeffs = &d_coeffs[seqBaseIdx];

    // Determine Warp Configuration (at run-time)
    const ushort numThreads = kMaxThreadsPerBlock; // blockDim.x;
    const ushort threadId = threadIdx.x;
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    
    assert((warpId==numWarps-1) ? (thisWarpSize == kMaxThreadsPerWarp) : true && "Error in thisWarpSize computation.");

    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute subseqlen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const ushort threadBaseIdx = !rev ? ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)) :  // backwards ordered base index to process by every thread
                                    ((threadId * elemsPerThread - max(threadTailId, 0)));                           // forward ordered base index


    //
    // Load inputs and coeffs of tile into thread-local arrays
    kT threadAccDInput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadAccDInput[i] = (i < threadSeqLen) ? d_outputs[threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] : 0;
        threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] : 1;
    }
    memio::load<kT, memcode>(threadAccDInput, d_outputs, seqLen, smem, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread);


    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
    }

    // Store outputs of Back Propoagation Through Time
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        d_inputs[threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] = threadAccDInput[i];
    }
    memio::store<kT, memcode>(d_inputs, threadAccDInput, seqLen, smem,0, seqLen,  threadBaseIdx, threadSeqLen, !rev);
    
    
    //
    // Compute and Store Coefficient Derivatives (element-wise shifted multiplication)
    // Outputs are shifted to the right (index -1) or if reverse shifted to the left (index +1) 
    // - edge case for (!rev && threadBaseIdx==first): out of bounds at outputs[-1] => threadShBaseIdx=0, threadShSeqLen=threadSeqLen-1
    // - edge case for (rev && threadBaseIdx==last): out of bounds at outputs[seqLen] => threadShSeqLen=threadSeqLen-1
    bool lastThread = (!rev && threadBaseIdx==0) || (rev && threadBaseIdx+threadSeqLen==seqLen);
    const ushort threadShBaseIdx = threadBaseIdx + (!rev ? (!lastThread ? -1 : 0) : +1);
    const ushort threadShSeqLen = threadSeqLen + ((lastThread && threadSeqLen>0) ? -1 : 0);

    // Load shifted outputs of tile into thread-local arrays
    kT threadDCoeff[kMaxElemsPerThread];
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadDCoeff[i] = (i < threadShSeqLen) ? outputs[threadShBaseIdx + (!rev ? (threadShSeqLen-i-1) : i)] : 0;
    }
    memio::load<kT, memcode>(threadDCoeff, outputs, seqLen, smem, threadShBaseIdx, threadShSeqLen, !rev, kT(0), kMaxElemsPerThread);
    
    // shifted element-wise multiplication
    for(ushort i = 0; 0 < algocode && i < kMaxElemsPerThread; i++) {
        threadDCoeff[i] *= threadAccDInput[i];
    }
    
    // Store outputs of shifted element-wise multiplication
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        d_coeffs[threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] = threadDCoeff[i];
    }
    memio::store<kT, memcode>(d_coeffs, threadDCoeff, seqLen, smem, threadBaseIdx, threadSeqLen, !rev);
}
// new line to avoid ptx syntax error "Parsing error near ''"