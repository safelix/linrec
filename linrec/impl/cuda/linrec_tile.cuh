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


// Helper function 
template <typename T>
constexpr T ceildiv(T lhs, T rhs){
    return ((lhs - 1) / rhs) + 1;
}

template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_fwd_tile_kernel(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen) {
    kT threadAccOutput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    __shared__ kT smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Determine Warp Configuration (at run-time)
    const ushort numThreads = kMaxThreadsPerBlock; //blockDim.x;
    const ushort threadId = threadIdx.x;
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    
    assert((warpId==numWarps-1) ? (thisWarpSize == kMaxThreadsPerWarp) : true && "Error in thisWarpSize computation.");


    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[seqBaseIdx+i]
    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute subseqlen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = threadId - (numThreads - numTailThreads);                              // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const int threadBaseIdx = threadId * elemsPerThread - max(threadTailId, 0);                     // base index to process by every thread

    assert((threadId==numThreads-1) ? (seqBaseIdx + seqLen == seqBaseIdx + threadBaseIdx + threadSeqLen) : true && "Error in threadBaseIdx or threadSeqLen computation.");


    // Load inputs and coeffs of tile into thread-local arrays
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadAccOutput[i] = (i < threadSeqLen) ? inputs[seqBaseIdx + threadBaseIdx + i] : 0;
        threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[seqBaseIdx + threadBaseIdx + i] : 1;
    }
    memio::load<kT, memcode>(threadAccOutput, inputs, threadBaseIdx, threadSeqLen, false, kT(0), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, threadBaseIdx, threadSeqLen, false, kT(1), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);

    //
    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
    }

    //
    // Store outputs
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        outputs[seqBaseIdx + threadBaseIdx + i] = threadAccOutput[i];
    }
    memio::store<kT, memcode>(outputs, threadAccOutput, threadBaseIdx, threadSeqLen, false, smem, seqBaseIdx, seqLen);
}


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_fwd_tile_kernel(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen, const bool rev) {
    kT threadAccOutput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    __shared__ kT smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Determine Warp Configuration (at run-time)
    const ushort numThreads = kMaxThreadsPerBlock; //blockDim.x;
    const ushort threadId = threadIdx.x;
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    
    assert((warpId==numWarps-1) ? (thisWarpSize == kMaxThreadsPerWarp) : true && "Error in thisWarpSize computation.");


    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[seqBaseIdx+i]
    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute subseqlen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = threadId - (numThreads - numTailThreads);                              // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const int threadBaseIdx = !rev ? (threadId * elemsPerThread - max(threadTailId, 0)) :                     // base index to process by every thread
                                    ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)); // reverse ordered base index

    assert((!rev && threadId==numThreads-1) ? (seqBaseIdx + seqLen == seqBaseIdx + threadBaseIdx + threadSeqLen) : true && "Error in threadBaseIdx or threadSeqLen computation.");
    assert((rev && threadId==0) ? (seqBaseIdx + seqLen == seqBaseIdx + threadBaseIdx + threadSeqLen) : true && "Error in threadBaseIdx or threadSeqLen computation.");


    // Load inputs and coeffs of tile into thread-local arrays
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadAccOutput[i] = (i < threadSeqLen) ? inputs[seqBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i)] : 0;
        threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[seqBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i)] : 1;
    }
    memio::load<kT, memcode>(threadAccOutput, inputs, threadBaseIdx, threadSeqLen, rev, kT(0), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, threadBaseIdx, threadSeqLen, rev, kT(1), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);

    //
    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
    }

    //
    // Store outputs
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        outputs[seqBaseIdx + threadBaseIdx + (!rev ? i : (threadSeqLen-1-i))] = threadAccOutput[i];
    }
    memio::store<kT, memcode>(outputs, threadAccOutput, threadBaseIdx, threadSeqLen, rev, smem, seqBaseIdx, seqLen);
}




template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_bwd_tile_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool rev) {
    kT threadAccDInput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    __shared__ kT smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Determine Warp Configuration (at run-time)
    const ushort numThreads = kMaxThreadsPerBlock; // blockDim.x;
    const ushort threadId = threadIdx.x;
    const ushort warpId = threadIdx.x / kMaxThreadsPerWarp;
    const ushort laneId = threadIdx.x % kMaxThreadsPerWarp;
    const ushort numWarps = ceildiv(numThreads, kMaxThreadsPerWarp);
    const ushort lastWarpSize = numThreads - kMaxThreadsPerWarp * (numWarps-1);
    const ushort thisWarpSize = (warpId==numWarps-1) ? lastWarpSize : kMaxThreadsPerWarp;
    
    assert((warpId==numWarps-1) ? (thisWarpSize == kMaxThreadsPerWarp) : true && "Error in thisWarpSize computation.");


    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently in reverse: inputs[seqBaseIdx-i]
    const ushort elemsPerThread = ceildiv(seqLen, (int) numThreads);                                // distribute subseqlen among numThreads
    const ushort numTailThreads = numThreads * elemsPerThread - seqLen;                             // last numTailThreads have one elem less
    const int threadTailId = threadId - (numThreads - numTailThreads);                              // tail start indicated by ..., 0, 1, 2, ...
    const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
    const int threadBaseIdx = !rev ? ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)) : // backwards ordered base index to process by every thread
                                    ((threadId * elemsPerThread - max(threadTailId, 0)));                       // forward ordered base index

    assert((!rev && threadId==0) ? (seqBaseIdx + seqLen == seqBaseIdx + threadBaseIdx + threadSeqLen) : true && "Error in threadBaseIdx or threadSeqLen computation.");
    assert((rev && threadId==numThreads-1) ? (seqBaseIdx + seqLen == seqBaseIdx + threadBaseIdx + threadSeqLen) : true && "Error in threadBaseIdx or threadSeqLen computation.");


    // Load inputs and coeffs of tile into thread-local arrays
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadAccDInput[i] = (i < threadSeqLen) ? d_outputs[seqBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] : 0;
        threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[seqBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] : 1;
    }
    memio::load<kT, memcode>(threadAccDInput, d_outputs, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);
    memio::load<kT, memcode>(threadAccCoeff, coeffs, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);
    

    //
    // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
    if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan
        _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
    }


    //
    // Store outputs of Back Propoagation Through Time
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        d_inputs[seqBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] = threadAccDInput[i];
    }
    memio::store<kT, memcode>(d_inputs, threadAccDInput, threadBaseIdx, threadSeqLen, !rev, smem, seqBaseIdx, seqLen);
    

    // Compute and Store Coefficient Derivatives (element-wise shifted multiplication) 
    kT threadShOutput[kMaxElemsPerThread]; 
    kT threadDCoeff[kMaxElemsPerThread];
    
    // Load outputs shifted to the right (-1) or if reverse shifted to the left (+1) 
    // - edge case (!rev && threadBaseIdx==0): out of bounds at outputs[-1] => threadShBaseIdx=0, threadShSeqLen=threadSeqLen-1
    // - edge case (rev && threadBaseIdx==last): out of bounds at outputs[seqLen] => threadShSeqLen=threadSeqLen-1
    const int threadShBaseIdx = threadBaseIdx + (!rev ? ((threadBaseIdx > 0) ? -1 : 0) : +1);
    const ushort threadShSeqLen = ((!rev && 0 < threadBaseIdx) || (rev && threadBaseIdx + threadSeqLen < seqLen)) ? threadSeqLen : threadSeqLen-1;
    
    for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
        threadShOutput[i] = (i < threadShSeqLen) ? outputs[seqBaseIdx + threadShBaseIdx + (!rev ? (threadShSeqLen-i-1) : i)] : 0;
    }
    memio::load<kT, memcode>(threadShOutput, outputs, threadShBaseIdx, threadShSeqLen, !rev, kT(0), kMaxElemsPerThread, smem, seqBaseIdx, seqLen);
    
    for(ushort i = 0; i < kMaxElemsPerThread; i++) {     // element-wise multiplication
        threadDCoeff[i] = threadShOutput[i] * threadAccDInput[i];
    }
    
    for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
        d_coeffs[seqBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] = threadDCoeff[i];
    }
    memio::store<kT, memcode>(d_coeffs, threadDCoeff, threadBaseIdx, threadSeqLen, !rev, smem, seqBaseIdx, seqLen);    
    
    return;
}