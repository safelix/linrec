#pragma once
#include <cuda.h>
#include <memio.cuh>
#include <linrec_tile.cuh>


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_pipe_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen, const bool rev) {
    kT threadAccOutput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    __shared__ kT seqAccOutput, seqAccCoeff;
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

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
    int tileBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[tileBaseIdx+i]
    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;
    tileBaseIdx += !rev ? 0 : max(seqLen - elemsPerTile, 0); // if reverse start with last tile
    
    int remainingElems = seqLen;
    while (remainingElems > 0) { // sequentially combine subsequences of size elemsPerTile if seqLen doesn't fit into one thread block

        const ushort tileSeqLen = min(remainingElems, elemsPerTile);  // elemsPerTile                 // length of the tile to scan with thread block
        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute subseqlen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = !rev ? (threadId * elemsPerThread - max(threadTailId, 0)) :                     // base index to process by every thread
                                        ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)); // reverse ordered base index

        // Load inputs and coeffs of tile into thread-local arrays
        for (ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
            const int index = tileBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i);
            assert((i < threadSeqLen) ? (0 <= index && index < seqLen * (blockIdx.x + 1)) : true && "Error in index computation.");
            threadAccOutput[i] = (i < threadSeqLen) ? inputs[index] : 0;
            threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[index] : 1;
        }
        memio::load<kT, memcode>(threadAccOutput, inputs, threadBaseIdx, threadSeqLen, rev, kT(0), kMaxElemsPerThread, smem, tileBaseIdx, tileSeqLen);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, threadBaseIdx, threadSeqLen, rev, kT(1), kMaxElemsPerThread, smem, tileBaseIdx, tileSeqLen);

        //
        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccOutput and -Gate 
            if (threadId == 0 && remainingElems < seqLen){
                threadAccOutput[0] = seqAccOutput * threadAccCoeff[0] + threadAccOutput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
        
            // Store last threadAccOutput and -Gate into seqAccOutput and and -Gate
            if (threadId == numThreads-1 && elemsPerTile < remainingElems) {
                seqAccOutput = threadAccOutput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs
        for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
            const int index = tileBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i);
            outputs[index] = threadAccOutput[i];
        }
        memio::store<kT, memcode>(outputs, threadAccOutput, threadBaseIdx, threadSeqLen, rev, smem, tileBaseIdx, tileSeqLen);

        remainingElems -= tileSeqLen; // update remainingElems
        tileBaseIdx += (!rev ? 1 : -1) * min(remainingElems, elemsPerTile); // update tileBaseIdx
    }

    return;
}




template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_pipe_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool rev) {
    kT threadAccDInput[kMaxElemsPerThread];
    kT threadAccCoeff[kMaxElemsPerThread];
    __shared__ kT seqAccOutput, seqAccCoeff;
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

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
    int tileBaseIdx = seqLen * blockIdx.x; // process sequences independently: inputs[tileBaseIdx+i]
    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;
    tileBaseIdx += !rev ? max(seqLen - elemsPerTile, 0) : 0; // if forward start with last tile
    
    int remainingElems = seqLen;
    while (remainingElems > 0) { // sequentially combine subsequences of size elemsPerTile if seqLen doesn't fit into one thread block

        const ushort tileSeqLen = min(remainingElems, elemsPerTile);  // elemsPerTile                 // length of the tile to scan with thread block
        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute subseqlen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = !rev ? ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)) : // backwards ordered base index to process by every thread
                                    ((threadId * elemsPerThread - max(threadTailId, 0)));                           // forward ordered base index

        // Load inputs and coeffs of tile into thread-local arrays
        for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
            const int index = tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i);
            assert((i < threadSeqLen) ? (0 <= index && index < seqLen * (blockIdx.x + 1)) : true && "Error in index computation.");
            threadAccDInput[i] = (i < threadSeqLen) ? d_outputs[index] : 0;
            threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[index] : 1;
        }
        memio::load<kT, memcode>(threadAccDInput, d_outputs, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread, smem, tileBaseIdx, tileSeqLen);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread, smem, tileBaseIdx, tileSeqLen);
        

        //
        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccDInput and -Gate 
            if (threadId == 0 && remainingElems < seqLen){
                threadAccDInput[0] = seqAccOutput * threadAccCoeff[0] + threadAccDInput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
        
            // Store last threadAccDInput and -Gate into seqAccOutput and and -Gate
            if (threadId == numThreads-1 && elemsPerTile < remainingElems) {
                seqAccOutput = threadAccDInput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs of Back Propoagation Through Time
        for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
            const int index = tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i);
            d_inputs[index] = threadAccDInput[i];
        }
        memio::store<kT, memcode>(d_inputs, threadAccDInput, threadBaseIdx, threadSeqLen, !rev, smem, tileBaseIdx, tileSeqLen);
        
        
        //
        // Compute and Store Coefficient Derivatives (element-wise shifted multiplication) 
        kT* threadDCoeff = threadAccCoeff;
        
        // Load outputs shifted to the right (index -1) or if reverse shifted to the left (index +1) 
        // - edge case for (!rev && lastTile): out of bounds at outputs[-1] => tileShBaseIdx=0, tileShSeqLen=tileSeqLen-1
        // - edge case for (rev && lastTile): out of bounds at outputs[seqLen] => tileShSeqLen=tileSeqLen-1
        // For all threads in lastTile: shift like in linrec_tile.cuh since tileShSeqLen=tileSeqLen-1
        // - for (!rev && lastThread in lastTile): out of bounds at outputs[-1] => threadShBaseIdx=0, threadShSeqLen=threadSeqLen-1
        // - for (rev && lastThread in lastTile): out of bounds at outputs[seqLen] => threadShSeqLen=threadSeqLen-1
        bool lastTile = (remainingElems == tileSeqLen);
        const int tileShBaseIdx = tileBaseIdx + (!rev ? (lastTile ? 0:-1) : +1);
        const ushort tileShSeqLen = tileSeqLen + (lastTile ? -1 : 0); 

        bool lastThread = (!rev && threadBaseIdx==0) || (rev && threadBaseIdx+threadSeqLen==tileSeqLen);
        const ushort threadShBaseIdx = threadBaseIdx + ((!rev && lastTile && !lastThread) ? -1 : 0 );
        const ushort threadShSeqLen = threadSeqLen + ((lastTile && lastThread && threadSeqLen>0) ? -1 : 0);


        for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
            const int index = tileShBaseIdx + threadShBaseIdx + (!rev ? (threadShSeqLen-i-1) : i);
            assert((i < threadShSeqLen) ? (0 <= index && index < seqLen * (blockIdx.x + 1)) : true && "Error in index computation.");
            threadDCoeff[i] = (i < threadShSeqLen) ? outputs[index] : 0;
        }
        memio::load<kT, memcode>(threadDCoeff, outputs, threadShBaseIdx, threadShSeqLen, !rev, kT(0), kMaxElemsPerThread, smem, tileShBaseIdx, tileShSeqLen);
        
        for(ushort i = 0; 0 < algocode && i < kMaxElemsPerThread; i++) {     // element-wise multiplication
            threadDCoeff[i] *= threadAccDInput[i];
        }
        
        for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
            const int index = tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i);
            d_coeffs[index] = threadDCoeff[i];
        }
        memio::store<kT, memcode>(d_coeffs, threadDCoeff, threadBaseIdx, threadSeqLen, !rev, smem, tileBaseIdx, tileSeqLen);    

        remainingElems -= tileSeqLen; // update remainingElems
        tileBaseIdx -= (!rev ? 1 : -1) * min(remainingElems, elemsPerTile); // update tileBaseIdx
    }
}
// new line to avoid ptx syntax error "Parsing error near ''"