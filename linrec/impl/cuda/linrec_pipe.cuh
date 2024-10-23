#pragma once
#include <cuda.h>
#include <memio.cuh>
#include <linrec_tile.cuh>


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_pipe_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen, const bool rev) {
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently in reverse: inputs[seqBaseIdx-i]
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
    

    __shared__ kT seqAccOutput, seqAccCoeff; // for comunication between tiles
    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;
    int tileBaseIdx = !rev ? 0 : max(seqLen - elemsPerTile, 0); // if reverse start with last tile
    while (true) {  // sequentially combine subsequences of size elemsPerTile if seqLen doesn't fit into one thread block
    
        const bool firstTile = !rev ? (tileBaseIdx <= 0) : (seqLen-elemsPerTile <= tileBaseIdx);
        const bool lastTile = !rev ? (seqLen-elemsPerTile <= tileBaseIdx) : (tileBaseIdx <= 0);
        const ushort tileSeqLen = !lastTile ? elemsPerTile : ((seqLen-1)%elemsPerTile)+1;             // length of the tile to scan with thread block

        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute subseqlen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = !rev ? (threadId * elemsPerThread - max(threadTailId, 0)) :                     // base index to process by every thread
                                        ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)); // reverse ordered base index


        //
        // Load inputs and coeffs of tile into thread-local arrays
        kT threadAccOutput[kMaxElemsPerThread];
        kT threadAccCoeff[kMaxElemsPerThread];
        for (ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
            threadAccOutput[i] = (i < threadSeqLen) ? inputs[tileBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i)] : 0;
            threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[tileBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i)] : 1;
        }
        memio::load<kT, memcode>(threadAccOutput, inputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, rev, kT(0), kMaxElemsPerThread);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, rev, kT(1), kMaxElemsPerThread);

        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccOutput and -Gate 
            if (threadId == 0 && !firstTile){
                threadAccOutput[0] = seqAccOutput * threadAccCoeff[0] + threadAccOutput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
        
            // Store last threadAccOutput and -Gate into seqAccOutput and and -Gate
            if (threadId == numThreads-1 && !lastTile) {
                seqAccOutput = threadAccOutput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs
        for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
            outputs[tileBaseIdx + threadBaseIdx + (!rev ? i : threadSeqLen-1-i)] = threadAccOutput[i];
        }
        memio::store<kT, memcode>(outputs, threadAccOutput, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, rev);


        //
        // Update tileBaseIdx or return
        if (lastTile)
            return;
        tileBaseIdx = !rev ? tileBaseIdx+elemsPerTile : max(tileBaseIdx-elemsPerTile, 0);
    }
}


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_pipe_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool rev) {
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


    __shared__ kT seqAccOutput, seqAccCoeff; // for comunication between tiles
    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;
    int tileBaseIdx = !rev ? max(seqLen - elemsPerTile, 0) : 0; // if forward start with last tile
    while (true) {  // sequentially combine subsequences of size elemsPerTile if seqLen doesn't fit into one thread block

        const bool firstTile = !rev ? (seqLen-elemsPerTile <= tileBaseIdx) : (tileBaseIdx <= 0);
        const bool lastTile = !rev ? (tileBaseIdx <= 0) : (seqLen-elemsPerTile <= tileBaseIdx);
        const ushort tileSeqLen = !lastTile ? elemsPerTile : ((seqLen-1)%elemsPerTile)+1;               // length of the tile to scan with thread block

        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute subseqlen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadId - (numThreads - numTailThreads);                        // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = !rev ? ((numThreads-1-threadId) * (elemsPerThread-1) + max(-threadTailId-1, 0)) : // backwards ordered base index to process by every thread
                                    ((threadId * elemsPerThread - max(threadTailId, 0)));                           // forward ordered base index


        //
        // Load inputs and coeffs of tile into thread-local arrays
        kT threadAccDInput[kMaxElemsPerThread];
        kT threadAccCoeff[kMaxElemsPerThread];
        for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
            threadAccDInput[i] = (i < threadSeqLen) ? d_outputs[tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] : 0;
            threadAccCoeff[i] = (i < threadSeqLen) ? coeffs[tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] : 1;
        }
        memio::load<kT, memcode>(threadAccDInput, d_outputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread);

        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccDInput and -Gate 
            if (threadId == 0 && !firstTile){
                threadAccDInput[0] = seqAccOutput * threadAccCoeff[0] + threadAccDInput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, laneId, thisWarpSize, warpId, numWarps, numThreads);
        
            // Store last threadAccDInput and -Gate into seqAccOutput and and -Gate
            if (threadId == numThreads-1 && !lastTile) {
                seqAccOutput = threadAccDInput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs of Back Propoagation Through Time
        for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
            d_inputs[tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] = threadAccDInput[i];
        }
        memio::store<kT, memcode>(d_inputs, threadAccDInput, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev);


        //
        // Compute and Store Coefficient Derivatives (element-wise shifted multiplication) 
        // Outputs are shifted to the right (index -1) or if reverse shifted to the left (index +1) 
        // - edge case for (!rev && lastTile): out of bounds at outputs[-1] => tileShBaseIdx=0, tileShSeqLen=tileSeqLen-1
        // - edge case for (rev && lastTile): out of bounds at outputs[seqLen] => tileShSeqLen=tileSeqLen-1
        const int tileShBaseIdx = tileBaseIdx + (!rev ? (lastTile ? 0:-1) : +1);
        const ushort tileShSeqLen = tileSeqLen + (lastTile ? -1 : 0); 

        // For all threads in lastTile: shift like in linrec_tile.cuh since tileShSeqLen=tileSeqLen-1
        bool lastThread = (!rev && threadBaseIdx==0) || (rev && threadBaseIdx+threadSeqLen==tileSeqLen);
        const ushort threadShBaseIdx = threadBaseIdx + ((!rev && lastTile && !lastThread) ? -1 : 0 ); // no +1!!
        const ushort threadShSeqLen = threadSeqLen + ((lastTile && lastThread && threadSeqLen>0) ? -1 : 0);

        // Load shifted outputs of tile into thread-local arrays
        kT threadDCoeff[kMaxElemsPerThread];
        for(ushort i = 0; memcode < 0 && i < kMaxElemsPerThread; i++) {
            threadDCoeff[i] = (i < threadShSeqLen) ? outputs[tileShBaseIdx + threadShBaseIdx + (!rev ? (threadShSeqLen-i-1) : i)] : 0;
        }
        memio::load<kT, memcode>(threadDCoeff, outputs, seqLen, smem, tileShBaseIdx, tileShSeqLen, threadShBaseIdx, threadShSeqLen, !rev, kT(0), kMaxElemsPerThread);
        
        // shifted element-wise multiplication
        for(ushort i = 0; 0 < algocode && i < kMaxElemsPerThread; i++) {
            threadDCoeff[i] *= threadAccDInput[i];
        }
        
        // Store outputs of shifted element-wise multiplication
        for(ushort i = 0; memcode < 0 && i < threadSeqLen; i++) {
            d_coeffs[tileBaseIdx + threadBaseIdx + (!rev ? (threadSeqLen-1-i) : i)] = threadDCoeff[i];
        }
        memio::store<kT, memcode>(d_coeffs, threadDCoeff, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev);


        //
        // Update tileBaseIdx or return
        if (!rev ? (tileBaseIdx <= 0) : (seqLen-elemsPerTile <= tileBaseIdx)) // recompute lastTile saves 1 register
            return;
        tileBaseIdx = !rev ? max(tileBaseIdx-elemsPerTile, 0) : tileBaseIdx+elemsPerTile;
    }
}
// new line to avoid ptx syntax error "Parsing error near ''"