#pragma once
#include <cuda.h>
#include <memio.cuh>
#include <cuhelpers.cuh>
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

    __shared__ kT seqAccOutput, seqAccCoeff; // for sequential accumulation between tiles
    if (threadIdx.x == 0) {
        seqAccOutput = 0;
        seqAccCoeff = 1;
    } __syncwarp(); // avoid divergence

    // Determine Tile Layout
    const ushort numThreads = blockDim.x;
    const ushort threadIdrev = !rev ? threadIdx.x : (numThreads - threadIdx.x - 1);                     // thread index, reversed to load reversed
    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;                                        // the default number of elements per tile

    int tileBaseIdx = !rev ? 0 : (seqLen - ceilmod(seqLen, (int) elemsPerTile));                        // if reverse start with last tile
    for (; !rev ? (tileBaseIdx < seqLen) : (0 <= tileBaseIdx); tileBaseIdx += !rev ? elemsPerTile : -elemsPerTile) { // linear scan over tiles

        const ushort tileSeqLen = min(seqLen - tileBaseIdx, elemsPerTile);                              // length of the tile to scan with thread block
        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute tileSeqLen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadIdrev - (numThreads - numTailThreads);                     // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = threadIdrev * elemsPerThread - max(threadTailId, 0);               // base index to process by every thread


        //
        // Load inputs and coeffs of tile into thread-local arrays
        kT threadAccOutput[kMaxElemsPerThread];
        kT threadAccCoeff[kMaxElemsPerThread];
        memio::load<kT, memcode>(threadAccOutput, inputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, rev, kT(0), kMaxElemsPerThread);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, rev, kT(1), kMaxElemsPerThread);

        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccOutput and -Gate 
            if (threadIdx.x == 0){
                threadAccOutput[0] = seqAccOutput * threadAccCoeff[0] + threadAccOutput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid race condition

            _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, numThreads);
        
            // Store last threadAccOutput and -Gate into seqAccOutput and and -Gate
            if (threadIdx.x == numThreads-1) {
                seqAccOutput = threadAccOutput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid race condition
        }

        // Store outputs
        memio::store<kT, memcode>(outputs, threadAccOutput, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, rev);
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

    __shared__ kT seqAccOutput, seqAccCoeff; // for sequential accumulation between tiles
    if (threadIdx.x == 0) {
        seqAccOutput = 0;
        seqAccCoeff = 1;
    } __syncwarp(); // avoid divergence

    // Determine Tile Layout
    const ushort numThreads = blockDim.x;
    const ushort threadIdrev = !rev ? (numThreads - threadIdx.x - 1) : threadIdx.x;                     // reversed thread index to load reversed by default
    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;                                        // the default number of elements per tile

    int tileBaseIdx = !rev ? (seqLen - ceilmod(seqLen, (int) elemsPerTile)) : 0;                        // if reverse start with last tile
    for (; !rev ? (0 <= tileBaseIdx) : (tileBaseIdx < seqLen); tileBaseIdx += !rev ? -elemsPerTile : elemsPerTile) { // linear scan over tiles

        const ushort tileSeqLen = min(seqLen - tileBaseIdx, elemsPerTile);                              // length of the tile to scan with thread block
        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute tileSeqLen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadIdrev - (numThreads - numTailThreads);                     // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = threadIdrev * elemsPerThread - max(threadTailId, 0);               // base index to process by every thread


        //
        // Load inputs and coeffs of tile into thread-local arrays
        kT threadAccDInput[kMaxElemsPerThread];
        kT threadAccCoeff[kMaxElemsPerThread];
        memio::load<kT, memcode>(threadAccDInput, d_outputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread, !rev ? -1 : 1);

        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccDInput and -Gate 
            if (threadIdx.x == 0){
                threadAccDInput[0] = seqAccOutput * threadAccCoeff[0] + threadAccDInput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, numThreads);
        
            // Store last threadAccDInput and -Gate into seqAccOutput and and -Gate
            if (threadIdx.x == numThreads-1) {
                seqAccOutput = threadAccDInput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs of Back Propoagation Through Time
        memio::store<kT, memcode>(d_inputs, threadAccDInput, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev);


        //
        // Load outputs shifted to the right or if reverse shifted to the left
        kT threadDCoeff[kMaxElemsPerThread];
        memio::load<kT, memcode>(threadDCoeff, outputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread, !rev ? 1 : -1);
        
        // Compute shifted element-wise multiplication
        for(ushort i = 0; 0 < algocode && i < kMaxElemsPerThread; i++) {
            threadDCoeff[i] *= threadAccDInput[i];
        }
        
        // Store outputs of shifted element-wise multiplication
        memio::store<kT, memcode>(d_coeffs, threadDCoeff, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev);
    }
}
// new line to avoid ptx syntax error "Parsing error near ''"