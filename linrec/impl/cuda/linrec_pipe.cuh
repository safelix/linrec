#pragma once
#include <cuda.h>
#include <memio.cuh>
#include <cuhelpers.cuh>
#include <linrec_tile.cuh>


template <typename kT, ushort kMaxElemsPerThread, ushort kMaxThreadsPerWarp, ushort kMaxThreadsPerBlock, int memcode, int algocode>
__global__ void __launch_bounds__(kMaxThreadsPerBlock)
linrec_pipe_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, int const seqLen, const bool rev) {
    extern __shared__ kT smem[]; // smem[kMaxElemsPerThread * kMaxThreadsPerBlock];
    __shared__ kT seqAccOutput, seqAccCoeff; // for sequential accumulation between tiles

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently in reverse: inputs[seqBaseIdx-i]
    inputs = &inputs[seqBaseIdx];
    coeffs = &coeffs[seqBaseIdx];
    outputs = &outputs[seqBaseIdx];

    // Determine Tile Layout
    const ushort numThreads = kMaxThreadsPerBlock; //blockDim.x;
    const ushort threadIdrev = !rev ? threadIdx.x : (numThreads - threadIdx.x - 1);                     // thread index, reversed to load reversed

    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;
    int tileBaseIdx = !rev ? 0 : max(seqLen - elemsPerTile, 0); // if reverse start with last tile
    while (true) {  // sequentially combine subsequences of size elemsPerTile if seqLen doesn't fit into one thread block
    
        const bool firstTile = !rev ? (tileBaseIdx <= 0) : (seqLen-elemsPerTile <= tileBaseIdx);
        const bool lastTile = !rev ? (seqLen-elemsPerTile <= tileBaseIdx) : (tileBaseIdx <= 0);
        const ushort tileSeqLen = !lastTile ? elemsPerTile : ceilmod(seqLen, (int) elemsPerTile);       // length of the tile to scan with thread block

        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute subseqlen among numThreads
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
            if (threadIdx.x == 0 && !firstTile){
                threadAccOutput[0] = seqAccOutput * threadAccCoeff[0] + threadAccOutput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccOutput, threadAccCoeff, kMaxElemsPerThread, numThreads);
        
            // Store last threadAccOutput and -Gate into seqAccOutput and and -Gate
            if (threadIdx.x == numThreads-1 && !lastTile) {
                seqAccOutput = threadAccOutput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs
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
    __shared__ kT seqAccOutput, seqAccCoeff; // for sequential accumulation between tiles

    // Layout: dim=(X,L), strides=(L,1)
    const int seqBaseIdx = seqLen * blockIdx.x; // process sequences independently in reverse: inputs[seqBaseIdx-i]
    d_outputs = &d_outputs[seqBaseIdx];
    coeffs = &coeffs[seqBaseIdx];
    outputs = &outputs[seqBaseIdx];
    d_inputs = &d_inputs[seqBaseIdx];
    d_coeffs = &d_coeffs[seqBaseIdx];

    // Determine Tile Layout
    const ushort numThreads = kMaxThreadsPerBlock; // blockDim.x;
    const ushort threadIdrev = !rev ? (numThreads - threadIdx.x - 1) : threadIdx.x;                     // reversed thread index to load reversed by default

    const ushort elemsPerTile = kMaxElemsPerThread * numThreads;
    int tileBaseIdx = !rev ? max(seqLen - elemsPerTile, 0) : 0; // if forward start with last tile
    while (true) {  // sequentially combine subsequences of size elemsPerTile if seqLen doesn't fit into one thread block

        const bool firstTile = !rev ? (seqLen-elemsPerTile <= tileBaseIdx) : (tileBaseIdx <= 0);
        const bool lastTile = !rev ? (tileBaseIdx <= 0) : (seqLen-elemsPerTile <= tileBaseIdx);
        const ushort tileSeqLen = !lastTile ? elemsPerTile : ceilmod(seqLen, (int) elemsPerTile);       // length of the tile to scan with thread block

        const ushort elemsPerThread = ceildiv(tileSeqLen, numThreads);                                  // distribute subseqlen among numThreads
        const ushort numTailThreads = numThreads * elemsPerThread - tileSeqLen;                         // last numTailThreads have one elem less
        const int threadTailId = (int) threadIdrev - (numThreads - numTailThreads);                     // tail start indicated by ..., 0, 1, 2, ...
        const ushort threadSeqLen = (threadTailId < 0) ? elemsPerThread : (elemsPerThread-1);           // sequence length processed by every thread
        const ushort threadBaseIdx = threadIdrev * elemsPerThread - max(threadTailId, 0);               // base index to process by every thread


        //
        // Load inputs and coeffs of tile into thread-local arrays
        kT threadAccDInput[kMaxElemsPerThread];
        kT threadAccCoeff[kMaxElemsPerThread];
        memio::load<kT, memcode>(threadAccDInput, d_outputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread);
        memio::load<kT, memcode>(threadAccCoeff, coeffs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(1), kMaxElemsPerThread);

        // Compute parallel scan on a tile (=subsequence) that fits into one thread block 
        if (algocode >= 1) { // level 1,2,3 of block-wise parallel scan

            // Combine seqAccOutput and and -Gate with first threadAccDInput and -Gate 
            if (threadIdx.x == 0 && !firstTile){
                threadAccDInput[0] = seqAccOutput * threadAccCoeff[0] + threadAccDInput[0];
                threadAccCoeff[0] =  seqAccCoeff * threadAccCoeff[0];
            } __syncthreads(); // avoid divergence

            _linrec_scan_tile_parallel_<kT, kMaxThreadsPerWarp, kMaxThreadsPerBlock, algocode>(threadAccDInput, threadAccCoeff, kMaxElemsPerThread, numThreads);
        
            // Store last threadAccDInput and -Gate into seqAccOutput and and -Gate
            if (threadIdx.x == numThreads-1 && !lastTile) {
                seqAccOutput = threadAccDInput[kMaxElemsPerThread-1];
                seqAccCoeff = threadAccCoeff[kMaxElemsPerThread-1];
            } __syncthreads(); // avoid divergence
        }

        // Store outputs of Back Propoagation Through Time
        memio::store<kT, memcode>(d_inputs, threadAccDInput, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev);


        //
        // Compute and Store Coefficient Derivatives (element-wise shifted multiplication) 
        // Load outputs shifted to the right or if reverse shifted to the left
        short shift = !rev ? 1 : -1;
        kT threadDCoeff[kMaxElemsPerThread];
        memio::load<kT, memcode>(threadDCoeff, outputs, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev, kT(0), kMaxElemsPerThread, shift);
        
        // shifted element-wise multiplication
        for(ushort i = 0; 0 < algocode && i < kMaxElemsPerThread; i++) {
            threadDCoeff[i] *= threadAccDInput[i];
        }
        
        // Store outputs of shifted element-wise multiplication
        memio::store<kT, memcode>(d_coeffs, threadDCoeff, seqLen, smem, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, !rev);


        //
        // Update tileBaseIdx or return
        if (!rev ? (tileBaseIdx <= 0) : (seqLen-elemsPerTile <= tileBaseIdx)) // recompute lastTile saves 1 register
            return;
        tileBaseIdx = !rev ? max(tileBaseIdx-elemsPerTile, 0) : tileBaseIdx+elemsPerTile;
    }
}
// new line to avoid ptx syntax error "Parsing error near ''"