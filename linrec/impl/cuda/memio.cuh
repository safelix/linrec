#pragma once
#include <cuda.h>
#include <cuda_pipeline.h>


namespace memio{

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
// https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#device-memory-spaces
// https://on-demand.gputechconf.com/gtc/2012/presentations/S0514-GTC2012-GPU-Performance-Analysis.pdf
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
template <typename kT, typename count_t>
__forceinline__  __device__  void copy_naive(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerThread) {
    for (count_t i = 0; i < elemsPerThread; i++) {
        dst[i] = src[i];
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_naive(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerThread, const bool rev) {
    for (count_t i = 0; i < elemsPerThread; i++) {
        count_t irev = !rev ? i : (elemsPerThread-i-1);
        dst[i] = src[irev];
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_naive(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerThread, const bool rev, const kT fill, const count_t maxElemsPerThread) {
    for (count_t i = 0; i < maxElemsPerThread; i++) {
        count_t irev = !rev ? i : (elemsPerThread-i-1);
        dst[i] = (i < elemsPerThread) ? src[irev] : fill;
    }
}


template <typename kT, typename count_t>
__forceinline__  __device__  void copy_coalesced(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerBlock) {
    const ushort numThreads = blockDim.x;
    const ushort threadId = threadIdx.x; // TODO: compute real threadIdx.z threadIdx.y 

    for (count_t i = threadId; i < elemsPerBlock; i += numThreads) {
        dst[i] = src[i]; 
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_coalesced16(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerBlock) {
    const ushort numThreads = blockDim.x;
    const ushort threadId = threadIdx.x; // TODO: compute real threadIdx.z threadIdx.y 

    const ushort k = 16 / sizeof(kT);        // num elements in vectorized loads/stores of 16 byte words
    struct __align__(k * sizeof(kT))  vec {kT data[k];};  // wrapper type for vectorized loads/stores of 16 byte words
    //assert((long)  dst % 16 == 0 && "Dst needs to be a multiple of 16 for alignment");
    //assert((long) src % 16 == 0 && "Src needs to be a multiple of 16 for alignment");

    // copy as vectors so long as they fit into elemsPerBlock
    count_t elemsPerBlockVec = elemsPerBlock - (elemsPerBlock % k);
    for (count_t i = k*threadId; i < elemsPerBlockVec; i += k*numThreads) {
        *reinterpret_cast<vec*>(&dst[i]) = *reinterpret_cast<const vec*>(&src[i]);
    }

    // copy elementwise until elemsPerBlock is reached
    for (count_t i = elemsPerBlockVec + threadId; i < elemsPerBlock; i += numThreads) {
        dst[i] = src[i]; 
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_coalesced16(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerBlock, const ushort align) {
    const ushort numThreads = blockDim.x;
    const ushort threadId = threadIdx.x; // TODO: compute real threadIdx.z threadIdx.y 

    const ushort k = 16 / sizeof(kT);        // num elements in vectorized loads/stores of 16 byte words
    struct __align__(k * sizeof(kT))  vec {kT data[k];};  // wrapper type for vectorized loads/stores of 16 byte words

    // copy elementwise until dst[i] and src[i] are aligned
    //assert(((long) &src[align] % 16 == 0) && "Dst is not aligned.");
    //assert(((long) &dst[align] % 16 == 0) && "Src is not aligned.");

    for (count_t i = threadId; i < align; i += numThreads) {
        dst[i] = src[i];
    }

    // copy as vectors so long as they fit into elemsPerBlock
    count_t elemsPerBlockVec = elemsPerBlock - ((elemsPerBlock - align) % k);
    for (count_t i = align + k*threadId; i < elemsPerBlockVec; i += k*numThreads) {
        *reinterpret_cast<vec*>(&dst[i]) = *reinterpret_cast<const vec*>(&src[i]);
    }

    // copy elementwise until elemsPerBlock is reached
    for (count_t i = elemsPerBlockVec + threadId; i < elemsPerBlock; i += numThreads) {
        dst[i] = src[i]; 
    }
}


template <typename kT, int memcode>
__forceinline__  __device__  void load(kT* dst, const kT* src, const int seqLen, kT* smem, const int tileBaseIdx, const ushort tileSeqLen, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev, const kT fill, const ushort maxElemsPerThread) {

    if (memcode==0) {
        copy_naive(dst, &src[tileBaseIdx + threadBaseIdx], threadSeqLen, rev, fill, maxElemsPerThread);
    } else if (memcode==1) {
        __syncthreads(); // avoid race condition
        copy_naive(&smem[threadBaseIdx], &src[tileBaseIdx + threadBaseIdx], threadSeqLen);
        copy_naive(dst, &smem[threadBaseIdx], threadSeqLen, rev, fill, maxElemsPerThread);
    } else if (memcode==2) {
        __syncthreads(); // avoid race condition
        const ushort offset = ((long) &src[tileBaseIdx] % 16) / sizeof(kT);
        const ushort align = 16 / sizeof(kT) - offset;
        copy_coalesced16(&smem[offset], &src[tileBaseIdx], tileSeqLen, align);
        __syncthreads();
        copy_naive(dst, &smem[offset + threadBaseIdx], threadSeqLen, rev, fill, maxElemsPerThread);
    }
}

template <typename kT, int memcode>
__forceinline__  __device__  void store(kT* dst, const kT* src, const int seqLen, kT* smem, const int tileBaseIdx, const ushort tileSeqLen, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev) {

    if (memcode==0) {
        copy_naive(&dst[tileBaseIdx + threadBaseIdx], src, threadSeqLen, rev); 
    } else if (memcode==1) {
        __syncthreads(); // avoid race condition
        copy_naive(&smem[threadBaseIdx], src, threadSeqLen, rev); 
        copy_naive(&dst[tileBaseIdx + threadBaseIdx], &smem[threadBaseIdx], threadSeqLen);
    } else if (memcode==2) {
        __syncthreads(); // avoid race condition
        copy_naive(&smem[threadBaseIdx], src, threadSeqLen, rev); 
        __syncthreads();
        copy_coalesced16(&dst[tileBaseIdx], smem, tileSeqLen);
    }
}


template <typename kT, int memcode>
__forceinline__  __device__  void load(kT* dst, const kT* src, const int seqLen, kT* smem, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev, const kT fill, const ushort maxElemsPerThread) {
    load<kT, memcode>(dst, src, seqLen, smem, 0, seqLen, threadBaseIdx, threadSeqLen, rev, fill, maxElemsPerThread);
}

template <typename kT, int memcode>
__forceinline__  __device__  void store(kT* dst, const kT* src, const int seqLen, kT* smem, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev) {
    store<kT, memcode>(dst, src, seqLen, smem, 0, seqLen, threadBaseIdx, threadSeqLen, rev);
}

} // namespace memio
