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
        count_t j = !rev ? i : (elemsPerThread-1)-i;
        dst[i] = src[j];
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_naive(kT* __restrict__ dst, const kT* __restrict__ src,  const count_t dstElemsPerThread, const count_t srcElemsPerThread, const bool rev, const kT fillval) {
    for (count_t i = 0; i < dstElemsPerThread; i++) {
        count_t j = !rev ? i : (srcElemsPerThread-1)-i;
        dst[i] = (i < srcElemsPerThread) ? src[j] : fillval;
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_naive(kT* __restrict__ dst, const kT* __restrict__ src,  const count_t dstElemsPerThread, const count_t srcElemsPerThread, const bool rev, const kT fillval, const count_t offset) {
    count_t lo = offset;
    count_t hi = offset+srcElemsPerThread;
    
    for (count_t i = 0; i < dstElemsPerThread; i++) {
        count_t j = !rev ? i-lo : (hi-1)-i;
        dst[i] = (lo <= i && i < hi) ? src[j] : fillval;
    }
}


template <typename kT, typename count_t>
__forceinline__  __device__  void copy_coalesced(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerBlock) {
    const ushort numThreads = blockDim.x;
    const ushort threadId = threadIdx.x;

    for (count_t i = threadId; i < elemsPerBlock; i += numThreads) {
        dst[i] = src[i]; 
    }
}

template <typename kT, typename count_t>
__forceinline__  __device__  void copy_coalesced16(kT* __restrict__ dst, const kT* __restrict__ src, const count_t elemsPerBlock) {
    const ushort numThreads = blockDim.x;
    const ushort threadId = threadIdx.x;

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
    const ushort threadId = threadIdx.x; 

    const ushort k = 16 / sizeof(kT);        // num elements in vectorized loads/stores of 16 byte words
    struct __align__(k * sizeof(kT))  vec {kT data[k];};  // wrapper type for vectorized loads/stores of 16 byte words

    // copy elementwise until dst[i] and src[i] are aligned
    //assert(((long) &src[offset] % 16 == 0) && "Dst is not aligned.");
    //assert(((long) &dst[offset] % 16 == 0) && "Src is not aligned.");

    const ushort offset = k - align;
    for (count_t i = threadId; i < offset; i += numThreads) {
        dst[i] = src[i];
    }

    // copy as vectors so long as they fit into elemsPerBlock
    count_t elemsPerBlockVec = elemsPerBlock - ((elemsPerBlock - offset) % k);
    for (count_t i = offset + k*threadId; i < elemsPerBlockVec; i += k*numThreads) {
        *reinterpret_cast<vec*>(&dst[i]) = *reinterpret_cast<const vec*>(&src[i]);
    }

    // copy elementwise until elemsPerBlock is reached
    for (count_t i = elemsPerBlockVec + threadId; i < elemsPerBlock; i += numThreads) {
        dst[i] = src[i]; 
    }
}


__forceinline__  __device__  ushort shiftIdx(const int seqLen, int &tileBaseIdx, ushort &tileSeqLen, ushort &threadBaseIdx, ushort &threadSeqLen, short shift = 0) {
    // clamp(tileShBaseIdx, 0, seqLen)
    int tileShBaseIdx = (shift < tileBaseIdx) ? (tileBaseIdx - shift) : 0;
    tileShBaseIdx = (tileBaseIdx <= seqLen + shift) ? tileShBaseIdx : seqLen;   

    // clamp(tileShBaseIdx + tileShSeqLen, max(tileBaseIdx + tileSeqLen, 0), seqLen)
    ushort tileShSeqLen = (shift < tileBaseIdx + tileSeqLen) ? tileBaseIdx + tileSeqLen - shift - tileShBaseIdx : 0;
    tileShSeqLen = (tileShBaseIdx + tileShSeqLen <= seqLen) ? tileShSeqLen : seqLen - tileShBaseIdx;

    // clamp(tileShBaseIdx + threadShBaseIdx, 0, tileShSeqLen)
    shift = (tileBaseIdx < shift) ? shift - tileBaseIdx : 0;
    ushort threadShBaseIdx = (shift < threadBaseIdx) ? (threadBaseIdx - shift) : 0;
    threadShBaseIdx = (threadBaseIdx <= tileShSeqLen + shift) ? threadShBaseIdx : tileShSeqLen;   

    // clamp(threadShBaseIdx + threadShSeqLen, max(threadBaseIdx + threadSeqLen, 0), tileShSeqLen)
    ushort threadShSeqLen = (shift < threadBaseIdx + threadSeqLen) ? threadBaseIdx + threadSeqLen - shift - threadShBaseIdx : 0;
    threadShSeqLen = (threadShBaseIdx + threadShSeqLen <= tileShSeqLen) ? threadShSeqLen : tileShSeqLen - threadShBaseIdx;
    
    ushort diff = threadSeqLen - threadShSeqLen; // difference in sequence length
    tileBaseIdx = tileShBaseIdx;
    tileSeqLen = tileShSeqLen;
    threadBaseIdx = threadShBaseIdx;
    threadSeqLen = threadShSeqLen;
    return diff;
}


template <typename kT, int memcode>
__forceinline__  __device__  void load(kT* dst, const kT* src, const int seqLen, kT* smem, int tileBaseIdx, ushort tileSeqLen, ushort threadBaseIdx, ushort threadSeqLen, const bool rev, const kT fill, const ushort maxElemsPerThread, const short shift = 0) {
    ushort offset = shiftIdx(seqLen, tileBaseIdx, tileSeqLen, threadBaseIdx, threadSeqLen, shift);
    offset = (!rev && threadBaseIdx==0) || (rev && tileBaseIdx+threadBaseIdx+threadSeqLen==seqLen) ? offset : 0;

    if (memcode < 0) {
        assert(0 <= tileBaseIdx && "Memory error");
        assert(tileBaseIdx + tileSeqLen <= seqLen && "Memory error");
        assert(threadBaseIdx + threadSeqLen <= tileSeqLen && "Memory error");
        assert(tileBaseIdx + threadBaseIdx + threadSeqLen <= seqLen && "Memory error");
        copy_naive(dst, &src[tileBaseIdx + threadBaseIdx], maxElemsPerThread, threadSeqLen, rev, fill, offset);
    }

    if (memcode==0) {
        copy_naive(dst, &src[tileBaseIdx + threadBaseIdx], maxElemsPerThread, threadSeqLen, rev, fill, offset);
    } else if (memcode==1) {
        __syncthreads(); // avoid race condition
        copy_naive(&smem[threadBaseIdx], &src[tileBaseIdx + threadBaseIdx], threadSeqLen);
        copy_naive(dst, &smem[threadBaseIdx], maxElemsPerThread, threadSeqLen, rev, fill, offset);
    } else if (memcode==2) {
        __syncthreads(); // avoid race condition
        const ushort align = ((long) &src[tileBaseIdx] % 16) / sizeof(kT);
        copy_coalesced16(&smem[align], &src[tileBaseIdx], tileSeqLen, align);
        __syncthreads(); // avoid race condition
        copy_naive(dst, &smem[align + threadBaseIdx], maxElemsPerThread, threadSeqLen, rev, fill, offset);
    }
}

template <typename kT, int memcode>
__forceinline__  __device__  void store(kT* dst, const kT* src, const int seqLen, kT* smem, const int tileBaseIdx, const ushort tileSeqLen, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev) {

    if (memcode < 0) {
        assert(0 <= tileBaseIdx && "Memory error");
        assert(tileBaseIdx + tileSeqLen <= seqLen && "Memory error");
        assert(threadBaseIdx + threadSeqLen <= tileSeqLen && "Memory error");
        assert(tileBaseIdx + threadBaseIdx + threadSeqLen <= seqLen && "Memory error");
        copy_naive(&dst[tileBaseIdx + threadBaseIdx], src, threadSeqLen, rev); 
    }

    if (memcode==0) {
        copy_naive(&dst[tileBaseIdx + threadBaseIdx], src, threadSeqLen, rev); 
    } else if (memcode==1) {
        __syncthreads(); // avoid race condition
        copy_naive(&smem[threadBaseIdx], src, threadSeqLen, rev); 
        copy_naive(&dst[tileBaseIdx + threadBaseIdx], &smem[threadBaseIdx], threadSeqLen);
    } else if (memcode==2) {
        __syncthreads(); // avoid race condition
        copy_naive(&smem[threadBaseIdx], src, threadSeqLen, rev); 
        __syncthreads(); // avoid race condition
        copy_coalesced16(&dst[tileBaseIdx], smem, tileSeqLen);
    }
}


template <typename kT, int memcode>
__forceinline__  __device__  void load(kT* dst, const kT* src, const int seqLen, kT* smem, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev, const kT fill, const ushort maxElemsPerThread, const short shift = 0) {
    load<kT, memcode>(dst, src, seqLen, smem, 0, seqLen, threadBaseIdx, threadSeqLen, rev, fill, maxElemsPerThread, shift);
}

template <typename kT, int memcode>
__forceinline__  __device__  void store(kT* dst, const kT* src, const int seqLen, kT* smem, const ushort threadBaseIdx, const ushort threadSeqLen, const bool rev) {
    store<kT, memcode>(dst, src, seqLen, smem, 0, seqLen, threadBaseIdx, threadSeqLen, rev);
}

} // namespace memio
