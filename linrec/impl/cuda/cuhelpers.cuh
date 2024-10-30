#pragma once
#include <iostream>
#include <cuda_runtime.h>

/**************************** CUDA Kernel Helpers ***************************/

// Integer ceiling division 
template <typename T>
constexpr T ceildiv(T lhs, T rhs){
    return ((lhs - 1) / rhs) + 1;
}

// Integer ceiling modulo 
template <typename T>
constexpr T ceilmod(T lhs, T rhs){
    return ((lhs - 1) % rhs) + 1;
}



/**************************** CUDA Runtime Helpers ***************************/

inline int cudaDeviceGetAttribute(cudaDeviceAttr attr, int device){
    int value;
    cudaDeviceGetAttribute(&value, attr, device);
    return value;
}

template<class T>
inline cudaFuncAttributes cudaFuncGetAttributes(T* entry){
    cudaFuncAttributes attrs;
    cudaFuncGetAttributes(&attrs, entry);
    return attrs;
}

inline std::map<std::string, int> cudaFuncAttributesAsMap(const cudaFuncAttributes &attrs){
    std::map<std::string, int> map;
    map["sharedSizeBytes"] = attrs.sharedSizeBytes; 
    map["constSizeBytes"] = attrs.constSizeBytes;
    map["localSizeBytes"] = attrs.localSizeBytes;
    map["maxThreadsPerBlock"] = attrs.maxThreadsPerBlock;
    map["numRegs"] = attrs.numRegs; 
    map["ptxVersion"] = attrs.ptxVersion;
    map["binaryVersion"] = attrs.binaryVersion;
    map["cacheModeCA"] = attrs.cacheModeCA;
    map["maxDynamicSharedSizeBytes"] = attrs.maxDynamicSharedSizeBytes;
    map["preferredShmemCarveout"] = attrs.preferredShmemCarveout;
    map["clusterDimMustBeSet"] = attrs.clusterDimMustBeSet;
    map["requiredClusterWidth"] = attrs.requiredClusterWidth;
    map["requiredClusterHeight"] = attrs.requiredClusterHeight;
    map["requiredClusterDepth"] = attrs.requiredClusterDepth;
    map["clusterSchedulingPolicyPreference"] = attrs.clusterSchedulingPolicyPreference;
    map["nonPortableClusterSizeAllowed"] = attrs.nonPortableClusterSizeAllowed;
    return map;
}

inline std::ostream& operator<<(std::ostream& os, const cudaFuncAttributes attr)
{
   return os << "cudaFuncAttributes[" << std::endl <<
            "    sharedSizeBytes=" << attr.sharedSizeBytes << std::endl << 
            "    constSizeBytes=" << attr.constSizeBytes << std::endl <<
            "    localSizeBytes=" << attr.localSizeBytes << std::endl <<
            "    maxThreadsPerBlock=" << attr.maxThreadsPerBlock << std::endl <<
            "    numRegs=" << attr.numRegs << std::endl << 
            "    ptxVersion=" << attr.ptxVersion << std::endl <<
            "    binaryVersion=" << attr.binaryVersion << std::endl <<
            "    cacheModeCA=" << attr.cacheModeCA << std::endl <<
            "    maxDynamicSharedSizeBytes=" << attr.maxDynamicSharedSizeBytes << std::endl <<
            "    preferredShmemCarveout=" << attr.preferredShmemCarveout << std::endl <<
            "    clusterDimMustBeSet=" << attr.clusterDimMustBeSet << std::endl <<
            "    requiredClusterWidth=" << attr.requiredClusterWidth << std::endl <<
            "    requiredClusterHeight=" << attr.requiredClusterHeight << std::endl <<
            "    requiredClusterDepth=" << attr.requiredClusterDepth << std::endl <<
            "    clusterSchedulingPolicyPreference=" << attr.clusterSchedulingPolicyPreference << std::endl <<
            "    nonPortableClusterSizeAllowed=" << attr.nonPortableClusterSizeAllowed << std::endl <<
            "]";
}; 