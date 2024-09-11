#pragma once
#include <iostream>
#include <cuda_runtime.h>


/**************************** CUDA Runtime Helpers ***************************/

int cudaDeviceGetAttribute(cudaDeviceAttr attr, int device){
    int value;
    cudaDeviceGetAttribute(&value, attr, device);
    return value;
}

template<class T>
cudaFuncAttributes cudaFuncGetAttributes(T* entry){
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, entry);
    return attr;
}

std::ostream& operator<<(std::ostream& os, cudaFuncAttributes attr)
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
