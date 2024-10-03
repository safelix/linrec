#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>

#include <linrec.h>
#include <dispatch.h>
#include <cuhelpers.cuh>
#include <linrec_tile.cuh>

using torch::Tensor;

using Option = std::map<std::string, int>;
inline int get(const Option& option, const std::string key, int default_value) {   
    return option.contains(key) ? option.at(key) : default_value;
}

Tensor linrec_tile_fwd(const Tensor &inputs, const Tensor &coeffs, const bool reverse, const Option& options) {
    TORCH_CHECK(inputs.sizes() == coeffs.sizes());               // same dimensions
    TORCH_CHECK(inputs.strides() == coeffs.strides());           // same strides
    TORCH_CHECK(inputs.device() == coeffs.device());             // same device
    TORCH_CHECK(inputs.is_cuda() && coeffs.is_cuda());           // both cuda
    TORCH_CHECK(inputs.scalar_type() == coeffs.scalar_type());   // same dtype
    TORCH_CHECK(inputs.scalar_type() == torch::ScalarType::Float); // TODO: dispatch

    // Select correct CUDA device and it's current stream, otherwise current device and it's default stream (0x0) are used
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html (streams are only managed by torch)
    const c10::cuda::CUDAGuard guard(inputs.device());          // calls cudaSetDevice internally
    auto stream = c10::cuda::getCurrentCUDAStream().stream();   // current stream might not be 0x0

    // Prepare Outputs
    Tensor outputs = torch::empty_like(inputs);

    // Infer number of sequences and sequence length
    //TORCH_CHECK(inputs.is_contiguous());      // memory is ordered as dimensions
    TORCH_CHECK(inputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    int seqlen = inputs.size(-1);               // the sequence length
    int numseq = inputs.numel() / seqlen;       // the number of sequences over batches, channels, etc

    // Unpack and determine compile-time arguments
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = get(options, "kMaxElemsPerThread", 16); 
    int kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", 32); 
    int kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", 1024); 

    // Dispatch templated function: instantiate compile-time configuration
    auto config = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};

    dispatch<CONFIG_LIST>(config, [&]<auto config>() {
        static constexpr int kMaxElemsPerThread = config[0];
        static constexpr int kMaxThreadsPerWarp = config[1];
        static constexpr int kMaxThreadsPerBlock = config[2];
        static constexpr int memcode = config[3];
        static constexpr int algocode = config[4];

        // check validity of inputs with respect to compile-time arguments
        static constexpr int kMaxElemsPerTile = kMaxElemsPerThread * kMaxThreadsPerBlock;
        TORCH_CHECK(seqlen <= kMaxElemsPerTile && "Input sequence is longer than maximum tile size.");
        TORCH_CHECK(inputs.numel() <= (1UL << 8*sizeof(int)-1)-1 && "For int seqBaseIdx, numel() needs to be smaller than 2147483647.");
        TORCH_CHECK(kMaxElemsPerTile <= (1UL << 8*sizeof(ushort)) && "For ushort indexing, kMaxElemsPerTile needs to be smaller than 65536.");

        // select kernel based on compile-time arguments 
        auto kernel = linrec_tile_fwd_kernel<float, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode>;

        // determine run-time arguments
        int blocks = numseq;
        int threads = kMaxThreadsPerBlock; // TODO: std::min(seqlen, kMaxThreadsPerBlock);
        int smem = (memcode > 0) ? seqlen * sizeof(float) : 0;

        // configure kernel run-time properties: 
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
        int maxDynamicSmemSize = cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, (int)inputs.get_device());
        TORCH_CHECK(smem <= maxDynamicSmemSize && "Not enough shared memory to accomodate coalesced loading of a tile (with memcode>0).");
        
        if (memcode > 0) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 67);   // >65792 bytes (>67% for Volta or newer) required for kMaxElemsPerThread=16
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);    // only works if smem <= maxDynamicSmemSize
            int availDynamicSmemSize = cudaFuncGetAttributes(kernel).maxDynamicSharedSizeBytes;
            TORCH_CHECK(smem <= availDynamicSmemSize && "Not enough shared memory granted for coalesed loading of a tile (with memcode>0).");
        }

        // launch kernel
        kernel<<<blocks, threads, smem, stream>>>(
            inputs.data_ptr<float>(),
            coeffs.data_ptr<float>(), 
            outputs.data_ptr<float>(), 
            seqlen, reverse);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }, "linrec_tile_fwd_kernel", CONFIG_NAMES); // names for errors

    return outputs;
}



std::tuple<Tensor, Tensor> linrec_tile_bwd(const Tensor &d_outputs, const Tensor &coeffs, const Tensor &outputs, bool reverse, const Option& options) {
    TORCH_CHECK(d_outputs.sizes() == coeffs.sizes() && coeffs.sizes() == outputs.sizes());                          // same dimensions
    TORCH_CHECK(d_outputs.strides() == coeffs.strides() && coeffs.strides() == outputs.strides());                  // same strides
    TORCH_CHECK(d_outputs.device() == coeffs.device() && coeffs.device() == outputs.device());                      // same device
    TORCH_CHECK(d_outputs.is_cuda() && coeffs.is_cuda() && outputs.is_cuda());                                      // all cuda
    TORCH_CHECK(d_outputs.scalar_type() == coeffs.scalar_type() && coeffs.scalar_type() == outputs.scalar_type());  // same dtype
    TORCH_CHECK(d_outputs.scalar_type() == torch::ScalarType::Float); // TODO: dispatch

    // Select correct CUDA device and it's current stream, otherwise current device and it's default stream (0x0) are used
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html (streams are only managed by torch)
    const c10::cuda::CUDAGuard guard(d_outputs.device());       // calls cudaSetDevice internally
    auto stream = c10::cuda::getCurrentCUDAStream().stream();   // current stream might not be 0x0

    // Prepare Outputs
    Tensor d_inputs = torch::empty_like(d_outputs);
    Tensor d_coeffs = torch::empty_like(coeffs);

    // Infer number of sequences and sequence length
    TORCH_CHECK(d_outputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    const int seqlen = d_outputs.size(-1);         // the sequence length
    const int numseq = d_outputs.numel() / seqlen; // the number of sequences over batches, channels, etc

    // Unpack and determine compile-time arguments
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = get(options, "kMaxElemsPerThread", 16); 
    int kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", 32); 
    int kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", 1024); 

    // Dispatch templated function: instantiate compile-time configuration
    auto config = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};
    
    dispatch<CONFIG_LIST>(config, [&]<auto config>() {
        static constexpr int kMaxElemsPerThread = config[0];
        static constexpr int kMaxThreadsPerWarp = config[1];
        static constexpr int kMaxThreadsPerBlock = config[2];
        static constexpr int memcode = config[3];
        static constexpr int algocode = config[4];

        // check validity of inputs with respect to compile-time arguments
        static constexpr int kMaxElemsPerTile = kMaxElemsPerThread * kMaxThreadsPerBlock;
        TORCH_CHECK(seqlen <= kMaxElemsPerTile && "Input sequence is longer than maximum tile size.");
        TORCH_CHECK(d_outputs.numel() <= (1UL << 8*sizeof(int)-1)-1 && "For int seqBaseIdx, numel() needs to be smaller than 2147483647.");
        TORCH_CHECK(kMaxElemsPerTile <= (1UL << 8*sizeof(ushort)) && "For ushort indexing, the kMaxElemsPerTile needs to be smaller than 65536.");

        // select kernel based on compile-time arguments 
        auto kernel = linrec_tile_bwd_kernel<float, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode>;
        
        // determine run-time arguments
        int blocks = numseq;
        int threads = kMaxThreadsPerBlock; // TODO: std::min(seqlen, kMaxThreadsPerBlock);
        int smem = (memcode > 0) ? seqlen * sizeof(float) : 0;

        // configure kernel run-time properties: 
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
        int maxDynamicSmemSize = cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin, (int)d_outputs.get_device());
        TORCH_CHECK(smem <= maxDynamicSmemSize && "Not enough shared memory to accomodate coalesed loading of a tile (with memcode>0).");
        
        if (memcode > 0) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 67);   // >65792 bytes (>67% for Volta or newer) required for kMaxElemsPerThread=16
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);    // only works if smem <= maxDynamicSmemSize
            int availDynamicSmemSize = cudaFuncGetAttributes(kernel).maxDynamicSharedSizeBytes;
            TORCH_CHECK(smem <= availDynamicSmemSize && "Not enough shared memory granted for coalesed loading of a tile (with memcode>0).");
        }

        // launch kernel
        kernel<<<blocks, threads, smem, stream>>>(
            d_outputs.data_ptr<float>(), 
            coeffs.data_ptr<float>(), 
            outputs.data_ptr<float>(), 
            d_inputs.data_ptr<float>(), 
            d_coeffs.data_ptr<float>(), 
            seqlen, reverse);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        }, "linrec_tile_bwd_kernel", CONFIG_NAMES); // names for errors

    return std::make_tuple(d_inputs, d_coeffs);
}


std::map<std::string, int> linrec_tile_attrs(const bool fwd, const Option& options) {
    
    // Unpack and determine compile-time arguments
    // TODO: make sure that all keys are in {"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock",  "memcode", "algocode"}
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = get(options, "kMaxElemsPerThread", 16); 
    int kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", 32); 
    int kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", 1024); 

    // Dispatch templated function: instantiate compile-time configuration
    auto config = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};
    
    cudaFuncAttributes attrs;

    dispatch<CONFIG_LIST>(config, [&]<auto config>() {
        static constexpr int kMaxElemsPerThread = config[0];
        static constexpr int kMaxThreadsPerWarp = config[1];
        static constexpr int kMaxThreadsPerBlock = config[2];
        static constexpr int memcode = config[3];
        static constexpr int algocode = config[4];
    
        // select kernel based on compile-time arguments
        if (fwd) {
            auto kernel = linrec_tile_fwd_kernel<float, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode>;
            attrs = cudaFuncGetAttributes(kernel);
        } else {
            auto kernel = linrec_tile_bwd_kernel<float, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode>;
            attrs = cudaFuncGetAttributes(kernel);
        }
    
    }, "linrec_tile_attrs", CONFIG_NAMES); // names for errors

    return cudaFuncAttributesAsMap(attrs);
}
