#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h> // TODO: why can't I just use torch::cuda::CUDAGuard
#include <c10/core/ScalarType.h>

#include <linrec.h>
#include <dispatch.h>
#include <linrec_tile.cuh>
#include <memio.cuh>
#include <utils.cuh>

using torch::Tensor;

using Option = std::map<std::string, int>;
int get(const Option& option, const std::string key, int default_value) {   
    return option.contains(key) ? option.at(key) : default_value;
}

static constexpr auto COMPILEPARAMS = std::array{
    // debug algo: kMaxElemsPerThread=32, memcode=-1
    std::array{16, 32, 32, -1, 0},
    std::array{16, 32, 32, -1, 1},
    std::array{16, 32, 32, -1, 2},
    std::array{16, 32, 32, -1, 3},

    // debug algo: kMaxElemsPerThread=1024, memcode=-1
    std::array{16, 32, 1024, -1, 0},
    std::array{16, 32, 1024, -1, 1},
    std::array{16, 32, 1024, -1, 2},
    std::array{16, 32, 1024, -1, 3},

    std::array{16, 32, 1024, 0, 0},
    std::array{16, 32, 1024, 0, 1},
    std::array{16, 32, 1024, 0, 2},
    std::array{16, 32, 1024, 0, 3},
    
    std::array{16, 32, 1024, 1, 0},
    std::array{16, 32, 1024, 1, 1},
    std::array{16, 32, 1024, 1, 2},
    std::array{16, 32, 1024, 1, 3},

    // bench I/O: algocode=0
    std::array{16, 32, 1024, -1, 3},
    std::array{16, 32, 1024, 0, 3},
    std::array{16, 32, 1024, 1, 3},

    // bench layout: memcode=0, algocode=3
    std::array{16, 32, 1024, 0, 3},
    std::array{8, 32, 1024, 0, 3},
    std::array{4, 32, 1024, 0, 3},

    std::array{16, 32, 512, 0, 3},
    std::array{8, 32, 512, 0, 3},
    std::array{4, 32, 512, 0, 3},

    std::array{16, 32, 256, 0, 3},
    std::array{8, 32, 256, 0, 3},
    std::array{4, 32, 256, 0, 3},

    std::array{16, 32, 128, 0, 3},
    std::array{8, 32, 128, 0, 3},
    std::array{4, 32, 128, 0, 3},

    // bench layout: memcode=1, algocode=3
    std::array{16, 32, 1024, 1, 3},
    std::array{8, 32, 1024, 1, 3},
    std::array{4, 32, 1024, 1, 3},

    std::array{16, 32, 512, 1, 3},
    std::array{8, 32, 512, 1, 3},
    std::array{4, 32, 512, 1, 3},

    std::array{16, 32, 256, 1, 3},
    std::array{8, 32, 256, 1, 3},
    std::array{4, 32, 256, 1, 3},

    std::array{16, 32, 128, 1, 3},
    std::array{8, 32, 128, 1, 3},
    std::array{4, 32, 128, 1, 3},
};


Tensor linrec_tile_fwd(const Tensor &inputs, const Tensor &coeffs, const bool reverse, const Option& options) {
    TORCH_CHECK(inputs.sizes() == coeffs.sizes());               // same dimensions
    TORCH_CHECK(inputs.strides() == coeffs.strides());           // same strides
    TORCH_CHECK(inputs.device() == coeffs.device());             // same device
    TORCH_CHECK(inputs.is_cuda() && coeffs.is_cuda());           // both cuda
    TORCH_CHECK(inputs.scalar_type() == coeffs.scalar_type());   // same dtype
    TORCH_CHECK(inputs.scalar_type() == torch::ScalarType::Float); // TODO: dispatch

    // Select correct CUDA device and default stream: otherwise cuda:0 is used
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    const at::cuda::CUDAGuard guard(inputs.get_device());
    //auto stream = torch::cuda::getCurrentCUDAStream().stream(); // should be used by default

    // Prepare Outputs
    Tensor outputs = torch::empty_like(inputs);

    // Infer number of sequences and sequence length
    //TORCH_CHECK(inputs.is_contiguous());      // memory is ordered as dimensions
    TORCH_CHECK(inputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    int seqlen = inputs.size(-1);               // the sequence length
    int numseq = inputs.numel() / seqlen;       // the number of sequences over batches, channels, etc

    // Unpack and determine compile-time arguments
    // TODO: make sure that all keys are in {"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"}
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = get(options, "kMaxElemsPerThread", 16); 
    int kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", 32); 
    int kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", 1024); 

    // Dispatch templated function: instantiate compile-time parameters
    auto paramnames = std::array{"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"};
    auto params = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};

    dispatch<COMPILEPARAMS>(params, [&]<auto params>() {
        static constexpr int kMaxElemsPerThread = params[0];
        static constexpr int kMaxThreadsPerWarp = params[1];
        static constexpr int kMaxThreadsPerBlock = params[2];
        static constexpr int memcode = params[3];
        static constexpr int algocode = params[4];

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
        kernel<<<blocks, threads, smem>>>(
            inputs.data_ptr<float>(),
            coeffs.data_ptr<float>(), 
            outputs.data_ptr<float>(), 
            seqlen, reverse);
    }, "linrec_tile_fwd_kernel", paramnames); // names for errors

    return outputs;
}



std::tuple<Tensor, Tensor> linrec_tile_bwd(const Tensor &d_outputs, const Tensor &coeffs, const Tensor &outputs, bool reverse, const Option& options) {
    TORCH_CHECK(d_outputs.sizes() == coeffs.sizes() && coeffs.sizes() == outputs.sizes());                          // same dimensions
    TORCH_CHECK(d_outputs.strides() == coeffs.strides() && coeffs.strides() == outputs.strides());                  // same strides
    TORCH_CHECK(d_outputs.device() == coeffs.device() && coeffs.device() == outputs.device());                      // same device
    TORCH_CHECK(d_outputs.is_cuda() && coeffs.is_cuda() && outputs.is_cuda());                                      // all cuda
    TORCH_CHECK(d_outputs.scalar_type() == coeffs.scalar_type() && coeffs.scalar_type() == outputs.scalar_type());  // same dtype
    TORCH_CHECK(d_outputs.scalar_type() == torch::ScalarType::Float); // TODO: dispatch

    // Select correct CUDA device and default stream: otherwise cuda:0 is used
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    const at::cuda::CUDAGuard guard(d_outputs.get_device());
    //auto stream = torch::cuda::getCurrentCUDAStream().stream();

    // Prepare Outputs
    Tensor d_inputs = torch::empty_like(d_outputs);
    Tensor d_coeffs = torch::empty_like(coeffs);

    // Infer number of sequences and sequence length
    TORCH_CHECK(d_outputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    const int seqlen = d_outputs.size(-1);         // the sequence length
    const int numseq = d_outputs.numel() / seqlen; // the number of sequences over batches, channels, etc

    // Unpack and determine compile-time arguments
    // TODO: make sure that all keys are in {"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock",  "memcode", "algocode"}
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = get(options, "kMaxElemsPerThread", 16); 
    int kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", 32); 
    int kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", 1024); 

    // Dispatch templated function: instantiate compile-time parameters
    auto paramnames = std::array{"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"};
    auto params = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};
    
    dispatch<COMPILEPARAMS>(params, [&]<auto params>() {
        static constexpr int kMaxElemsPerThread = params[0];
        static constexpr int kMaxThreadsPerWarp = params[1];
        static constexpr int kMaxThreadsPerBlock = params[2];
        static constexpr int memcode = params[3];
        static constexpr int algocode = params[4];

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
        kernel<<<blocks, threads, smem>>>(
            d_outputs.data_ptr<float>(), 
            coeffs.data_ptr<float>(), 
            outputs.data_ptr<float>(), 
            d_inputs.data_ptr<float>(), 
            d_coeffs.data_ptr<float>(), 
            seqlen, reverse);
        }, "linrec_tile_bwd_kernel", paramnames); // names for errors

    return std::make_tuple(d_inputs, d_coeffs);
}
