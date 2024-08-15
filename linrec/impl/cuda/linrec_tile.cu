#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h> // TODO: why can't I just use torch::cuda::CUDAGuard
#include <c10/core/ScalarType.h>

#include <linrec.h>
#include <dispatch.h>
#include <linrec_tile.cuh>
#include <memio.cuh>

using torch::Tensor;

using Option = std::map<std::string, int>;
int get(const Option& option, const std::string key, int default_value) {   
    return option.contains(key) ? option.at(key) : default_value;
}

static constexpr auto VALIDPARAMS = std::array{
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

    // tune: memcode=0, algocode=3
    std::array{4, 32, 1024, 0, 3},
    std::array{4, 32, 512, 0, 3},
    std::array{4, 32, 256, 0, 3},
    std::array{8, 32, 1024, 0, 3},
    std::array{8, 32, 512, 0, 3},
    std::array{8, 32, 256, 0, 3},
    std::array{16, 32, 1024, 0, 3},
    std::array{16, 32, 512, 0, 3},
    std::array{16, 32, 256, 0, 3},

    // tune: memcode=1, algocode=3
    std::array{4, 32, 1024, 1, 3},
    std::array{4, 32, 512, 1, 3},
    std::array{4, 32, 256, 1, 3},
    std::array{8, 32, 1024, 1, 3},
    std::array{8, 32, 512, 1, 3},
    std::array{8, 32, 256, 1, 3},
    std::array{16, 32, 1024, 1, 3},
    std::array{16, 32, 512, 1, 3},
    std::array{16, 32, 256, 1, 3},
};


Tensor linrec_fwd_tile(const Tensor &inputs, const Tensor &coeffs, const bool reverse, const Option& options) {
    TORCH_CHECK(inputs.sizes() == coeffs.sizes());               // same dimensions
    TORCH_CHECK(inputs.strides() == coeffs.strides());           // same strides
    TORCH_CHECK(inputs.device() == coeffs.device());             // same device
    TORCH_CHECK(inputs.is_cuda() && coeffs.is_cuda());           // both cuda
    TORCH_CHECK(inputs.scalar_type() == coeffs.scalar_type());   // same dtype
    TORCH_CHECK(inputs.scalar_type() == torch::ScalarType::Float); // TODO: dispatch

    // Select correct CUDA device and default stream: otherwise cuda:0 is used
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    const at::cuda::CUDAGuard guard((char)inputs.get_device());
    //auto stream = torch::cuda::getCurrentCUDAStream().stream(); // should be used by default

    // Prepare Outputs
    Tensor outputs = torch::empty_like(inputs);

    // Infer number of sequences and sequence length
    //TORCH_CHECK(inputs.is_contiguous());      // memory is ordered as dimensions
    TORCH_CHECK(inputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    int seqlen = inputs.size(-1);               // the sequence length
    int numseq = inputs.numel() / seqlen;       // the number of sequences over batches, channels, etc

    // Prepare Kernel Traits, Unpack Keyword Argument: cudaOccupancyMaxPotentialBlockSize?
    // TODO: make sure that all keys are in {"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"}
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = 4;
    int kMaxThreadsPerWarp = 32;
    int kMaxThreadsPerBlock = 1024;
    kMaxElemsPerThread = get(options, "kMaxElemsPerThread", kMaxElemsPerThread); 
    kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", kMaxThreadsPerWarp); 
    kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", kMaxThreadsPerBlock); 

    int blocks = numseq;
    int threads = kMaxThreadsPerBlock; // TODO: std::min(seqlen, kMaxThreadsPerBlock);
    blocks = get(options, "blocks", blocks);
    threads = get(options, "threads", threads);
    
    
    TORCH_CHECK(kMaxElemsPerThread <= (1 << 8*sizeof(ushort)) && "For ushort indexing, kMaxElemsPerThread needs to be smaller than 65536.");
    TORCH_CHECK(seqlen <= kMaxElemsPerThread * kMaxThreadsPerBlock && "Input sequence is longer than maximum tile size.");
     
    // Dispatch templated function: instantiate compile-time parameters
    auto paramnames = std::array{"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"};
    auto params = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};
    
    dispatch<VALIDPARAMS>(params, [&]<auto params>() {
        static constexpr int kMaxElemsPerThread = params[0];
        static constexpr int kMaxThreadsPerWarp = params[1];
        static constexpr int kMaxThreadsPerBlock = params[2];
        static constexpr int memcode = params[3];
        static constexpr int algocode = params[4];
        linrec_fwd_tile_kernel<float, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode><<<blocks, threads>>>(
            inputs.data_ptr<float>(),
            coeffs.data_ptr<float>(), 
            outputs.data_ptr<float>(), 
            seqlen, reverse);
        }, "linrec_fwd_tile_kernel", paramnames); // name for errors

    return outputs;
}



std::tuple<Tensor, Tensor> linrec_bwd_tile(const Tensor &d_outputs, const Tensor &coeffs, const Tensor &outputs, bool reverse, const Option& options) {
    TORCH_CHECK(d_outputs.sizes() == coeffs.sizes() && coeffs.sizes() == outputs.sizes());                          // same dimensions
    TORCH_CHECK(d_outputs.strides() == coeffs.strides() && coeffs.strides() == outputs.strides());                  // same strides
    TORCH_CHECK(d_outputs.device() == coeffs.device() && coeffs.device() == outputs.device());                      // same device
    TORCH_CHECK(d_outputs.is_cuda() && coeffs.is_cuda() && outputs.is_cuda());                                      // all cuda
    TORCH_CHECK(d_outputs.scalar_type() == coeffs.scalar_type() && coeffs.scalar_type() == outputs.scalar_type());  // same dtype
    TORCH_CHECK(d_outputs.scalar_type() == torch::ScalarType::Float); // TODO: dispatch

    // Select correct CUDA device and default stream: otherwise cuda:0 is used
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    const at::cuda::CUDAGuard guard((char)d_outputs.get_device());
    //auto stream = torch::cuda::getCurrentCUDAStream().stream();

    // Prepare Outputs
    Tensor d_inputs = torch::empty_like(d_outputs);
    Tensor d_coeffs = torch::empty_like(coeffs);

    // Infer number of sequences and sequence length
    TORCH_CHECK(d_outputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    const int seqlen = d_outputs.size(-1);         // the sequence length
    const int numseq = d_outputs.numel() / seqlen; // the number of sequences over batches, channels, etc

    // Prepare Kernel Traits, Unpack Keyword Argument: cudaOccupancyMaxPotentialBlockSize?
    // TODO: make sure that all keys are in {"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock",  "memcode", "algocode"}
    int memcode = get(options, "memcode", 0);
    int algocode = get(options, "algocode", 3);

    int kMaxElemsPerThread = 4;
    int kMaxThreadsPerWarp = 32;
    int kMaxThreadsPerBlock = 1024;
    kMaxElemsPerThread = get(options, "kMaxElemsPerThread", kMaxElemsPerThread); 
    kMaxThreadsPerWarp = get(options, "kMaxThreadsPerWarp", kMaxThreadsPerWarp); 
    kMaxThreadsPerBlock = get(options, "kMaxThreadsPerBlock", kMaxThreadsPerBlock); 

    int blocks = numseq;
    int threads = kMaxThreadsPerBlock; //TODO: std::min(seqlen, kMaxThreadsPerBlock)?
    blocks = get(options, "blocks", blocks);
    threads = get(options, "threads", threads);

    TORCH_CHECK(kMaxElemsPerThread <= (1 << 8*sizeof(ushort)) && "For ushort indexing, kMaxElemsPerThread needs to be smaller than 65536.");
    TORCH_CHECK(seqlen <= kMaxElemsPerThread * kMaxThreadsPerBlock && "Input sequence is longer than maximum tile size.");
     
    // Dispatch templated function: instantiate compile-time parameters
    auto paramnames = std::array{"kMaxElemsPerThread", "kMaxThreadsPerWarp", "kMaxThreadsPerBlock", "memcode", "algocode"};
    auto params = std::array{kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode};
    
    //https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-8-x
    dispatch<VALIDPARAMS>(params, [&]<auto params>() {
        static constexpr int kMaxElemsPerThread = params[0];
        static constexpr int kMaxThreadsPerWarp = params[1];
        static constexpr int kMaxThreadsPerBlock = params[2];
        static constexpr int memcode = params[3];
        static constexpr int algocode = params[4];
        linrec_bwd_tile_kernel<float, kMaxElemsPerThread, kMaxThreadsPerWarp, kMaxThreadsPerBlock, memcode, algocode><<<blocks, threads>>>(
            d_outputs.data_ptr<float>(), 
            coeffs.data_ptr<float>(), 
            outputs.data_ptr<float>(), 
            d_inputs.data_ptr<float>(), 
            d_coeffs.data_ptr<float>(), 
            seqlen, reverse);
        }, "linrec_bwd_tile_kernel", paramnames); // name for errors

    return std::make_tuple(d_inputs, d_coeffs);
}
