#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>

#include <dispatch.h>
#include <linrec.h>
#include <linrec_ref.cuh>

using torch::Tensor;

Tensor linrec_ref_fwd(const Tensor &inputs, const Tensor &coeffs, const bool reverse) {
    TORCH_CHECK(inputs.sizes() == coeffs.sizes());               // same dimensions
    TORCH_CHECK(inputs.strides() == coeffs.strides());           // same strides
    TORCH_CHECK(inputs.device() == coeffs.device());             // same device
    TORCH_CHECK(inputs.is_cuda() && coeffs.is_cuda());           // both cuda
    TORCH_CHECK(inputs.scalar_type() == coeffs.scalar_type());   // same dtype

    // Select correct CUDA device and it's current stream, otherwise current device and it's default stream (0x0) are used
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html (streams are only managed by torch)
    const c10::cuda::CUDAGuard guard(inputs.device());          // calls cudaSetDevice internally
    auto stream = c10::cuda::getCurrentCUDAStream().stream();   // current stream might not be 0x0

    // Prepare Outputs
    Tensor outputs = torch::empty_like(inputs);

    // Infer number of sequences and sequence length
    TORCH_CHECK(inputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    const int seqlen = inputs.size(-1);         // the sequence length
    const int numseq = inputs.numel() / seqlen; // the number of sequences over batches, channels, etc

    // Prepare Kernel Traits
    const int threads = 1; 
    const int blocks = numseq;

    // Dispatch templated function: instantiate compile-time parameters
    static constexpr std::array SCALARTYPES = {torch::ScalarType::Float};
    torch::ScalarType scalar_t = inputs.scalar_type();

    dispatch<SCALARTYPES>(scalar_t, [&]<auto scalar_t>() {
        using kT = typename c10::impl::ScalarTypeToCPPTypeT<scalar_t>;
        linrec_ref_fwd_kernel<kT><<<blocks, threads, 0, stream>>>(
                inputs.data_ptr<kT>(), 
                coeffs.data_ptr<kT>(), 
                outputs.data_ptr<kT>(), 
                seqlen, reverse);
    }, "linrec_ref_fwd_kernel", "scalar_t"); // name for errors

    return outputs;
}


std::tuple<Tensor, Tensor> linrec_ref_bwd(const Tensor &d_outputs, const Tensor &coeffs, const Tensor &outputs, bool reverse) {
    TORCH_CHECK(d_outputs.sizes() == coeffs.sizes() && coeffs.sizes() == outputs.sizes());                          // same dimensions
    TORCH_CHECK(d_outputs.strides() == coeffs.strides() && coeffs.strides() == outputs.strides());                  // same strides
    TORCH_CHECK(d_outputs.device() == coeffs.device() && coeffs.device() == outputs.device());                      // same device
    TORCH_CHECK(d_outputs.is_cuda() && coeffs.is_cuda() && outputs.is_cuda());                                      // all cuda
    TORCH_CHECK(d_outputs.scalar_type() == coeffs.scalar_type() && coeffs.scalar_type() == outputs.scalar_type());  // same dtype

    // Select correct CUDA device and it's current stream, otherwise current device and it's default stream (0x0) are used
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html (streams are only managed by torch)
    const c10::cuda::CUDAGuard guard(d_outputs.device());          // calls cudaSetDevice internally
    auto stream = c10::cuda::getCurrentCUDAStream().stream();   // current stream might not be 0x0

    // Prepare Outputs
    Tensor d_inputs = torch::empty_like(d_outputs);
    Tensor d_coeffs = torch::empty_like(coeffs);

    // Infer number of sequences and sequence length
    TORCH_CHECK(d_outputs.stride(-1) == 1);        // inner most dimension is last (weaker requirement than contiguous)
    const int seqlen = d_outputs.size(-1);         // the sequence length
    const int numseq = d_outputs.numel() / seqlen; // the number of sequences over batches, channels, etc

    // Prepare Kernel Traits
    const int threads = 1; 
    const int blocks = numseq;

    // Dispatch templated function: instantiate compile-time parameters
    static constexpr std::array SCALARTYPES = {torch::ScalarType::Float};
    torch::ScalarType scalar_t = d_outputs.scalar_type();

    dispatch<SCALARTYPES>(scalar_t, [&]<auto scalar_t>() {
        using kT = typename c10::impl::ScalarTypeToCPPTypeT<scalar_t>;
        linrec_ref_bwd_kernel<float><<<blocks, threads, 0, stream>>>(
                d_outputs.data_ptr<float>(), 
                coeffs.data_ptr<float>(), 
                outputs.data_ptr<float>(), 
                d_inputs.data_ptr<float>(), 
                d_coeffs.data_ptr<float>(), 
                seqlen, reverse);
    }, "linrec_ref_bwd_kernel", "scalar_t"); // name for errors

    return std::make_tuple(d_inputs, d_coeffs);
}

