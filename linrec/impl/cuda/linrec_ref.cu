#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/ScalarType.h>

#include <dispatch.h>
#include <linrec.h>
#include <linrec_ref.cuh>

using torch::Tensor;

Tensor linrec_fwd_ref(const Tensor &inputs, const Tensor &coeffs, const bool reverse) {
    TORCH_CHECK(inputs.sizes() == coeffs.sizes());               // same dimensions
    TORCH_CHECK(inputs.strides() == coeffs.strides());           // same strides
    TORCH_CHECK(inputs.device() == coeffs.device());             // same device
    TORCH_CHECK(inputs.is_cuda() && coeffs.is_cuda());           // both cuda
    TORCH_CHECK(inputs.scalar_type() == coeffs.scalar_type());   // same dtype

    // Select correct CUDA device and default stream: otherwise cuda:0 is used
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    const at::cuda::CUDAGuard guard((char)inputs.get_device());
    //auto stream = torch::cuda::getCurrentCUDAStream().stream();

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
        linrec_fwd_ref_kernel<kT><<<blocks, threads>>>(
                inputs.data_ptr<kT>(), 
                coeffs.data_ptr<kT>(), 
                outputs.data_ptr<kT>(), 
                seqlen, reverse);
    }, "linrec_fwd_ref_kernel", "scalar_t"); // name for errors

    return outputs;
}


std::tuple<Tensor, Tensor> linrec_bwd_ref(const Tensor &d_outputs, const Tensor &coeffs, const Tensor &outputs, bool reverse) {
    TORCH_CHECK(d_outputs.sizes() == coeffs.sizes() && coeffs.sizes() == outputs.sizes());                          // same dimensions
    TORCH_CHECK(d_outputs.strides() == coeffs.strides() && coeffs.strides() == outputs.strides());                  // same strides
    TORCH_CHECK(d_outputs.device() == coeffs.device() && coeffs.device() == outputs.device());                      // same device
    TORCH_CHECK(d_outputs.is_cuda() && coeffs.is_cuda() && outputs.is_cuda());                                      // all cuda
    TORCH_CHECK(d_outputs.scalar_type() == coeffs.scalar_type() && coeffs.scalar_type() == outputs.scalar_type());  // same dtype

    // Select correct CUDA device and default stream: otherwise cuda:0 is used
    // https://pytorch.org/cppdocs/notes/tensor_cuda_stream.html
    const c10::cuda::CUDAGuard guard((char)d_outputs.get_device()); 
    //auto stream = torch::cuda::getCurrentCUDAStream().stream();

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
        linrec_bwd_ref_kernel<float><<<blocks, threads>>>(
                d_outputs.data_ptr<float>(), 
                coeffs.data_ptr<float>(), 
                outputs.data_ptr<float>(), 
                d_inputs.data_ptr<float>(), 
                d_coeffs.data_ptr<float>(), 
                seqlen, reverse);
    }, "linrec_bwd_ref_kernel", "scalar_t"); // name for errors

    return std::make_tuple(d_inputs, d_coeffs);
}

