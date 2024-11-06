#include <string>
#include <vector>
#include <algorithm>

#include <torch/torch.h>
#include <linrec.h>


int parse(std::vector<std::string> args, std::string flag, int num) {
    auto it = std::find(args.begin(), args.end(), flag);
    if (it != args.end())
        num = std::stoi(*(it+1));
    return num;
}

int main(int argc, char *argv[]) {

    using namespace torch;

    // Minimal Parser
    std::vector<std::string> args(argv, argv + argc);
    int numseq = parse(args, "--numseq", 142*100);
    int seqlen = parse(args, "--seqlen", 65536);
    bool reverse = parse(args, "--reverse", false);
    int device_id = parse(args, "--device", 0);
    auto config = std::map<std::string, int>{
        {"kMaxElemsPerThread", parse(args, "--kMaxElemsPerThread", 4)},
        {"kMaxThreadsPerBlock", parse(args, "--kMaxThreadsPerBlock", 1024)},
        {"memcode", parse(args, "--memcode", 0)},
        {"algocode", parse(args, "--algocode", 3)},
    };


    // Create Tensors: https://pytorch.org/cppdocs/notes/tensor_creation.html
    auto shape = IntArrayRef({numseq, seqlen});
    auto dtype = ScalarType::Float;
    auto device = (device_id == -1) ? Device("cpu") : Device(DeviceType::CUDA, device_id);
    auto options = TensorOptions().dtype(dtype).device(device).requires_grad(false);
    
    Tensor inputs = torch::randn(shape, options);
    Tensor coeffs = torch::rand(shape, options);
    Tensor d_outputs = torch::randn(shape, options);
    std::cout << "inputs: " << inputs.toString() << inputs.sizes() << std::endl << std::flush;
    std::cout << "coeffs: " << coeffs.toString() << coeffs.sizes() << std::endl << std::flush;
    std::cout << "d_outputs: " << d_outputs.toString() << d_outputs.sizes() << std::endl << std::flush;

    // Reference implementations
    Tensor outputs_ref = linrec_ref_fwd(inputs, coeffs, reverse);
    auto[d_inputs_ref, d_coeffs_ref] = linrec_ref_bwd(d_outputs, coeffs, outputs_ref);    // structured binding

    // Forward implementation 
    Tensor outputs = linrec_pipe_fwd(inputs, coeffs, reverse, config);
    std::cout << "linrec_pipe_fwd:";
    std::cout << " reg=" << linrec_pipe_attrs(true, config)["numRegs"];
    std::cout << ", lmem=" << linrec_pipe_attrs(true, config)["localSizeBytes"] << std::endl;
    std::cout << "outputs: " << outputs.toString() << outputs.sizes();
    std::cout << ", error=" << (outputs_ref -  outputs).abs().max().item() << std::endl << std::flush;
    
    // Backward implementations
    auto[d_inputs, d_coeffs] = linrec_pipe_bwd(d_outputs, coeffs, outputs, reverse, config);
    std::cout << "linrec_pipe_bwd:";
    std::cout << " reg=" << linrec_pipe_attrs(false, config)["numRegs"];
    std::cout << ", lmem=" << linrec_pipe_attrs(false, config)["localSizeBytes"] << std::endl;
    std::cout << "d_inputs: " << d_inputs.toString() << d_inputs.sizes();
    std::cout << ", error=" << (d_inputs_ref -  d_inputs).abs().max().item() << std::endl << std::flush;
    std::cout << "d_coeffs: " << d_coeffs.toString() << d_coeffs.sizes();
    std::cout << ", error=" << (d_coeffs_ref -  d_coeffs).abs().max().item() << std::endl << std::flush;

}