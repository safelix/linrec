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
    int numseq = parse(args, "--numseq", 14200);
    int seqlen = parse(args, "--seqlen", 4*1024);
    bool reverse = parse(args, "--reverse", false);
    int device_id = parse(args, "--device", 0);
    auto linrec_options = std::map<std::string, int>{
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
    
    // Forward
    Tensor inputs = torch::randn(shape, options);
    Tensor coeffs = torch::exp(-torch::rand(shape, options));
    std::cout << "inputs: " << inputs.toString() << inputs.sizes() << std::endl << std::flush;
    std::cout << "coeffs: " << coeffs.toString() << coeffs.sizes() << std::endl << std::flush;

    Tensor solution, outputs; 
    
    solution = linrec_ref_fwd(inputs, coeffs, reverse);
    if (linrec_options.at("algocode") == 0)
        solution = inputs;
    outputs = linrec_tile_fwd(inputs, coeffs, reverse, linrec_options);
    std::cout << "outputs: " << outputs.toString() << outputs.sizes();
    std::cout << ", error=" << (solution -  outputs).abs().max().item() << std::endl << std::flush;
    
    // Backward
    Tensor d_outputs = torch::randn(shape, options);
    std::cout << "d_outputs: " << d_outputs.toString() << d_outputs.sizes() << std::endl << std::flush;

    /*
    auto[d_inputs, d_coeffs] = linrec_ref_bwd(d_outputs, coeffs, outputs);    // structured binding
    std::cout << "d_inputs: " << d_inputs.toString() << d_inputs.sizes() << std::endl;
    std::cout << "d_coeffs: " << d_coeffs.toString() << d_coeffs.sizes() << std::endl;
    */

    //TupleTensor outputs = linrec_tile_bwd(inputs, coeffs);
}