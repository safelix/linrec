#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <linrec.h>

// 
// https://pybind11.readthedocs.io/en/stable/basics.html#creating-bindings-for-a-simple-function
// https://pybind11.readthedocs.io/en/stable/advanced/functions.html#accepting-args-and-kwargs
// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html#casting-back-and-forth
torch::Tensor linrec_fwd_tile_kwargs(const torch::Tensor &inputs, const torch::Tensor &coeffs, const bool reverse, const py::kwargs& kwargs) {
        return linrec_fwd_tile(inputs, coeffs, reverse, kwargs.cast<std::map<std::string, int>>());
}
std::tuple<torch::Tensor, torch::Tensor> linrec_bwd_tile_kwargs(const torch::Tensor &d_outputs, const torch::Tensor &coeffs, const torch::Tensor &outputs, const bool reverse, const py::kwargs& kwargs) {
        return linrec_bwd_tile(d_outputs, coeffs, outputs, reverse, kwargs.cast<std::map<std::string, int>>());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linrec_fwd_ref", torch::wrap_pybind_function(linrec_fwd_ref), 
                "Reference CUDA imlplementation of linear recursion forward pass.",
                py::arg("inputs"), py::arg("coeffs"), py::arg("reverse")=false);

    m.def("linrec_bwd_ref", torch::wrap_pybind_function(linrec_bwd_ref), 
                "Reference CUDA imlplementation of linear recursion backward pass.",
                py::arg("d_outputs"), py::arg("coeffs"), py::arg("outputs"), py::arg("reverse")=false);

    
    m.def("linrec_fwd_tile", torch::wrap_pybind_function(linrec_fwd_tile_kwargs), 
                "Parallel CUDA implementation of linear recursion forward pass.",
                py::arg("inputs"), py::arg("coeffs"), py::arg("reverse")=false);
    //m.def("linrec_fwd_tile", torch::wrap_pybind_function(linrec_fwd_tile), 
    //            "Parallel CUDA implementation of linear recursion forward pass.",
    //            py::arg("inputs"), py::arg("coeffs"), py::arg("reverse")=false, py::arg("options")=std::map<std::string, int>());       
 
    m.def("linrec_bwd_tile", torch::wrap_pybind_function(linrec_bwd_tile_kwargs), 
                "Parallel CUDA implementation of linear recursion backward pass.",
                py::arg("d_outputs"), py::arg("coeffs"), py::arg("outputs"), py::arg("reverse")=false);
}