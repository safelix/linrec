#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <linrec.h>

/* PyTorch uses pybind11 to interface between C++ and Python:
    - https://pybind11.readthedocs.io/en/stable/basics.html#creating-bindings-for-a-simple-function
    - https://pybind11.readthedocs.io/en/stable/advanced/functions.html#accepting-args-and-kwargs
    - https://pybind11.readthedocs.io/en/stable/advanced/pycpp/object.html#casting-back-and-forth

Named arguments and defaults are exposed via the `py::arg("arg")=default` syntax. Keyword arguments
are exposed via the py::kwargs as the last argument type of C++ function descriptions. For this, we
wrap the functions in linrec.h to take a py::kwargs argument, cast it to std::map<std::string, int>
and call the original function. The resulting functions would look like so:
```
torch::Tensor linrec_tile_fwd_kwargs(const torch::Tensor &inputs, const torch::Tensor &coeffs, const bool reverse, const py::kwargs& kwargs) {
        return linrec_tile_fwd(inputs, coeffs, reverse, kwargs.cast<std::map<std::string, int>>());
}
``` */

inline auto cast_kwarg(auto& arg) {
    return arg;
}
inline auto cast_kwarg(const py::kwargs& kwarg) {
    return kwarg.cast<const std::map<std::string, int>>();
}
template <typename Ret, typename... Arg>
inline auto cast_kwarg(Ret (*f)(Arg...)) {
    return [f](std::conditional_t<std::is_same_v<Arg, const std::map<std::string, int>&>, const py::kwargs&, Arg>... arg) {
        return f(cast_kwarg(arg)...);
    };
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("linrec_ref_fwd", torch::wrap_pybind_function(linrec_ref_fwd), 
                "Reference CUDA imlplementation of linear recursion forward pass.",
                py::arg("inputs"), py::arg("coeffs"), py::arg("reverse")=false);

    m.def("linrec_ref_bwd", torch::wrap_pybind_function(linrec_ref_bwd), 
                "Reference CUDA imlplementation of linear recursion backward pass.",
                py::arg("d_outputs"), py::arg("coeffs"), py::arg("outputs"), py::arg("reverse")=false);

    
    m.attr("config_list") = py::cast(CONFIG_LIST);
    m.attr("config_names") = py::cast(CONFIG_NAMES);
    m.def("linrec_tile_fwd", torch::wrap_pybind_function(cast_kwarg(linrec_tile_fwd)), 
                "Parallel CUDA implementation of linear recursion forward pass.",
                py::arg("inputs"), py::arg("coeffs"), py::arg("reverse")=false);

    m.def("linrec_tile_bwd", torch::wrap_pybind_function(cast_kwarg(linrec_tile_bwd)), 
                "Parallel CUDA implementation of linear recursion backward pass.",
                py::arg("d_outputs"), py::arg("coeffs"), py::arg("outputs"), py::arg("reverse")=false);

    m.def("linrec_tile_attrs", torch::wrap_pybind_function(cast_kwarg(linrec_tile_attrs)), 
                "Get CUDA attributes of linrec_tile_fwd_kernel or linrec_tile_bwd_kernel.",
                py::arg("fwd"));
                

    m.def("linrec_pipe_fwd", torch::wrap_pybind_function(cast_kwarg(linrec_pipe_fwd)), 
                "Parallel CUDA implementation of linear recursion forward pass.",
                py::arg("inputs"), py::arg("coeffs"), py::arg("reverse")=false);
 
    m.def("linrec_pipe_bwd", torch::wrap_pybind_function(cast_kwarg(linrec_pipe_bwd)), 
                "Parallel CUDA implementation of linear recursion backward pass.",
                py::arg("d_outputs"), py::arg("coeffs"), py::arg("outputs"), py::arg("reverse")=false);
    
    m.def("linrec_pipe_attrs", torch::wrap_pybind_function(cast_kwarg(linrec_pipe_attrs)), 
                "Get CUDA attributes of linrec_pipe_fwd_kernel or linrec_pipe_bwd_kernel.",
                py::arg("fwd"));
}