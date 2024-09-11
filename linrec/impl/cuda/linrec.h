#pragma once
#include <torch/torch.h>

#include <map>
#include <tuple>


torch::Tensor linrec_ref_fwd(const torch::Tensor &inputs,
                             const torch::Tensor &coeffs,
                             const bool reverse = false);

std::tuple<torch::Tensor, torch::Tensor> linrec_ref_bwd(
                            const torch::Tensor &d_outputs, 
                            const torch::Tensor &coeffs,
                            const torch::Tensor &outputs, 
                            const bool reverse = false);

torch::Tensor linrec_tile_fwd(const torch::Tensor &inputs,
                              const torch::Tensor &coeffs,
                              const bool reverse = false,
                              const std::map<std::string, int> &kwargs = {});

std::tuple<torch::Tensor, torch::Tensor> linrec_tile_bwd(
                            const torch::Tensor &d_outputs, 
                            const torch::Tensor &coeffs,
                            const torch::Tensor &outputs,
                            const bool reverse = false,
                            const std::map<std::string, int> &kwargs = {});
