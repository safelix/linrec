#pragma once
#include <torch/torch.h>

#include <map>
#include <tuple>
#include <dispatch.h>

static constexpr auto COMPILEPARAMS = concat(
    //product(std::array{16}, std::array{32}, std::array{1024}, std::array{-1, 0, 1, 2}, std::array{0, 3}) // debug
    product(std::array{16}, std::array{32}, std::array{32, 1024}, std::array{-1}, std::array{0, 1, 2, 3}), // demo algo
    product(std::array{4, 8, 16}, std::array{32}, std::array{32, 64, 128, 256, 512, 1024}, std::array{0, 1, 2}, std::array{0, 3})  // tuning
);


// Reference
torch::Tensor linrec_ref_fwd(const torch::Tensor &inputs,
                             const torch::Tensor &coeffs,
                             const bool reverse = false);

std::tuple<torch::Tensor, torch::Tensor> linrec_ref_bwd(
                            const torch::Tensor &d_outputs, 
                            const torch::Tensor &coeffs,
                            const torch::Tensor &outputs, 
                            const bool reverse = false);


// Tile
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

std::map<std::string, int> linrec_tile_attrs(const bool fwd,  const std::map<std::string, int> &kwargs = {});


// Pipe
torch::Tensor linrec_pipe_fwd(const torch::Tensor &inputs,
                              const torch::Tensor &coeffs,
                              const bool reverse = false,
                              const std::map<std::string, int> &kwargs = {});

std::tuple<torch::Tensor, torch::Tensor> linrec_pipe_bwd(
                            const torch::Tensor &d_outputs, 
                            const torch::Tensor &coeffs,
                            const torch::Tensor &outputs,
                            const bool reverse = false,
                            const std::map<std::string, int> &kwargs = {});

std::map<std::string, int> linrec_pipe_attrs(const bool fwd,  const std::map<std::string, int> &kwargs = {});
