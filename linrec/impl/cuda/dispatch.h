#pragma once
#include <array>
#include <sstream>
#include <type_traits>

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

    
/*  Dispatch templated function: instantiate compile-time parameters

    auto paramnames = std::array{"scalar_t", "myparam"};
    auto params = std::array{ScalarTypeToNum(inputs.scalar_type()), 0};
    std::cin >> params[1];

    static constexpr auto COMPILEPARAMS = std::array{
        std::array{int(ScalarType::Float), 1},
        std::array{int(ScalarType::Float), 16},
        std::array{int(ScalarType::Int), 16},
        std::array{int(ScalarType::Int), 32},
    }

    dispatch<COMPILEPARAMS>(params, [&]<auto params>() {
        using T = typename NumToCppType<params[0]>;
        static constexpr myparam = params[1];

        mytemplatedfunc<T, myparam><<<blocks, threads>>>( 
                inputs.data_ptr<T>(),
                outputs.data_ptr<T>(),
            );
        }, "mytemplatedfunc", std::array{"scalar_t", "myparam"});

Inspired by: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/Dispatch.h#L126
*/

template <typename T1, typename T2>
std::string printParams(T1 name, T2 value) {
  std::stringstream ss;
  ss << name << value;
  return ss.str();
}

template <typename T1, typename T2, size_t N1, size_t N2>
std::string printParams(std::array<T1, N1> names, std::array<T2, N2> values) {
  std::stringstream ss;
  for (int i = 0; i < values.size(); i++) {
    if (i < names.size()) ss << names[i] << "=";
    if (i < values.size()) ss << values[i];
    if (i + 1 < values.size()) ss << ", ";
  }
  return ss.str();
}

template <std::array COMPILEPARAMS, std::size_t I = 0>
inline void dispatch(auto param, auto &&func, std::string funcname,
                     auto paramname) {
  static constexpr auto PARAM = COMPILEPARAMS[I];
  static_assert(std::is_same_v<std::remove_cvref_t<decltype(PARAM)>, std::remove_cvref_t<decltype(param)>>,
      "COMPILEPARAMS[i] and param must have same type (Note: COMPILEPARAMS might get flattened if it contains only one param).");

  if (PARAM == param) {
    func.template operator()<PARAM>();  // call with compile-time param
    return;
  }

  if constexpr (I + 1 < COMPILEPARAMS.size()) {
    dispatch<COMPILEPARAMS, I + 1>(param, func, funcname, paramname);
    return;
  }

  TORCH_CHECK_NOT_IMPLEMENTED(false, "'", funcname, "' is not compiled for template parameters [",
                printParams(paramname, param), "] (dispatch.h).")
}


template <typename T, size_t... Ns>
constexpr auto product(const std::array<T, Ns>&... arrays) {
    constexpr size_t cols = sizeof...(Ns);
    constexpr size_t rows = (Ns *  ...);
    std::array<std::array<T, cols>, rows> out{};

    // compute cumulative Ns
    std::array<size_t, cols+1> cumNs = {1, arrays.size()...};
    for (int i = 1; i < cols+1; i++){
        cumNs[i] = cumNs[i-1] * cumNs[i];
    }

    // compute array entries for every row
    for (size_t row = 0; row < rows; row++) {
        size_t col = 0;
        out[row] = std::array{arrays[row / cumNs[col++] % arrays.size()]...};
    }
    return out;
}


template <typename T, size_t... Ns>
constexpr auto concat(const std::array<T, Ns> &...arrays) {
    std::array<T, (Ns + ...)> out{};

    int i = 0;
    ([&](auto array){
        for(int j = 0; j < array.size(); j++) {
            out[i++] = array[j];
        }
    }(arrays), ...);

    return out;
}


/*
// Non-static types are only supported with C++23 ranges:
// templated argument needs static storage description,
// but 'static' in constexpr function is only supported in c++23.
template <auto &COMPILEPARAMS>
constexpr void dispatch(auto &param, auto &&func) {

    static constexpr auto PARAM = *std::ranges::begin(COMPILEPARAMS);
    static constexpr auto TAIL = COMPILEPARAMS | std::views::drop(1);

    if (PARAM == param) {
        func.template operator()<PARAM>(); // call with compile-time param
        return;
    }

    if constexpr (std::ranges::size(COMPILEPARAMS) > 1) {
        dispatch<TAIL>(param, func);
        return;
    }

    TORCH_CHECK_NOT_IMPLEMENTED(false, "'", funcname, "' is not compiled for template parameters [",
                printParams(paramname, param), "] (dispatch.h).")}
*/

