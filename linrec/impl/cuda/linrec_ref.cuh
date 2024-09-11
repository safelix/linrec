#pragma once
#include <cuda.h>

/* 
 * Reference Implementations Linear Recurrence 
*/

template <typename kT>
__global__ void linrec_ref_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, const int seqLen) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x; // threads block process channels independently: inputs[seqBaseIdx + i]
    
    // Linear Recursion
    outputs[seqBaseIdx] = inputs[seqBaseIdx];   // set start element
    for(int i = 1; i < seqLen; i++) {           // linear scan
        outputs[seqBaseIdx + i] = outputs[seqBaseIdx + i - 1] * coeffs[seqBaseIdx + i] + inputs[seqBaseIdx + i];
    }
}

template <typename kT>
__global__ void linrec_ref_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x + (seqLen - 1); // threads block process channels independently: inputs[seqBaseIdx + i]
    
    // Linear Backwards Recursion
    d_inputs[seqBaseIdx] = d_outputs[seqBaseIdx];   // set start d_input
    for(int i = 1; i < seqLen; i++) {               // linear scan
        d_inputs[seqBaseIdx - i] = d_inputs[seqBaseIdx - i + 1] * coeffs[seqBaseIdx - i] + d_outputs[seqBaseIdx - i];
    }

    // element-wise shifted multiplication
    for(int i = 0; i < seqLen-1; i++) {             // no scan
        d_coeffs[seqBaseIdx - i] = outputs[seqBaseIdx - i - 1] * d_inputs[seqBaseIdx - i];
    }
    d_coeffs[seqBaseIdx - (seqLen - 1)] = 0;      // set last d_coeff
}


/* 
 * Reference Implementations Linear Recurrence: overload with reverse flag
*/

template <typename kT>
__global__ void linrec_ref_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, const int seqLen, const bool reverse) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x; // threads block process channels independently: inputs[seqBaseIdx + i]
    
    int s = reverse ? -1 : 1;  // if reverse: start at end and subtract index (flip sign)
    seqBaseIdx = seqBaseIdx + (reverse ? (seqLen - 1) : 0); // inputs[seqBaseIdx ± i]

    // Linear Recursion
    outputs[seqBaseIdx] = inputs[seqBaseIdx];   // set start element
    for(int i = 1; i < seqLen; i++) {           // linear scan
        outputs[seqBaseIdx + s*i] = outputs[seqBaseIdx + s*i - s*1] * coeffs[seqBaseIdx + s*i] + inputs[seqBaseIdx + s*i];
    }
}

template <typename kT>
__global__ void linrec_ref_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool reverse) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x + (seqLen - 1); // threads block process channels independently: inputs[seqBaseIdx + i]
    
    int s = reverse ? -1 : 1;  // if reverse: start at front and add index (flip sign)
    seqBaseIdx = seqBaseIdx - (reverse ? (seqLen - 1) : 0); // inputs[seqBaseIdx ± i]

    // Linear Backwards Recursion
    d_inputs[seqBaseIdx] = d_outputs[seqBaseIdx];   // set start d_input
    for(int i = 1; i < seqLen; i++) {               // linear scan
        d_inputs[seqBaseIdx - s*i] = d_inputs[seqBaseIdx - s*i + s*1] * coeffs[seqBaseIdx - s*i] + d_outputs[seqBaseIdx - s*i];
    }

    // element-wise shifted multiplication
    for(int i = 0; i < seqLen-1; i++) {             // no scan
        d_coeffs[seqBaseIdx - s*i] = outputs[seqBaseIdx - s*i - s*1] * d_inputs[seqBaseIdx - s*i];
    }
    d_coeffs[seqBaseIdx - s*(seqLen - 1)] = 0;      // set last d_coeff
}

