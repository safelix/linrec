#pragma once
#include <cuda.h>

/* 
 * Reference Implementations Linear Recurrence 
*/

template <typename kT>
__global__ void linrec_ref_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, const int seqLen) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x;   // threads block process channels independently: inputs[seqBaseIdx + i]
    inputs = &inputs[seqBaseIdx];           // get pointer to sequence
    coeffs = &coeffs[seqBaseIdx];           // get pointer to sequence
    outputs = &outputs[seqBaseIdx];         // get pointer to sequence

    // Linear Recurrence
    outputs[0] = inputs[0];                         // set start element
    for(int i = 1; i < seqLen; i++) {               // linear scan
        outputs[i] = outputs[i-1] * coeffs[i] + inputs[i];
    }
}

template <typename kT>
__global__ void linrec_ref_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x;   // threads block process channels independently: inputs[seqBaseIdx + i]
    d_outputs = &d_outputs[seqBaseIdx];     // get pointer to sequence
    coeffs = &coeffs[seqBaseIdx];           // get pointer to sequence
    outputs = &outputs[seqBaseIdx];         // get pointer to sequence
    d_inputs = &d_inputs[seqBaseIdx];       // get pointer to sequence
    d_coeffs = &d_coeffs[seqBaseIdx];       // get pointer to sequence

    // Linear Backwards Recurrence
    d_inputs[seqLen-1] = d_outputs[seqLen-1];       // set start d_input
    for(int i = seqLen-2; i >= 0; i--) {            // linear scan
        d_inputs[i] = d_inputs[i+1] * coeffs[i+1] + d_outputs[i];
    }

    // element-wise shifted multiplication
    for(int i = 1; i < seqLen; i++) {               // no scan
        d_coeffs[i] = outputs[i-1] * d_inputs[i];
    }
    d_coeffs[0] = 0;                                // set remaining d_coeff
}


/* 
 * Reference Implementations Linear Recurrence: overload with reverse flag
*/

template <typename kT>
__global__ void linrec_ref_fwd_kernel(const kT* inputs, const kT* coeffs, kT* outputs, const int seqLen, const bool rev) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x;   // threads block process channels independently: inputs[seqBaseIdx + i]
    inputs = &inputs[seqBaseIdx];           // get pointer to sequence
    coeffs = &coeffs[seqBaseIdx];           // get pointer to sequence
    outputs = &outputs[seqBaseIdx];         // get pointer to sequence

    // Linear Recurrence
    outputs[!rev ? 0 : (seqLen-1)] = inputs[!rev ? 0 : (seqLen-1)];     // set start element
    
    
    for(int i = !rev ? 1:(seqLen-2); !rev ? (i < seqLen) : (0 <= i); i += !rev ? 1:-1) {          // linear scan
        outputs[i] = outputs[i - (!rev ? 1:-1)] * coeffs[i] + inputs[i];
    }
}

template <typename kT>
__global__ void linrec_ref_bwd_kernel(const kT* d_outputs, const kT* coeffs, const kT* outputs, kT* d_inputs, kT* d_coeffs, const int seqLen, const bool rev) {
    
    // Layout: dim=(numseq,seqLen), strides=(seqLen,1)
    int seqBaseIdx = seqLen * blockIdx.x;   // threads block process channels independently: inputs[seqBaseIdx + i]
    d_outputs = &d_outputs[seqBaseIdx];     // get pointer to sequence
    coeffs = &coeffs[seqBaseIdx];           // get pointer to sequence
    outputs = &outputs[seqBaseIdx];         // get pointer to sequence
    d_inputs = &d_inputs[seqBaseIdx];       // get pointer to sequence
    d_coeffs = &d_coeffs[seqBaseIdx];       // get pointer to sequence

    // Linear Backwards Recurrence
    d_inputs[!rev ? (seqLen-1) : 0] = d_outputs[!rev ? (seqLen-1) : 0];                         // set start element

    for(int i = !rev ? (seqLen-2):1; !rev ? (0<=i):(i<seqLen); i -= !rev ? 1:-1) {          // linear scan
        d_inputs[i] = d_inputs[i + (!rev ? 1:-1)] * coeffs[i + (!rev ? 1:-1)] + d_outputs[i];
    }

    // element-wise shifted multiplication
    for(int i = !rev ? 1:0; i < seqLen - (!rev ? 0:-1); i++) {             // no scan
        d_coeffs[i] = outputs[i - (!rev ? 1:-1)] * d_inputs[i];
    }
    d_coeffs[!rev ? 0:(seqLen-1)] = 0;      // set remaining d_coeff
}

