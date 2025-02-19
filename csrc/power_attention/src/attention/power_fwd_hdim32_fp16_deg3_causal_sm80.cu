// Copyright (c) 2024, Sean Zhang.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "power_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 32, 3, true>(Power_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim32<cutlass::half_t, 3, true>(params, stream);
}
