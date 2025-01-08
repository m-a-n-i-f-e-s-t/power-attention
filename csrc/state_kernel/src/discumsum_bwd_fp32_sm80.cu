// Copyright (c) 2024. Sean Zhang.
// Splitting the different degrees to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "discumsum_launch_template.h"

template<>
void run_discumsum_bwd<float>(Discumsum_bwd_params &params, cudaStream_t stream) {
    run_discumsum_bwd_<float>(params, stream);
}
