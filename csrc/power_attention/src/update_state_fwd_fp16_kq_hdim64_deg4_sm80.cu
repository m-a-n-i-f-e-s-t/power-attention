// Copyright (c) 2024. Sean Zhang.
// Splitting the different degrees to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "update_state_launch_template.h"

template<>
void run_compute_update_states<cutlass::half_t, 64, 4>(Update_state_params &params, cudaStream_t stream) {
    run_update_states_fwd_<cutlass::half_t, 64, 4>(params, stream);
}
