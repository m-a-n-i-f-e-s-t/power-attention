// Copyright (c) 2024. Sean Zhang.
// Splitting the different degrees to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "query_state_launch_template.h"

template<>
void run_compute_query_states_bwd<cutlass::half_t, 32, 2>(Query_state_bwd_params &params, cudaStream_t stream) {
    run_compute_query_states_bwd_<cutlass::half_t, 32, 2>(params, stream);
}
