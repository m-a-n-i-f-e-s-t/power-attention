#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "state.h"
#include "kernel_traits.h"
#include "utils.h"
#include "sympow.h"
#include "power_utils.h"

// #define QUERY_STATE_FWD_DEBUG 1
#define DEBUG_THREAD_QFWD (tid == 0 && head_id == 1 && batch_id == 0 && chunk_id == 1)
namespace power_attention
{

    using namespace cute;

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 1) query_power_attention_fwd(__grid_constant__ const Query_state_params params)
    {
        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using index_t = typename Kernel_traits::index_t;
        // Shape parameters
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int BlockQ = Kernel_traits::BlockQ;
        constexpr int Headdim = Kernel_traits::Headdim;
        constexpr int PaddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int InnerBlock = Kernel_traits::InnerBlock;
        constexpr int OuterBlock = Kernel_traits::OuterBlock;
        constexpr int NWarps = Kernel_traits::NWarps;
        constexpr bool DoubleBuffer = Kernel_traits::DoubleBuffer;
        // Shared memory.
        extern __shared__ char smem_[];

        // Thread, lane, warp index
        const int tid = threadIdx.x;

        // Block index
        const int qid = blockIdx.x;
        const int head_id = blockIdx.y;
        const int batch_id = blockIdx.z / params.num_chunks;
        const int chunk_id = blockIdx.z % params.num_chunks;
        const int query_chunk_id = chunk_id;

        // dblock index
        int dblock = 0;

        const index_t q_offset = batch_id * params.batch_stride + query_chunk_id * params.chunk_stride + qid * BlockQ * params.seq_stride + head_id * params.head_stride;
        const index_t s_offset = batch_id * params.batch_stride_s + chunk_id * params.chunk_stride_s + head_id * params.head_stride_s + dblock * BlockD * Headdim;
        const index_t log_g_offset = batch_id * params.batch_stride_log_g + chunk_id * params.chunk_stride_log_g + qid * BlockQ * params.seq_stride_log_g + head_id * params.head_stride_log_g;
        const index_t y_attn_offset = batch_id * params.batch_stride_y_attn + chunk_id * params.chunk_stride_y_attn + qid * BlockQ * params.seq_stride_y_attn + head_id * params.head_stride_y_attn;
        const index_t rowmax_offset = batch_id * params.batch_stride_rowmax + query_chunk_id * params.chunk_stride_rowmax + qid * BlockQ * params.seq_stride_rowmax + head_id * params.head_stride_rowmax;
        const index_t phi_offset = batch_id * params.batch_stride_phi + query_chunk_id * params.chunk_stride_phi + qid * BlockQ * params.seq_stride_phi + head_id * params.head_stride_phi + dblock * BlockD;
        const int numBlockD = (PaddedExpandedDim + BlockD - 1) / BlockD;
        
        // ====================================== Input Shapes ======================================
        // Q: (Batch, Chunk, Seq, Head, Headdim)
        // S: (Batch, Chunk, Head, PaddedExpandedDim, Headdim)
        // C: (PaddedExpandedDim)
        // O: (Batch, Chunk, Seq, Head, Headdim)
        // Norm: (Batch, Chunk, Head, PaddedExpandedDim)
        // Norm_out: (Batch, Chunk, Seq, Head)
        // Phi: (Batch, Chunk, Seq, Head, PaddedExpandedDim)

        // ====================================== Global tensors ======================================
        // Q
        Tensor gQ = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + q_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockQ, Headdim)

        // S
        Tensor gS = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.s_ptr) + s_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            make_stride(Int<Headdim>{}, _1{})); // (BlockD, Headdim)


        // O
        Tensor gO = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + q_offset),
            make_layout(Shape<Int<BlockQ>, Int<Headdim>>{}, make_stride(params.seq_stride, Stride<_1>{}))); // (BlockQ, Headdim)

        // Y
        Tensor gY = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.Y_attn_ptr) + q_offset),
            make_layout(Shape<Int<BlockQ>, Int<Headdim>>{}, make_stride(params.seq_stride, Stride<_1>{}))); // (BlockQ, Headdim)

        // rowmax
        Tensor gRowmax = make_tensor(
            make_gmem_ptr(reinterpret_cast<float *>(params.rowmax_ptr) + rowmax_offset),
            make_layout(Shape<Int<BlockQ>>{}, make_stride(params.seq_stride_rowmax))); // (BlockQ)

        // Phi
        Tensor gPhi = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.phi_ptr) + phi_offset),
            make_layout(Shape<Int<BlockQ>, Int<BlockD>>{}, make_stride(params.seq_stride_phi, 1))); // (BlockQ, BlockD)

        // log_G
        Tensor gLogG = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.log_G_ptr) + log_g_offset),
            make_layout(Shape<Int<BlockQ>>{}, make_stride(params.seq_stride_log_g))); // (BlockQ)

        // ====================================== Shared memory tensors ======================================
        Tensor sQ = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutQ{}); // (BlockQ, Headdim)

        Tensor sS = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sQ.data()) + size(sQ))),
            typename Kernel_traits::SmemLayoutS{}); // (BlockD, Headdim)
        Tensor sSt = make_tensor(
            sS.data(),
            typename Kernel_traits::SmemLayoutSt{}); // (Headdim, BlockD)
        Tensor sStNoSwizzle = make_tensor(
            sS.data(),
            typename Kernel_traits::SmemLayoutStNoSwizzle{}); // (Headdim, BlockD)
            
        Tensor sLogG = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sS.data()) + (DoubleBuffer ? 2 : 1) * size(sS))),
            typename Kernel_traits::SmemLayoutLogG{});

        Tensor sY = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sLogG.data()) + size(sLogG))),
            typename Kernel_traits::SmemLayoutY{});

        Tensor srowmax = make_tensor(
            make_smem_ptr(reinterpret_cast<float *>(&(*sY.data()) + (params.fused ? size(sY) : 0))),
            typename Kernel_traits::SmemLayoutRowmax{}); // (BlockQ)

        Tensor sPQ = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*srowmax.data()) + (params.fused ? size(srowmax) : 0))),
            typename Kernel_traits::SmemLayoutPhiQ{}); // (BlockQ, BlockD)
        

        typename Kernel_traits::GmemCopyTileQ gmem_tiled_copy_Q;
        typename Kernel_traits::GmemCopyTileS gmem_tiled_copy_S;
        typename Kernel_traits::GmemCopyTilePhiQ gmem_tiled_copy_PhiQ;
        typename Kernel_traits::GmemCopyTileLogG gmem_tiled_copy_LogG;
        typename Kernel_traits::GmemCopyTileY gmem_tiled_copy_Y;
        typename Kernel_traits::GmemCopyTileRowmax gmem_tiled_copy_Rowmax;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tid);
        auto gmem_thr_copy_S = gmem_tiled_copy_S.get_thread_slice(tid);
        auto gmem_thr_copy_PhiQ = gmem_tiled_copy_PhiQ.get_thread_slice(tid);
        auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tid);
        auto gmem_thr_copy_Y = gmem_tiled_copy_Y.get_thread_slice(tid);
        auto gmem_thr_copy_Rowmax = gmem_tiled_copy_Rowmax.get_thread_slice(tid);

        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);      // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);      // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tSgS = gmem_thr_copy_S.partition_S(gS);      // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockD)
        Tensor tSsS = gmem_thr_copy_S.partition_D(sS);      // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tPgP = gmem_thr_copy_PhiQ.partition_S(gPhi); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockD)
        Tensor tPsP = gmem_thr_copy_PhiQ.partition_D(sPQ);  // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tLogGgLogG = gmem_thr_copy_LogG.partition_S(gLogG); // ((CPY_ATOM_I), TILE_Q)
        Tensor tLogGsLogG = gmem_thr_copy_LogG.partition_D(sLogG); // ((CPY_ATOM_I), TILE_Q)
        Tensor tYgY = gmem_thr_copy_Y.partition_S(gY); // ((CPY_ATOM_I), TILE_Q)
        Tensor tYsY = gmem_thr_copy_Y.partition_D(sY); // ((CPY_ATOM_I), TILE_Q)
        Tensor tRMgRM = gmem_thr_copy_Rowmax.partition_S(gRowmax); // ((CPY_ATOM_I), TILE_Q)
        Tensor tRMsRM = gmem_thr_copy_Rowmax.partition_D(srowmax); // ((CPY_ATOM_I), TILE_Q)

        // MMA stuff
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);
        Tensor tSrPQ = thr_mma.partition_fragment_A(sPQ);

        // These handles smem->register copy for mma
        // Copy Atom Retiling
        auto smem_tiled_copy_PhiQ = make_tiled_copy_A(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_PhiQ = smem_tiled_copy_PhiQ.get_thread_slice(tid);
        Tensor tSsPQ = smem_thr_copy_PhiQ.partition_S(sPQ);

        auto smem_tiled_copy_St = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_St = smem_tiled_copy_St.get_thread_slice(tid);
        Tensor tSsSt = smem_thr_copy_St.partition_S(sSt);

        Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<BlockQ>, Int<Headdim>>{});

        // Predicates
        Tensor cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));
        Tensor tScS = gmem_thr_copy_S.partition_S(cS);

        // Loaders and bumpers
        auto load_Q = [&]() {
            Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
            Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, BlockQ);
        };
        auto load_S = [&]() {
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_S, tSgS, tSsS, tScS, BlockD);
            tSgS.data() = tSgS.data() + index_t(BlockD * Headdim);
        };
        auto bump_sS = [&](const int dblock) {
            if constexpr (DoubleBuffer) {
                tSsS.data() = tSsS.data() + (dblock % 2 == 0 ? size(sS) : -size(sS));
            }
        };
        auto save_Phi = [&]() {
            Tensor cPQ = make_identity_tensor(make_shape(size<0>(sPQ), size<1>(sPQ)));
            Tensor tPQcPQ = gmem_thr_copy_PhiQ.partition_S(cPQ);
            power_attention::copy</*Is_even_MN=*/true>(gmem_tiled_copy_PhiQ, tPsP, tPgP, tPQcPQ, BlockQ);
            tPgP.data() = tPgP.data() + index_t(BlockD);
        };
        auto load_log_G = [&]() {
            if (params.gating) {
                Tensor clogG = make_identity_tensor(sLogG.shape());
                Tensor tlogGclogG = gmem_thr_copy_LogG.partition_S(clogG);
                power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_LogG, tLogGgLogG, tLogGsLogG, tlogGclogG, BlockQ);
            }
        };
        auto load_Y = [&]() {
            Tensor cY = make_identity_tensor(sY.shape());
            Tensor tYcY = gmem_thr_copy_Y.partition_S(cY);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Y, tYgY, tYsY, tYcY, std::min(BlockQ, params.chunk_seq_len - qid * BlockQ));
        };
        auto load_rowmax = [&]() {
            Tensor cRM = make_identity_tensor(srowmax.shape());
            Tensor tRMcRM = gmem_thr_copy_Rowmax.partition_S(cRM);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Rowmax, tRMgRM, tRMsRM, tRMcRM, BlockQ);
        };


        // sympow state
        power_attention::SymowStateExpanderBlockP2<Kernel_traits> sympow;

        // Prologue
        clear(acc_o);
        // load Q
        load_log_G();
        load_Q();
        cute::cp_async_fence();

        // start load S, Y, y
        if (params.fused) {
            load_Y();
            load_rowmax();
        }
        load_S();
        bump_sS(dblock);
        cute::cp_async_fence();

        // Main loop
        // skip the first chunk if it's the first chunk and the initial state is zero
        if (params.non_zero_initial_state || chunk_id > 0) {
            for (int inner_bid = 0, outer_bid = 0; dblock < numBlockD; ++dblock)
            {
                // start loading next N
                if (DoubleBuffer && dblock < numBlockD - 1) {
                    load_S();
                    bump_sS(dblock + 1);
                }
                cute::cp_async_fence();
                
                // make sure Q, N are in smem
                power_attention::cp_async_wait<1>();
                __syncthreads();

    #ifdef QUERY_STATE_FWD_DEBUG
                if (DEBUG_THREAD_QFWD) {
                    printf("dblock: %d, inner_bid: %d, outer_bid: %d, sQ:\n", dblock, inner_bid, outer_bid);
                    print_tensor(sQ);
                    printf("\n");
                }
    #endif

                // expand states
                Tensor tSrPQ = thr_mma.partition_fragment_A(sPQ);
                if (params.expand) {
                    tSrPQ = sympow.expandState(sQ, sPQ, sLogG, tiled_mma, inner_bid, outer_bid, params);
                }

    #ifdef QUERY_STATE_FWD_DEBUG
                Tensor tSsPQ_ = thr_mma.partition_A(sPQ);
                cute::copy(tSrPQ, tSsPQ_);
                __syncthreads();
                if (DEBUG_THREAD_QFWD) {
                    printf("dblock: %d, inner_bid: %d, outer_bid: %d, sPQ:\n", dblock, inner_bid, outer_bid);
                    print_tensor(sPQ);
                    printf("\n");
                }
                __syncthreads();
    #endif
                
                if (params.return_phi) {
                    Tensor tSsPQ = thr_mma.partition_A(sPQ);
                    cute::copy(tSrPQ, tSsPQ);
                    __syncthreads();
                }

    #ifdef QUERY_STATE_FWD_DEBUG
                if (DEBUG_THREAD_QFWD) {
                    printf("dblock: %d, inner_bid: %d, outer_bid: %d, sS:\n", dblock, inner_bid, outer_bid);
                    Tensor sS_now = make_tensor(sS.data() + (DoubleBuffer ? (dblock % 2 == 0 ? 0 : size(sS)) : 0), sS.layout());
                    print_tensor(sS_now);
                    printf("\n");
                }
    #endif

                // do mma between phi(Q) and S
                Tensor tSrSt = thr_mma.partition_fragment_B(sSt);
                if (params.use_multiplier) {
                    power_attention::gemm_rs</*rescale_B=*/true>(acc_o, tSrPQ, tSrSt, tSsSt, tiled_mma, smem_tiled_copy_St, smem_thr_copy_St, params.multiplier);
                } else {
                    power_attention::gemm_rs</*rescale_B=*/false>(acc_o, tSrPQ, tSrSt, tSsSt, tiled_mma, smem_tiled_copy_St, smem_thr_copy_St, 1.0f);
                }
                if (DoubleBuffer) {
                    tSsSt.data() = tSsSt.data() + int(dblock % 2 == 0 ? size(sSt) : -size(sSt));
                }

    #ifdef QUERY_STATE_FWD_DEBUG
                if (DEBUG_THREAD_QFWD) {
                    printf("dblock: %d, inner_bid: %d, outer_bid: %d, acc_o:\n", dblock, inner_bid, outer_bid);
                    print_tensor(acc_o);
                    printf("\n");
                }
    #endif

                if (params.return_phi)
                {
                    save_Phi();
                    cute::cp_async_wait<0>(); // wait all
                    __syncthreads();
                }

                if (!DoubleBuffer && dblock < numBlockD - 1) {
                    load_S();
                    cute::cp_async_fence();
                }

                // bump inner_bid, outer_bid
                binfo_bump<OuterBlock, InnerBlock>(inner_bid, outer_bid);
                __syncthreads();
            }
        } else {
            power_attention::cp_async_wait<0>();
            __syncthreads();
        }

        // Epilogue
        // reduce acc_y within warps
        Tensor sO = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutO{}); // (BlockQ, Headdim)


        // wait for Y and y
        if (params.fused) {
            power_attention::cp_async_wait<0>();
            __syncthreads();

#ifdef QUERY_STATE_FWD_DEBUG
            if (DEBUG_THREAD_QFWD) {
                printf("sY:\n");
                print_tensor(sY);
                printf("\n");
                printf("sy:\n");
                print_tensor(sy);
                printf("\n");
                printf("srowmax:\n");
                print_tensor(srowmax);
                printf("\n");
            }
            __syncthreads();
#endif
            
            using R_ROWMAX = decltype(power::read_row_summary<size<1>(acc_o) * 2, Kernel_traits::TiledMma>(srowmax));
            R_ROWMAX r_rowmax;
            // read rowmax into register
            r_rowmax = power::read_row_summary<size<1>(acc_o) * 2, Kernel_traits::TiledMma>(srowmax);

            auto smem_tiled_copy_Y = make_tiled_copy_C(
                typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_Y = smem_tiled_copy_Y.get_thread_slice(tid);
            Tensor taccYsY = smem_thr_copy_Y.partition_S(sY);   
            Tensor rY = make_tensor<Element>(acc_o.layout());
            Tensor taccYrY = smem_thr_copy_Y.retile_D(rY);
            // read Y into register
            cute::copy(smem_tiled_copy_Y, taccYsY, taccYrY);

#ifdef QUERY_STATE_FWD_DEBUG
            if (DEBUG_THREAD_QFWD) {
                printf("rY:\n");
                print_tensor(rY);
                printf("\n");
                printf("r_rowmax:\n");
                print_tensor(r_rowmax);
                printf("\n");
                printf("acc_o: \n");
                print_tensor(acc_o);
                printf("\n");
                printf("acc_y: \n");
                print_tensor(acc_y);
                printf("\n");
                printf("params.multiplier: %f\n", params.multiplier);
                printf("params.multiplier_squared: %f\n", params.multiplier_squared);
                printf("params.use_multiplier: %d\n", params.use_multiplier);
                printf("params.non_zero_initial_state: %d\n", params.non_zero_initial_state);
            }
#endif

            Tensor acc_o_rowcol = make_tensor(acc_o.data(), power::convert_layout_acc_rowcol(acc_o.layout()));
            Tensor rY_rowcol = make_tensor(rY.data(), power::convert_layout_acc_rowcol(rY.layout()));
            static_assert(decltype(size<0>(acc_o_rowcol))::value == decltype(size<0>(r_rowmax))::value, "rowmax and acc_o must have the same number of rows");
            static_assert(decltype(size<0>(acc_o_rowcol))::value == decltype(size<0>(rY_rowcol))::value, "acc_o_rowcol and Y_rowcol must have the same number of rows");
            static_assert(decltype(size<1>(acc_o_rowcol))::value == decltype(size<1>(rY_rowcol))::value, "acc_o_rowcol and Y_rowcol must have the same number of columns");

            // apply rowmax to output
            if (params.non_zero_initial_state || chunk_id > 0) {
                CUTE_UNROLL
                for (int m = 0; m < size<0>(acc_o_rowcol); m++) {
                    float scale_attn = expf(-r_rowmax(m));
                    if (params.multiplier_squared <= scale_attn) {
                        float true_scale = params.multiplier_squared;
                        float attn_factor = true_scale / scale_attn;
                        CUTE_UNROLL
                        for (int n = 0; n < size<1>(acc_o_rowcol); n++) {
                            acc_o_rowcol(m, n) += rY_rowcol(m, n) * attn_factor; // add adjusted Y to acc_o
                        }
                    } else {
                        float true_scale = scale_attn;
                        float qs_factor = true_scale / params.multiplier_squared;
                        CUTE_UNROLL
                        for (int n = 0; n < size<1>(acc_o_rowcol); n++) {
                            acc_o_rowcol(m, n) = rY_rowcol(m, n) + acc_o_rowcol(m, n) * qs_factor; // add adjusted Y to acc_o
                        }
                    }
                }
            } else { // in this branch, the initial state is zero so we don't need to add anything
                CUTE_UNROLL
                for (int m = 0; m < size<0>(acc_o_rowcol); m++) {
                    CUTE_UNROLL
                    for (int n = 0; n < size<1>(acc_o_rowcol); n++) {
                        acc_o_rowcol(m, n) = rY_rowcol(m, n); // assign Y to acc_o
                    }
                }
            }

        } 

        __syncthreads();

        // copy acc_o back to smem first
        Tensor rO = power_attention::convert_type<Element>(acc_o);
        Tensor cO = make_identity_tensor(sO.shape());
        auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
        Tensor taccOcO = smem_thr_copy_O.partition_S(cO);
        Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

        power_attention::copy</*Is_even_MN=*/false>(smem_tiled_copy_O, taccOrO, taccOsO, taccOcO, BlockQ);

        // copy acc_o back to global memory
        auto gmem_tiled_copy_O = make_tiled_copy_C(
            typename Kernel_traits::GmemCopyTileO{},
            tiled_mma);
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
        Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOcO = gmem_thr_copy_O.partition_S(cO);

        __syncthreads();

#ifdef QUERY_STATE_FWD_DEBUG
        if (DEBUG_THREAD_QFWD) {
            printf("sO:\n");
            print_tensor(sO);
            printf("\n");
        }
#endif

        power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_O, tOsO, tOgO, tOcO, std::min(BlockQ, params.chunk_seq_len - qid * BlockQ));
    }

} // namespace power_attention
