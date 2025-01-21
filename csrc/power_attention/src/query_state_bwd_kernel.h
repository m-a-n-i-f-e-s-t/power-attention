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

// #define QUERY_STATE_BWD_DEBUG 1
// #define QUERY_STATE_BWD_DSDN_DEBUG 1
// #define QUERY_STATE_BWD_DQ_DEBUG 1
// #define QUERY_STATE_BWD_DSDN_DEBUG 1

#define DEBUGGER_THREAD_DSDN (threadIdx.x == 0 && did == 0 && head_id == 0 && blockIdx.z == 0)
#define DEBUGGER_THREAD_DQ (threadIdx.x == 0 && qid == 0 && head_id == 0 && blockIdx.z == 0)
namespace power_attention
{

    using namespace cute;
    using index_t = int64_t;

#define ASSERT_SIZE_EQ(S, D)                \
    CUTE_STATIC_ASSERT(size(S) == size(D)); \
    CUTE_STATIC_ASSERT(rank(S) == rank(D))

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 1) query_state_bwd_dSdN(__grid_constant__ const Query_state_bwd_params params)
    {

        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using index_t = typename Kernel_traits::index_t;
        using multiindex_t = typename Kernel_traits::multiindex_t;
        using C_type = typename Kernel_traits::C_type;
        using Multiply_vector_t = float;
        // Shape parameters
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int BlockQ = Kernel_traits::BlockQ;
        constexpr int Headdim = Kernel_traits::Headdim;
        constexpr int paddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int InnerBlock = Kernel_traits::InnerBlock;
        constexpr int OuterBlock = Kernel_traits::OuterBlock;
        constexpr bool DoubleBuffer = Kernel_traits::DoubleBuffer;
        // Shared memory.
        __align__(128) extern __shared__ char smem_[];

        // Thread, lane, warp index
        const int tid = threadIdx.x;
        const int did = blockIdx.x;
        const auto info = power_attention::binfo(did, OuterBlock, InnerBlock);
        const auto inner_bid = std::get<0>(info);
        const auto outer_bid = std::get<1>(info);
        const auto is_on_diagonal = std::get<2>(info);
        const int head_id = blockIdx.y;
        const int batch_id = blockIdx.z / params.num_chunks;
        const int chunk_id = blockIdx.z % params.num_chunks;

        int q_block = (params.chunk_seq_len + BlockQ - 1) / BlockQ - 1;

        auto phi_offset_ = [&](int batch_id, int chunk_id, int head_id, int did)
        {
            index_t head_stride = paddedExpandedDim;
            index_t seq_stride = params.num_heads * head_stride;
            index_t chunk_stride = params.chunk_seq_len * seq_stride;
            index_t batch_stride = params.num_chunks * chunk_stride;
            return batch_id * batch_stride + chunk_id * chunk_stride + q_block * BlockQ * seq_stride + head_id * head_stride + did * BlockD;
        };

        // strides
        const index_t q_offset = batch_id * params.batch_stride + chunk_id * params.chunk_stride + head_id * params.head_stride + q_block * BlockQ * params.seq_stride;
        const index_t rowmax_offset = batch_id * params.batch_stride_rowmax + chunk_id * params.chunk_stride_rowmax + q_block * BlockQ * params.seq_stride_rowmax + head_id * params.head_stride_rowmax;
        const index_t dS_offset = batch_id * params.batch_stride_s + chunk_id * params.chunk_stride_s + head_id * params.head_stride_s + did * BlockD * Headdim;
        const index_t phi_offset = phi_offset_(batch_id, chunk_id, head_id, did);
        const index_t log_G_offset = batch_id * params.batch_stride_log_g + chunk_id * params.chunk_stride_log_g + q_block * BlockQ * params.seq_stride_log_g + head_id * params.head_stride_log_g;

        // ====================== Global tensors =============================
        // Represent global tensor for Q
        Tensor gQ = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + q_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockQ, Headdim)

        // Represent global tensor for dY
        Tensor gdY = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dY_ptr) + q_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockQ, Headdim)

        // Represent global tensor for rowmax
        Tensor growmax = make_tensor(
            make_gmem_ptr(reinterpret_cast<float *>(params.rowmax_ptr) + rowmax_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.seq_stride_rowmax)); // (BlockQ)

        // Represent global tensor for dS
        Tensor gdS = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.ds_ptr) + dS_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            Stride<Int<Headdim>, _1>{}); // (BlockD, Headdim)

        // Represent global tensor for Phi(Q)
        Tensor gPhi = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.phi_ptr) + phi_offset),
            Shape<Int<BlockQ>, Int<BlockD>>{},
            make_stride(params.num_heads * paddedExpandedDim, _1{})); // (BlockQ, BlockD)

        // log_G
        Tensor glogG = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.log_G_ptr) + log_G_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.seq_stride_log_g)); // (BlockQ)

        // ====================== Share memory tensors =============================
        Tensor sQ = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutQ{}); // (BlockQ, Headdim)
        Tensor sQt = make_tensor(
            sQ.data(),
            typename Kernel_traits::SmemLayoutQt{}); // (Headdim, BlockQ)

        Tensor sdY = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sQ.data()) + int((DoubleBuffer ? 2 : 1) * size(sQ)))),
            typename Kernel_traits::SmemLayoutdY{}); // (BlockQ, Headdim)
        Tensor sdYt = make_tensor(
            sdY.data(),
            typename Kernel_traits::SmemLayoutdYt{}); // (Headdim, BlockQ)

        Tensor srowmax = make_tensor(
            make_smem_ptr(reinterpret_cast<float *>(&(*sdY.data()) + int((DoubleBuffer ? 2 : 1) * size(sdY)))),
            typename Kernel_traits::SmemLayoutRowmax{}); // (BlockQ)

        Tensor slogG = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*srowmax.data()) + int(params.has_rowmax ? (DoubleBuffer ? 2 : 1) * size(srowmax) : 0))),
            typename Kernel_traits::SmemLayoutLogG{}); // (BlockQ)

        Tensor sPQt = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*slogG.data()) + int((DoubleBuffer ? 2 : 1) * size(slogG)))),
            Layout<Shape<Int<BlockD>, Int<BlockQ>>>{}); // (BlockD, BlockQ)
        Tensor sPQ = make_tensor(
            sPQt.data(),
            composition(sPQt.layout(), make_layout(Shape<Int<BlockQ>, Int<BlockD>>{}, GenRowMajor{}))); // (BlockQ, BlockD)

        // output tensors
        Tensor sdS = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutdS{}); // (BlockD, Headdim)


        // main steps:
        // For each blockQ:
        //     * Compute phi(Q) block
        //     * Matmul phi(Q) block with dY block, get dS
        //     * Matmul phi(Q) block with dy block, get dN

        typename Kernel_traits::GmemCopyTileQ gmem_tiled_copy_Q;
        typename Kernel_traits::GmemCopyTiledY gmem_tiled_copy_dY;
        typename Kernel_traits::GmemCopyTilePhiQ gmem_tiled_copy_PhiQ;
        typename Kernel_traits::GmemCopyTileLogG gmem_tiled_copy_LogG;
        typename Kernel_traits::GmemCopyTileRowmax gmem_tiled_copy_Rowmax;

        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tid);
        auto gmem_thr_copy_dY = gmem_tiled_copy_dY.get_thread_slice(tid);
        auto gmem_thr_copy_PhiQ = gmem_tiled_copy_PhiQ.get_thread_slice(tid);
        auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tid);
        auto gmem_thr_copy_Rowmax = gmem_tiled_copy_Rowmax.get_thread_slice(tid);

        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockQ)
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tdYgdY = gmem_thr_copy_dY.partition_S(gdY); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockQ)
        Tensor tdYsdY = gmem_thr_copy_dY.partition_D(sdY); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tPsP = gmem_thr_copy_PhiQ.partition_D(sPQ); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tPgP = gmem_thr_copy_PhiQ.partition_S(gPhi); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tlogGglogG = gmem_thr_copy_LogG.partition_S(glogG); // ((CPY_ATOM_I), 1)
        Tensor tlogGslogG = gmem_thr_copy_LogG.partition_D(slogG); // ((CPY_ATOM_I), 1)
        Tensor tRMgRM = gmem_thr_copy_Rowmax.partition_S(growmax); // ((CPY_ATOM_I), 1)
        Tensor tRMsRM = gmem_thr_copy_Rowmax.partition_D(srowmax); // ((CPY_ATOM_I), 1)


        // predicates
        Tensor cQdY = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor clogG = make_identity_tensor(make_shape(size<0>(slogG)));
        Tensor tlogGcLogG = gmem_thr_copy_LogG.partition_D(clogG); // ((CPY_ATOM_I), 1)
        Tensor tQdYcQdY = gmem_thr_copy_Q.partition_S(cQdY); // we can do this because GmemCopyTileQ and GmemCopyTiledY are the same
        Tensor cRM = make_identity_tensor(srowmax.shape());
        Tensor tRMcRM = gmem_thr_copy_Rowmax.partition_S(cRM);
        
        // loaders and bumpers
        auto load_Q = [&]() {
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQdYcQdY, BlockQ);
            tQgQ.data() = tQgQ.data() + index_t(-BlockQ * params.seq_stride);
        };
        auto load_dY = [&]() {
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dY, tdYgdY, tdYsdY, tQdYcQdY, BlockQ);
            tdYgdY.data() = tdYgdY.data() + index_t(-BlockQ * params.seq_stride);
        };
        auto load_rowmax = [&]() {
            if (params.has_rowmax) {
                power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Rowmax, tRMgRM, tRMsRM, tRMcRM, BlockQ);
                tRMgRM.data() = tRMgRM.data() + index_t(-BlockQ * params.seq_stride_rowmax);
            }
        };
        auto load_log_G = [&]() {
            if (params.gating) {
                power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_LogG, tlogGglogG, tlogGslogG, tlogGcLogG, BlockQ);
                tlogGglogG.data() = tlogGglogG.data() + index_t(-BlockQ * params.num_heads);
            }
        };
        auto save_Phi = [&]() {
            Tensor cP = make_identity_tensor(gPhi.shape());
            Tensor tPcP = gmem_thr_copy_PhiQ.partition_S(cP);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_PhiQ, tPsP, tPgP, tPcP, BlockQ);
            tPgP.data() = tPgP.data() + index_t(-BlockQ * params.num_heads * paddedExpandedDim);
        };


        // MMA stuff
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);

        auto smem_tiled_copy_dYt = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_dYt = smem_tiled_copy_dYt.get_thread_slice(tid);
        Tensor tdSsdYt = smem_thr_copy_dYt.partition_S(sdYt);

        Tensor acc_ds = partition_fragment_C(tiled_mma, Shape<Int<BlockD>, Int<Headdim>>{});

        power_attention::SymowStateExpanderBlockP2<Kernel_traits> sympow;

        // Prologue
        clear(acc_ds);

        // Initialize position for share memory pointers
        if (DoubleBuffer && q_block % 2 == 1) {
            sQt.data() = sQt.data() + int(size(sQ));
            srowmax.data() = srowmax.data() + int(size(srowmax));
            tQsQ.data() = tQsQ.data() + int(size(sQ));
            tdSsdYt.data() = tdSsdYt.data() + int(size(sdY));
            tdYsdY.data() = tdYsdY.data() + int(size(sdY));
            tlogGslogG.data() = tlogGslogG.data() + int(size(slogG));
            tRMsRM.data() = tRMsRM.data() + int(size(srowmax));
        }

        // start loading Q, dy and dY
        load_Q();
        load_log_G();
        if constexpr (DoubleBuffer) {
            tQsQ.data() = tQsQ.data() + int((q_block % 2 == 0) ? size(sQ) : -size(sQ));
            tlogGslogG.data() = tlogGslogG.data() + int((q_block % 2 == 0) ? size(slogG) : -size(slogG));
        }
        load_rowmax();
        if constexpr (DoubleBuffer) {
            tRMsRM.data() = tRMsRM.data() + int((q_block % 2 == 0) ? size(srowmax) : -size(srowmax));   
        }
        cute::cp_async_fence();
        load_dY();
        if constexpr (DoubleBuffer) {
            tdYsdY.data() = tdYsdY.data() + int((q_block % 2 == 0) ? size(sdY) : -size(sdY));
        }
        cute::cp_async_fence();

        // Main loop
        // conditionally skip the first chunk if the initial state is 0, in which case acc_ds and acc_dn are zero
        if (params.non_zero_initial_state || chunk_id > 0) {
            for (; q_block >= 0; --q_block)
            {
                // start loading next Q, dy and dY, if double buffer is enabled
                if (DoubleBuffer && q_block > 0) {
                    load_Q();
                    tQsQ.data() = tQsQ.data() + int((q_block % 2 == 1) ? size(sQ) : -size(sQ));
                    load_log_G();
                    tlogGslogG.data() = tlogGslogG.data() + int((q_block % 2 == 1) ? size(slogG) : -size(slogG));
                    load_rowmax();
                    tRMsRM.data() = tRMsRM.data() + int((q_block % 2 == 1) ? size(srowmax) : -size(srowmax));
                }
                cute::cp_async_fence();
                if (DoubleBuffer && q_block > 0) {
                    load_dY();
                    tdYsdY.data() = tdYsdY.data() + int((q_block % 2 == 1) ? size(sdY) : -size(sdY));
                }
                cute::cp_async_fence();

                // wait for Q
                cute::cp_async_wait<DoubleBuffer ? 3 : 1>();
                __syncthreads();

    #ifdef QUERY_STATE_BWD_DSDN_DEBUG
                if (DEBUGGER_THREAD_DSDN) {
                    printf("q_block: %d, sQt: %p\n", q_block, sQt.data());
                    print_tensor(sQt);
                    if (params.has_rowmax) {
                        printf("q_block: %d, srowmax before expand: %p\n", q_block, srowmax.data());
                        print_tensor(srowmax);
                    }
                }
    #endif

                // expand states
                Tensor tdSrPQt = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&]() {
                    return sympow.expandState<INNER_BID>(sQt, sPQt, srowmax, slogG, tiled_mma, outer_bid, is_on_diagonal, params);
                });

    #ifdef QUERY_STATE_BWD_DSDN_DEBUG
                if (DEBUGGER_THREAD_DSDN) {
                    if (params.has_rowmax) {
                        printf("qblock: %d, srowmax after expand: %p\n", q_block, srowmax.data());
                        print_tensor(srowmax);
                    }
                    printf("q_block: %d, after expand tdSrPQt: \n", q_block);
                    print_tensor(tdSrPQt);
                    printf("\n");
                }
    #endif

                if constexpr (DoubleBuffer) {
                    sQt.data() = sQt.data() + int((q_block % 2 == 0) ? size(sQ) : -size(sQ));
                    slogG.data() = slogG.data() + int((q_block % 2 == 0) ? size(slogG) : -size(slogG));
                    if (params.has_rowmax) {
                        srowmax.data() = srowmax.data() + int((q_block % 2 == 0) ? size(srowmax) : -size(srowmax));
                    }
                }

                // if (params.return_phi) {
                //     Tensor tdSsPQt = thr_mma.partition_A(sPQt);
                //     cute::copy(tdSrPQt, tdSsPQt);
                //     __syncthreads();
                // }

                // wait for dY
                power_attention::cp_async_wait<DoubleBuffer ? 2 : 0>();
                __syncthreads();

#ifdef QUERY_STATE_BWD_DSDN_DEBUG
                if (DEBUGGER_THREAD_DSDN) {
                    printf("q_block: %d, tdSsdYt: \n", q_block);
                    print_tensor(tdSsdYt);
                    printf("\n");
                }
#endif

                // mma
                Tensor tdSrdYt = thr_mma.partition_fragment_B(sdYt);
                // TODO: right now we are only scaling dPhi(Q), if this becomes a problem we can 
                // scale dY instead, at a cost of lower perf
                power_attention::gemm_rs<false>(acc_ds, tdSrPQt, tdSrdYt, tdSsdYt, tiled_mma, smem_tiled_copy_dYt, smem_thr_copy_dYt);
                __syncthreads();

#ifdef QUERY_STATE_BWD_DSDN_DEBUG
                if (DEBUGGER_THREAD_DSDN) {
                    printf("q_block: %d, after mma acc_ds: \n", q_block);
                    print_tensor(acc_ds);
                    printf("\n");
                }
#endif

                if constexpr (DoubleBuffer) {
                    sdYt.data() = sdYt.data() + int((q_block % 2 == 0) ? size(sdY) : -size(sdY));
                    tdSsdYt.data() = tdSsdYt.data() + int((q_block % 2 == 0) ? size(sdY) : -size(sdY));
                }

                // if not double buffer, this is where we load next Q, dy and dY
                if (!DoubleBuffer && q_block > 0) {
                    load_Q();
                    load_rowmax();
                    cute::cp_async_fence();
                    load_dY();
                    cute::cp_async_fence();
                }

                // if (params.return_phi) {
                //     save_Phi();
                //     __syncthreads();
                // }
            }
        } else {
            power_attention::cp_async_wait<0>();
            __syncthreads();
        }

        // Epilogue
        // First reduce acc_dn within warps
        __syncthreads();

        // copy acc_ds back to shared memory first
        Tensor rdS = power_attention::convert_type<Element>(acc_ds);
        auto smem_tiled_copy_dS = make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tid);
        Tensor taccdSrdS = smem_thr_copy_dS.retile_S(rdS);
        Tensor taccdSsdS = smem_thr_copy_dS.partition_D(sdS);

        cute::copy(smem_tiled_copy_dS, taccdSrdS, taccdSsdS);
        __syncthreads();

#ifdef QUERY_STATE_BWD_DSDN_DEBUG
        if (DEBUGGER_THREAD_DSDN) {
            print("sdS: \n");
            print_tensor(sdS);
            print("\n");
        }
#endif

        // copy back to global memory
        typename Kernel_traits::GmemCopyTiledS gmem_tiled_copy_dS;
        auto gmem_thr_copy_dS = gmem_tiled_copy_dS.get_thread_slice(tid);
        Tensor cdS = make_identity_tensor(make_shape(size<0>(gdS), size<1>(gdS)));
        Tensor tdScdS = gmem_thr_copy_dS.partition_S(cdS);

        Tensor tdSsdS = gmem_thr_copy_dS.partition_S(sdS);
        Tensor tdSgdS = gmem_thr_copy_dS.partition_D(gdS);

        power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dS, tdSsdS, tdSgdS, tdScdS, BlockD);
    }

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 1) query_state_bwd_dQ(__grid_constant__ const Query_state_bwd_params params)
    {

        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using index_t = typename Kernel_traits::index_t;
        using multiindex_t = typename Kernel_traits::multiindex_t;
        using C_type = typename Kernel_traits::C_type;
        // Shape parameters
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int BlockQ = Kernel_traits::BlockQ;
        constexpr int Headdim = Kernel_traits::Headdim;
        constexpr int paddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int InnerBlock = Kernel_traits::InnerBlock;
        constexpr int OuterBlock = Kernel_traits::OuterBlock;
        constexpr bool DoubleBuffer = Kernel_traits::DoubleBuffer;
        constexpr bool dY_in_regs = Kernel_traits::S_in_regs;
        constexpr int NWarps = Kernel_traits::NWarps;
        // Shared memory.
        extern __shared__ char smem_[];

        // Thread, lane, warp index
        const int tid = threadIdx.x;
        const int qid = blockIdx.x;
        const int head_id = blockIdx.y;
        const int batch_id = blockIdx.z / params.num_chunks;
        const int chunk_id = blockIdx.z % params.num_chunks;
        constexpr int numBlockD = (paddedExpandedDim + BlockD - 1) / BlockD;

        auto log_G_offset_ = [&](int batch_id, int chunk_id, int head_id)
        {
            index_t chunk_stride = params.chunk_seq_len * params.num_heads;
            index_t batch_stride = params.num_chunks * chunk_stride;
            return batch_id * batch_stride + chunk_id * chunk_stride + head_id + qid * BlockQ * params.num_heads;
        };

        const index_t s_offset = batch_id * params.batch_stride_s + chunk_id * params.chunk_stride_s + head_id * params.head_stride_s;
        const index_t Q_offset = batch_id * params.batch_stride + chunk_id * params.chunk_stride + qid * BlockQ * params.seq_stride + head_id * params.head_stride;
        const index_t dY_offset = batch_id * params.batch_stride_dY + chunk_id * params.chunk_stride_dY + qid * BlockQ * params.seq_stride_dY + head_id * params.head_stride_dY;
        const index_t dY_attn_offset = batch_id * params.batch_stride_dY_attn + chunk_id * params.chunk_stride_dY_attn + qid * BlockQ * params.seq_stride_dY_attn + head_id * params.head_stride_dY_attn;
        const index_t rowmax_offset = batch_id * params.batch_stride_rowmax + chunk_id * params.chunk_stride_rowmax + qid * BlockQ * params.seq_stride_rowmax + head_id * params.head_stride_rowmax;
        const index_t logG_offset = batch_id * params.batch_stride_log_g + chunk_id * params.chunk_stride_log_g + qid * BlockQ * params.seq_stride_log_g + head_id * params.head_stride_log_g;

        // Global memory tensor for Q
        Tensor gQ = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + Q_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{}));

        // Global memory tensor for S
        Tensor gS = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.s_ptr) + s_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            Stride<Int<Headdim>, _1>{});

        // Global memory tensor for dY
        Tensor gdY = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dY_ptr) + dY_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride_dY, _1{}));

        // Global memory tensor for dY_attn
        Tensor gdY_attn = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dY_attn_ptr) + dY_attn_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride_dY_attn, _1{}));

        // Global memory tensor for rowmax
        Tensor growmax = make_tensor(
            make_gmem_ptr(reinterpret_cast<float *>(params.rowmax_ptr) + rowmax_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.seq_stride_rowmax));

        // Global memory tensor for dQ
        Tensor gdQ = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + Q_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{}));

        // Global memory tensor for logG
        Tensor glogG = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.log_G_ptr) + logG_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.seq_stride_log_g));

        // Global memory tensor for dlogG
        Tensor gdlogG = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dlog_G_ptr) + logG_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.seq_stride_log_g));

        // Share memory tensors
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

        Tensor sdY = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sS.data()) + int((DoubleBuffer ? 2 : 1) * size(sS)))),
            typename Kernel_traits::SmemLayoutdY{}); // (BlockQ, Headdim)
        Tensor sdYt = make_tensor(
            sdY.data(),
            typename Kernel_traits::SmemLayoutdYt{}); // (Headdim, BlockQ)
        
        Tensor slogG = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sdY.data()) + size(sdY))),
            typename Kernel_traits::SmemLayoutLogG{}); // (BlockQ)

        Tensor srowmax = make_tensor(
            make_smem_ptr(reinterpret_cast<float *>(&(*slogG.data()) + size(slogG))),
            typename Kernel_traits::SmemLayoutRowmax{}); // (BlockQ)

        Tensor sdPQ = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*srowmax.data()) + (params.has_rowmax ? size(srowmax) : 0))),
            typename Kernel_traits::SmemLayoutPhiQ{}); // (BlockQ, BlockD)

        // be careful we are reusing smem_ here
        Tensor sdQ = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutdQ{}); // (BlockQ, Headdim)


        typename Kernel_traits::GmemCopyTileQ gmem_tiled_copy_Q;
        typename Kernel_traits::GmemCopyTileS gmem_tiled_copy_S;
        typename Kernel_traits::GmemCopyTiledY gmem_tiled_copy_dY;
        typename Kernel_traits::GmemCopyTiledS gmem_tiled_copy_dS;
        typename Kernel_traits::GmemCopyTiledLogG gmem_tiled_copy_logG;
        typename Kernel_traits::GmemCopyTileRowmax gmem_tiled_copy_rowmax;

        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tid);
        auto gmem_thr_copy_S = gmem_tiled_copy_S.get_thread_slice(tid);
        auto gmem_thr_copy_dY = gmem_tiled_copy_dY.get_thread_slice(tid);
        auto gmem_thr_copy_dS = gmem_tiled_copy_dS.get_thread_slice(tid);
        auto gmem_thr_copy_logG = gmem_tiled_copy_logG.get_thread_slice(tid);
        auto gmem_thr_copy_rowmax = gmem_tiled_copy_rowmax.get_thread_slice(tid);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tSgS = gmem_thr_copy_S.partition_S(gS);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockD)
        Tensor tSsS = gmem_thr_copy_S.partition_D(sS);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tdYgdY = gmem_thr_copy_dY.partition_S(gdY); // ((CPY_ATOM_I, CPY_ATOM_J), 1, TILE_N)
        Tensor tdYsdY = gmem_thr_copy_dY.partition_D(sdY); // ((CPY_ATOM_I, CPY_ATOM_J), 1, TILE_N)
        Tensor tRMgRM = gmem_thr_copy_rowmax.partition_S(growmax); // ((CPY_ATOM_I), 1)
        Tensor tRMsRM = gmem_thr_copy_rowmax.partition_D(srowmax); // ((CPY_ATOM_I), 1)
        Tensor tlogGglogG = gmem_thr_copy_logG.partition_S(glogG); // ((CPY_ATOM_I), 1)
        Tensor tlogGslogG = gmem_thr_copy_logG.partition_D(slogG); // ((CPY_ATOM_I), 1)

        // Predicates
        Tensor cS = make_identity_tensor(make_shape(size<0>(sS), size<1>(sS)));
        Tensor tScS = gmem_thr_copy_S.partition_S(cS);

        // Loaders and bumpers
        auto load_Q = [&]() {
            Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
            Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, BlockQ);
        };
        auto load_dY = [&]() {
            Tensor cdY = make_identity_tensor(make_shape(size<0>(sdY), size<1>(sdY)));
            Tensor tdYcdY = gmem_thr_copy_dY.partition_S(cdY);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dY, tdYgdY, tdYsdY, tdYcdY, BlockQ);
        };
        auto save_dY = [&]() {
            typename Kernel_traits::GmemCopyTiledQ gmem_tiled_copy_back_dY;
            auto gmem_thr_copy_back_dY = gmem_tiled_copy_back_dY.get_thread_slice(tid);
            Tensor cdY = make_identity_tensor(make_shape(size<0>(sdY), size<1>(sdY)));
            Tensor tdYcdY_back = gmem_thr_copy_back_dY.partition_D(cdY);
            Tensor tdYsdY_back = gmem_thr_copy_back_dY.partition_S(sdY);
            Tensor tdYgdY_back = gmem_thr_copy_back_dY.partition_D(gdY_attn);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_back_dY, tdYsdY_back, tdYgdY_back, tdYcdY_back, BlockQ);
        };
        auto load_S = [&]() {
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_S, tSgS, tSsS, tScS, BlockD);
            tSgS.data() = tSgS.data() + int(BlockD * Headdim);
        };
        auto load_rowmax = [&]() {
            if (params.has_rowmax) {
                Tensor cRM = make_identity_tensor(make_shape(size(srowmax)));
                Tensor tRMcRM = gmem_thr_copy_rowmax.partition_S(cRM);
                power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_rowmax, tRMgRM, tRMsRM, tRMcRM, BlockQ);
            }
        };
        auto load_logG = [&]() {
            if (params.gating) {
                Tensor clogG = make_identity_tensor(make_shape(size(slogG)));
                Tensor tlogGclogG = gmem_thr_copy_logG.partition_S(clogG);
                power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_logG, tlogGglogG, tlogGslogG, tlogGclogG, BlockQ);
            }
        };
        auto bump_sS = [&](const int d_block) {
            if constexpr (DoubleBuffer) {
                tSsS.data() = tSsS.data() + int((d_block % 2 == 0) ? size(sS) : -size(sS));
            }
        };

        // MMA stuff
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);
        
        // create tensors for smem->register for mma_dPQ
        auto smem_tiled_copy_dY = make_tiled_copy_A(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_dY = smem_tiled_copy_dY.get_thread_slice(tid);

        auto smem_tiled_copy_S = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tid);

        Tensor tdPQsS = smem_thr_copy_S.partition_S(sS);
        Tensor tdPQsdY = smem_thr_copy_dY.partition_S(sdY);

        SymowStateExpanderBlockP2<Kernel_traits> sympow;

        Tensor tdPQrdY = thr_mma.partition_fragment_A(sdY);
    
        // Prologue
        // Copy dY and first S block
        static_assert(dY_in_regs, "dY must be in registers for this iteration");
        if constexpr (dY_in_regs) {
            load_dY();
            load_rowmax();
            cute::cp_async_fence();
        }

        load_S();
        bump_sS(0);

        // load Q
        load_Q();
        load_logG();
        cute::cp_async_fence();

        if constexpr (dY_in_regs) {
            power_attention::cp_async_wait<1>();
            __syncthreads();
            // read dY from shared memory
            Tensor tdPQrdY_copy_view = smem_thr_copy_dY.retile_D(tdPQrdY);
            cute::copy(smem_tiled_copy_dY, tdPQsdY, tdPQrdY_copy_view);
            if (params.has_rowmax) { // if fused
                Tensor r_rowmax = power::read_row_summary<size<1>(tdPQrdY) * 2, typename Kernel_traits::TiledMma>(srowmax);
                Tensor tdPQrdY_rowcol = make_tensor(tdPQrdY.data(), convert_layout_rA_rowcol(tdPQrdY.layout()));
                Tensor rdY_attn = make_tensor_like(tdPQrdY);
                Tensor rdY_attn_rowcol = make_tensor(rdY_attn.data(), convert_layout_rA_rowcol(rdY_attn.layout()));
                if (params.non_zero_initial_state || chunk_id > 0) {
                    // Scaling formula:
                    // * qs_scale = multiplier^2
                    // * attn_scale = exp(-rowmax)
                    // * if qs_scale > attn_scale:
                    // *   dY *= attn_scale / qs_scale = exp(-rowmax) / multiplier^2
                    // *   rdy *= attn_scale / qs_scale = exp(-rowmax) / multiplier^2
                    // * else:
                    // *   dY_attn = dY * qs_scale / attn_scale
                    // *   rdy_attn = rdy * qs_scale / attn_scale
                    // *
                    // * Note:
                    // *   In the first case, because dY will be adjusted by qs_scale again in mma, i.e.,
                    // *     dPhi(Q) = dY * qs_scale @ S^T + dy * qs_scale @ N^T
                    // *   we can fuse the scaling here, i.e.,
                    // *   If qs_scale > attn_scale:
                    // *     dY *= attn_scale
                    // *     rdy *= attn_scale
                    // *   else:
                    // *     dY_attn = dY * qs_scale / attn_scale
                    // *     rdy_attn = rdy * qs_scale / attn_scale
                    // *   Then in mma, we can do:
                    // *     dPhi(Q) = dY_attn @ S^T + rdy_attn @ N^T
                    CUTE_UNROLL
                    for (int i = 0; i < size(r_rowmax); i++) {
                        r_rowmax(i) = expf(-r_rowmax(i));
                    }
                    CUTE_UNROLL
                    for (int m = 0; m < size<0>(tdPQrdY_rowcol); m++) {
                        if (params.multiplier_squared > (r_rowmax(m))) {
                            CUTE_UNROLL
                            for (int n = 0; n < size<1>(tdPQrdY_rowcol); n++) {
                                rdY_attn_rowcol(m, n) = tdPQrdY_rowcol(m, n);
                                tdPQrdY_rowcol(m, n) = static_cast<Element>(power_attention::fp32_mul(tdPQrdY_rowcol(m, n), r_rowmax(m)));
                            }
                        } else {
                            CUTE_UNROLL
                            for (int n = 0; n < size<1>(tdPQrdY_rowcol); n++) {
                                rdY_attn_rowcol(m, n) = static_cast<Element>(power_attention::fp32_mul(tdPQrdY_rowcol(m, n), params.multiplier_squared / r_rowmax(m)));
                                tdPQrdY_rowcol(m, n) = static_cast<Element>(power_attention::fp32_mul(tdPQrdY_rowcol(m, n), params.multiplier_squared));
                            }
                        }
                    }
                } else { // if zero initial state and chunk_id == 0, just copy dY to rdY_attn
                    cute::copy(tdPQrdY, rdY_attn);
                }

                // copy dY back to shared memory and global memory for attention backward
                // TODO (sean): use stmatrix when available
                auto smem_tiled_copy_back_dY = make_tiled_copy_A(
                    Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
                auto smem_thr_copy_back_dY = smem_tiled_copy_back_dY.get_thread_slice(tid);
                Tensor rdY_attn_copy_view = smem_thr_copy_back_dY.retile_S(rdY_attn);
                Tensor tdPQsdY_D = smem_thr_copy_back_dY.partition_D(sdY);
                cute::copy(smem_tiled_copy_back_dY, rdY_attn_copy_view, tdPQsdY_D);

                __syncthreads();
                save_dY();
            } else if (params.use_multiplier && (params.non_zero_initial_state || chunk_id > 0)) {
                // if not fused, we need to scale dY by multiplier^2, scale dy by multiplier^2
                power_attention::elementwise_product(tdPQrdY, params.multiplier, tdPQrdY);
            }

#ifdef QUERY_STATE_BWD_DQ_DEBUG
            if (DEBUGGER_THREAD_DQ) {
                print("tdPQrdY: \n");
                print_tensor(tdPQrdY);
                print("\n");
            }
#endif

        } else {
            load_dY();
            load_rowmax();
            cute::cp_async_fence();
        }

        // accumulator for dQ
        Tensor acc_dq = partition_fragment_C(tiled_mma, Shape<Int<BlockQ>, Int<Headdim>>{}); // ((2, 2), 1, 4)
        clear(acc_dq);

        // Main loop
        // conditionally skip the first chunk if the initial state is 0, in which case acc_dq is zero
        if (params.non_zero_initial_state || chunk_id > 0) {
            for (int d_block = 0, inner_bid = 0, outer_bid = 0; d_block < numBlockD; ++d_block)
            {
                if (DoubleBuffer && d_block < numBlockD - 1) // start loading next S, N
                {
                    // load S block
                    load_S();
                    bump_sS(d_block + 1);
                }
                cute::cp_async_fence();
                // wait for everything to be loaded except for the next round
                cute::cp_async_wait<DoubleBuffer ? 1 : 0>();
                __syncthreads();

    #ifdef QUERY_STATE_BWD_DQ_DEBUG
                if (DEBUGGER_THREAD_DQ) {
                    // print("QUERY_STATE_BWD_DQ, sS: \n");
                    // print_tensor(sS);
                    // print("\n");
                    // print("QUERY_STATE_BWD_DQ, sN: \n");
                    // print_tensor(sN);
                    // print("\n");
                    if (params.has_rowmax) {
                        print("QUERY_STATE_BWD_DQ, srowmax: \n");
                        print_tensor(srowmax);
                        print("\n");
                    }
                    // print("QUERY_STATE_BWD_DQ, sQ: \n");
                    // print_tensor(sQ);
                    // print("\n");
                    // print("QUERY_STATE_BWD_DQ, sdy: \n");
                    // print_tensor(sdy);
                    // print("\n");
                    // print("QUERY_STATE_BWD_DQ, sdY: \n");
                    // print_tensor(sdY);
                    // print("\n");
                }
    #endif

                // matmul for dPhiQ
                Tensor acc_dpq = partition_fragment_C(tiled_mma, Shape<Int<BlockQ>, Int<BlockD>>{});

                Tensor tdPQrS = thr_mma.partition_fragment_B(sS);
                if (params.has_rowmax) { // if fused, we don't need to scale dY and rdy, it's already done
                    power_attention::gemm</*A_in_regs=*/dY_in_regs, /*B_in_regs=*/false, /*rescale_B=*/false, /*rescale_A=*/false>(acc_dpq, tdPQrdY, tdPQrS, tdPQsdY, tdPQsS, tiled_mma, smem_tiled_copy_dY, smem_tiled_copy_S, smem_thr_copy_dY, smem_thr_copy_S);
                } else if (params.use_multiplier) { // if not fused, dY is alread scaled by multiplier if it's already in register, but S needs to be scaled
                    power_attention::gemm</*A_in_regs=*/dY_in_regs, /*B_in_regs=*/false, /*rescale_B=*/true, /*rescale_A=*/!dY_in_regs>(acc_dpq, tdPQrdY, tdPQrS, tdPQsdY, tdPQsS, tiled_mma, smem_tiled_copy_dY, smem_tiled_copy_S, smem_thr_copy_dY, smem_thr_copy_S, params.multiplier, params.multiplier);
                } else { // if no scaling is required
                    power_attention::gemm</*A_in_regs=*/dY_in_regs, /*B_in_regs=*/false, /*rescale_B=*/false, /*rescale_A=*/false>(acc_dpq, tdPQrdY, tdPQrS, tdPQsdY, tdPQsS, tiled_mma, smem_tiled_copy_dY, smem_tiled_copy_S, smem_thr_copy_dY, smem_thr_copy_S);
                }
                
                if constexpr (DoubleBuffer) {
                    tdPQsS.data() = tdPQsS.data() + int((d_block % 2 == 0) ? size(sS) : -size(sS));
                } else {
                    __syncthreads();
                    load_S();
                    cute::cp_async_fence();
                }

                // backprop to dQ
                sympow.graddQK</*Adjust_Coefficient=*/false>(sQ, acc_dpq, acc_dq, tiled_mma, inner_bid, outer_bid);

                // bump inner_bid, outer_bid
                binfo_bump<OuterBlock, InnerBlock>(inner_bid, outer_bid);

                __syncthreads();
            }
        } else {
            power_attention::cp_async_wait<0>();
            __syncthreads();
        }

        // Epilogue
        // backprop one more time for logG, and compute dlogG
        if (params.gating) {

#ifdef QUERY_STATE_BWD_DQ_DEBUG
            if (DEBUGGER_THREAD_DQ) {
                print("QUERY_STATE_BWD_DLOGG, has gating:\n");
            }
#endif
            Tensor acc_dq_rowcol = make_tensor(acc_dq.data(), convert_layout_acc_rowcol(acc_dq.layout()));
            Tensor rdlogG = make_tensor<ElementAccum>(get<0>(acc_dq_rowcol.layout()));
            clear(rdlogG);
            // load Q to register
            auto smem_tiled_copy_Q = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, Element>{}, tiled_mma);
            auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
            Tensor tQsQ = smem_thr_copy_Q.partition_S(sQ); // ((4), TILE_Q, TILE_HEADDIM)
            Tensor acc_rQ = make_tensor<Element>(thr_mma.partition_C(sQ).shape()); // ((4), TILE_Q, TILE_HEADDIM)
            Tensor rlogG_col = make_tensor<ElementAccum>(get<0>(acc_rQ.layout()));
            Tensor acc_rQ_rowcol = make_tensor(acc_rQ.data(), convert_layout_acc_rowcol(acc_rQ.layout()));
            Tensor tQrQ = smem_thr_copy_Q.retile_D(acc_rQ); // ((4), TILE_Q, TILE_HEADDIM)
            cute::copy(smem_tiled_copy_Q, tQsQ, tQrQ);

            // load logG to register
            if constexpr (size(rlogG_col) > 2) {
                CUTE_UNROLL
                for (int i = 0; i < size(rlogG_col); i++) {
                    rlogG_col(i) = slogG((i / 2) * NWarps * 16 + (i % 2) * 8 + (tid % 32) / 4 + (tid / 32) * 16);
                }
            }
            else {
                CUTE_UNROLL
                for (int i = 0; i < size(rlogG_col); i++) {
                    rlogG_col(i) = slogG(i * 8 + (tid % 32) / 4 + (tid / 32) * 16);
                }
            }

            // backprop one more time for logG
            CUTE_UNROLL
            for (int m = 0; m < size<0>(acc_dq_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dq_rowcol); n++) {
                    acc_dq_rowcol(m, n) *= expf(rlogG_col(m) / 2.0f);
                }
            }

            // compute rdlogG
            CUTE_UNROLL
            for (int m = 0; m < size<0>(rdlogG); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(rdlogG); n++) {
                    rdlogG(m) += static_cast<ElementAccum>(acc_rQ_rowcol(m, n) * acc_dq_rowcol(m, n) * expf(rlogG_col(m) / 2.0f) / 2.0f);
                }
            }

            // put dlogG back in shared memory
            __syncthreads(); // sync necessary because we reuse slogG for sdlogG
            Tensor sdlogG = make_tensor(
                slogG.data(),
                typename Kernel_traits::SmemLayoutLogG{}); // (BlockQ)
            
            if constexpr (size(rdlogG) > 2) {
                CUTE_UNROLL
                for (int i = 0; i < size(rdlogG); i++) {
                    sdlogG((i / 2) * NWarps * 16 + (i % 2) * 8 + (tid % 32) / 4 + (tid / 32) * 16) = rdlogG(i);
                }
            }
            else {
                CUTE_UNROLL
                for (int i = 0; i < size(rdlogG); i++) {
                    sdlogG(i * 8 + (tid % 32) / 4 + (tid / 32) * 16) = rdlogG(i);
                }
            }

            // put dlogG back to global memory
            typename Kernel_traits::GmemCopyTileLogG gmem_tiled_copy_dlogG;
            auto gmem_thr_copy_dlogG = gmem_tiled_copy_dlogG.get_thread_slice(tid);
            Tensor cdlogG = make_identity_tensor(sdlogG.shape());
            Tensor tdlogGcdlogG = gmem_thr_copy_dlogG.partition_S(cdlogG);
            Tensor tdlogGgdlogG = gmem_thr_copy_dlogG.partition_D(gdlogG);
            Tensor tdlogGsdlogG = gmem_thr_copy_dlogG.partition_S(sdlogG);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dlogG, tdlogGsdlogG, tdlogGgdlogG, tdlogGcdlogG, BlockQ);
        }

#ifdef QUERY_STATE_BWD_DQ_DEBUG
        if (DEBUGGER_THREAD_DQ) {
            print("QUERY_STATE_BWD_DQ, final acc_dq: \n");
            print_tensor(acc_dq);
            print("\n");
        }
#endif

        // convert dQaccum to dQ, put back in shared memory
        Tensor rdQ = make_tensor<Element>(acc_dq.layout());
        power_attention::convert_type(acc_dq, rdQ);
        auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma);
        auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tid);
        Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);
        Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);

#ifdef QUERY_STATE_BWD_DQ_DEBUG
        if (DEBUGGER_THREAD_DQ) {
            print("QUERY_STATE_BWD_DQ, rdQ: \n");
            print_tensor(rdQ);
            print("\n");
        }
#endif

        __syncthreads(); // this sync is necessary because we reuse sQ for sdQ
        cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);

        // put dQ back to global memory
        typename Kernel_traits::GmemCopyTiledQ gmem_tiled_copy_dQ;
        auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tid);
        Tensor cdQ = make_identity_tensor(sdQ.shape());
        Tensor tdQcdQ = gmem_thr_copy_dQ.partition_S(cdQ);
        Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
        Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sQ);

        __syncthreads();

#ifdef QUERY_STATE_BWD_DQ_DEBUG
        if (DEBUGGER_THREAD_DQ) {
            print("QUERY_STATE_BWD_DQ, sdQ: \n");
            print_tensor(sdQ);
            print("\n");
        }
#endif
        power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dQ, tdQsdQ, tdQgdQ, tdQcdQ, BlockQ);
    }

    template <typename Kernel_traits, typename Params>
    inline __device__ void query_state_bwd_dSdNdQ_impl(const Params &params, const int did, const int head_id, const int batch_id, const int chunk_id)
    {
        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using index_t = typename Kernel_traits::index_t;
        using multiindex_t = typename Kernel_traits::multiindex_t;
        using C_type = typename Kernel_traits::C_type;
        using MMA_ATOM = typename Kernel_traits::MMA_ATOM;
        using Multiply_vector_t = float;
        // Shape parameters
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int BlockQ = Kernel_traits::BlockQ;
        constexpr int Headdim = Kernel_traits::Headdim;
        constexpr int NWarps = Kernel_traits::NWarps;
        constexpr int paddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int NThreads = Kernel_traits::NThreads;
        constexpr int InnerBlock = Kernel_traits::InnerBlock;
        constexpr int OuterBlock = Kernel_traits::OuterBlock;
        constexpr bool DoubleBuffer = Kernel_traits::DoubleBuffer;
        constexpr bool S_in_regs = Kernel_traits::S_in_regs;
        constexpr bool rescale = std::is_same_v<Element, cutlass::half_t>;
        static_assert(paddedExpandedDim % BlockD == 0, "paddedExpandedDim must be divisible by BlockD");
        // Shared memory.
        __align__(128) extern __shared__ char smem_[];

        // Thread, lane, warp index
        const int tid = threadIdx.x;

        int q_block = (params.chunk_seq_len + BlockQ - 1) / BlockQ - 1;

#ifdef QUERY_STATE_BWD_DEBUG
#define DEBUGGER_THREAD_QSBWD (tid == 0 && did == 0 && head_id == 0 && batch_id == 0 && chunk_id == 0)
#endif

        auto dN_offset_ = [&](int batch_id, int chunk_id, int head_id, int did)
        {
            index_t head_stride = paddedExpandedDim;
            index_t chunk_stride = params.num_heads * head_stride;
            index_t batch_stride = params.num_chunks * chunk_stride;
            return batch_id * batch_stride + chunk_id * chunk_stride + head_id * head_stride + did * BlockD;    
        };
        auto phi_offset_ = [&](int batch_id, int chunk_id, int head_id, int did)
        {
            index_t head_stride = paddedExpandedDim;
            index_t seq_stride = params.num_heads * head_stride;
            index_t chunk_stride = params.chunk_seq_len * seq_stride;
            index_t batch_stride = params.num_chunks * chunk_stride;
            return batch_id * batch_stride + chunk_id * chunk_stride + q_block * BlockQ * seq_stride + head_id * head_stride + did * BlockD;
        };
        auto log_G_offset_ = [&](int batch_id, int chunk_id, int head_id)
        {
            index_t chunk_stride = params.chunk_seq_len * params.num_heads;
            index_t batch_stride = params.num_chunks * chunk_stride;
            return batch_id * batch_stride + chunk_id * chunk_stride + q_block * BlockQ * params.num_heads + head_id;
        };

        // strides
        const index_t qdY_offset = batch_id * params.batch_stride + chunk_id * params.chunk_stride + head_id * params.head_stride + q_block * BlockQ * params.seq_stride;
        const index_t dQaccum_offset = qdY_offset + (params.deterministic ? (blockIdx.x * params.dq_accum_split_stride) : 0);
        const index_t dy_offset = batch_id * params.batch_stride_dy + chunk_id * params.chunk_stride_dy + head_id * params.head_stride_dy + q_block * BlockQ * params.seq_stride_dy;
        const index_t dS_offset = batch_id * params.batch_stride_s + chunk_id * params.chunk_stride_s + head_id * params.head_stride_s + did * BlockD * Headdim;
        const index_t dN_offset = dN_offset_(batch_id, chunk_id, head_id, did);
        const index_t phi_offset = phi_offset_(batch_id, chunk_id, head_id, did);
        const index_t log_G_offset = log_G_offset_(batch_id, chunk_id, head_id);

        // ====================== Global tensors =============================
        // Represent global tensor for Q
        Tensor gQ = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + qdY_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockQ, Headdim)

        // Represent global tensor for dY
        Tensor gdY = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.dY_ptr) + qdY_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockQ, Headdim)

        // Represent global tensor for dy
        Tensor gdy = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dy_ptr) + dy_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.num_heads)); // (BlockQ)

        // Represent global tensor for S
        Tensor gS = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.s_ptr) + dS_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            Stride<Int<Headdim>, _1>{});

        // Represent global tensor for dS
        Tensor gdS = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.ds_ptr) + dS_offset),
            Shape<Int<BlockD>, Int<Headdim>>{},
            Stride<Int<Headdim>, _1>{}); // (BlockD, Headdim)

        // Global memory tensor for norm
        Tensor gN = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.norm_ptr) + dN_offset),
            Shape<Int<BlockD>>{},
            Stride<_1>{});

        // Represent global tensor for dnorm
        Tensor gdN = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dnorm_ptr) + dN_offset),
            Shape<Int<BlockD>>{},
            Stride<_1>{}); // (BlockD)

        // Represent global tensor for Phi(Q)
        Tensor gPhi = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element *>(params.phi_ptr) + phi_offset),
            Shape<Int<BlockQ>, Int<BlockD>>{},
            make_stride(params.num_heads * paddedExpandedDim, _1{})); // (BlockQ, BlockD)

        // log_G
        Tensor glogG = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.log_G_ptr) + log_G_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.num_heads)); // (BlockQ)

        // dlog_G
        Tensor gdlogG = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dlog_G_ptr) + log_G_offset),
            Shape<Int<BlockQ>>{},
            make_stride(params.num_heads));

        // gdQaccum
        Tensor gdQaccum = make_tensor(
            make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.gdQaccum_ptr) + dQaccum_offset),
            Shape<Int<BlockQ>, Int<Headdim>>{},
            make_stride(params.seq_stride, _1{})); // (BlockQ, Headdim)

        // ====================== Share memory tensors =============================
        Tensor sQ = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutQ{}); // (BlockQ, Headdim)
        Tensor sQt = make_tensor(
            sQ.data(),
            typename Kernel_traits::SmemLayoutQt{}); // (Headdim, BlockQ)

        Tensor sdY = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sQ.data()) + (DoubleBuffer ? 2 : 1) * size(sQ))),
            typename Kernel_traits::SmemLayoutdY{}); // (BlockQ, Headdim)
        Tensor sdYt = make_tensor(
            sdY.data(),
            typename Kernel_traits::SmemLayoutdYt{}); // (Headdim, BlockQ)

        Tensor sS = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(&(*sdY.data()) + size(sdY))),
            typename Kernel_traits::SmemLayoutS{}); // (BlockD, Headdim)
            
        Tensor sN = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sS.data()) + size(sS))),
            typename Kernel_traits::SmemLayoutN{}); // (BlockD)

        Tensor sdy = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sN.data()) + size(sN))),
            typename Kernel_traits::SmemLayoutdy{}); // (BlockQ)

        Tensor sdQaccum = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sdy.data()) + size(sdy))),
            typename Kernel_traits::SmemLayoutdQaccum{}); // (BlockQ, BlockD)

        Tensor slogG = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sdQaccum.data()) + size(sdQaccum))),
            typename Kernel_traits::SmemLayoutLogG{}); // (BlockQ)

        // This will not actually be used, only serves for shape information
        Tensor sPQt = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*slogG.data()) + (DoubleBuffer ? 2 : 1) * size(slogG))),
            Layout<Shape<Int<BlockD>, Int<BlockQ>>>{}); // (BlockD, BlockQ)
        Tensor sPQ = make_tensor(
            sPQt.data(),
            composition(sPQt.layout(), make_layout(Shape<Int<BlockQ>, Int<BlockD>>{}, GenRowMajor{}))); // (BlockQ, BlockD)

        // output tensors
        Tensor sdS = make_tensor(
            make_smem_ptr(reinterpret_cast<Element *>(smem_)),
            typename Kernel_traits::SmemLayoutdS{}); // (BlockD, Headdim)

        Tensor sdN = make_tensor(
            make_smem_ptr(reinterpret_cast<ElementAccum *>(&(*sdS.data()) + size(sdS))),
            typename Kernel_traits::SmemLayoutdN{}); // (BlockD)


        // main steps:
        // prologue:
        //  * Load S, N block into smem
        //  * Load Q, dY, dy, log_G block into smem
        //  * If S_in_regs, load S^T, N^T into registers
        // main loop:
        //  * load next Q, log_G block
        //  * syncthreads, wait for prev Q, log_G, dY, dy
        //  * expand states, Q + log_G -> tdSrPQt
        //  * init acc_dPQ
        //  * PQ^T @ dY -> acc_dS
        //  * dY @ S^T -> acc_dPQ
        //  * load next dY
        //  * if gating
        //    * read log_G -> rlogG
        //    * read Q -> rQ
        //  * PQ^T @ dy -> acc_dN
        //  * dy @ N^T -> acc_dPQ
        //  * load next dy
        //  * acc_dPQ -> rdQ, using rlogG, atomicAdd
        //  * acc_dPQ -> rdlogG, using rQ, reduction, then atomicAdd
        // epilogue:
        //  * syncthreads
        //  * write back dS, dN


        typename Kernel_traits::GmemCopyTileQ gmem_tiled_copy_Q;
        typename Kernel_traits::GmemCopyTileS gmem_tiled_copy_S;
        typename Kernel_traits::GmemCopyTileN gmem_tiled_copy_N;
        typename Kernel_traits::GmemCopyTiledY gmem_tiled_copy_dY;
        typename Kernel_traits::GmemCopyTiledy gmem_tiled_copy_dy;
        typename Kernel_traits::GmemCopyTilePhiQ gmem_tiled_copy_PhiQ;
        typename Kernel_traits::GmemCopyTileLogG gmem_tiled_copy_LogG;

        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tid);
        auto gmem_thr_copy_S = gmem_tiled_copy_S.get_thread_slice(tid);
        auto gmem_thr_copy_N = gmem_tiled_copy_N.get_thread_slice(tid);
        auto gmem_thr_copy_dY = gmem_tiled_copy_dY.get_thread_slice(tid);
        auto gmem_thr_copy_dy = gmem_tiled_copy_dy.get_thread_slice(tid);
        auto gmem_thr_copy_PhiQ = gmem_tiled_copy_PhiQ.get_thread_slice(tid);
        auto gmem_thr_copy_LogG = gmem_tiled_copy_LogG.get_thread_slice(tid);

        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockQ)
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tSgS = gmem_thr_copy_S.partition_S(gS);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tSsS = gmem_thr_copy_S.partition_D(sS);     // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tNgN = gmem_thr_copy_N.partition_S(gN);      // ((CPY_ATOM_I), 1)
        Tensor tNsN = gmem_thr_copy_N.partition_D(sN);      // ((CPY_ATOM_I), 1)
        Tensor tdYgdY = gmem_thr_copy_dY.partition_S(gdY); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N, nBlockQ)
        Tensor tdYsdY = gmem_thr_copy_dY.partition_D(sdY); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tdygdy = gmem_thr_copy_dy.partition_S(gdy); // ((CPY_ATOM_I), 1, nBlockQ)
        Tensor tdysdy = gmem_thr_copy_dy.partition_D(sdy); // ((CPY_ATOM_I), 1)
        Tensor tPsP = gmem_thr_copy_PhiQ.partition_D(sPQ); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tPgP = gmem_thr_copy_PhiQ.partition_S(gPhi); // ((CPY_ATOM_I, CPY_ATOM_J), TILE_M, TILE_N)
        Tensor tlogGglogG = gmem_thr_copy_LogG.partition_S(glogG); // ((CPY_ATOM_I), 1)
        Tensor tlogGslogG = gmem_thr_copy_LogG.partition_D(slogG); // ((CPY_ATOM_I), 1)

        // predicates
        Tensor cQdY = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
        Tensor cdy = make_identity_tensor(make_shape(size(sdy)));
        Tensor cN = make_identity_tensor(make_shape(size(sN)));
        Tensor cS = make_identity_tensor(sS.shape());

        Tensor tQdYcQdY = gmem_thr_copy_Q.partition_S(cQdY); // we can do this because GmemCopyTileQ and GmemCopyTiledY are the same
        Tensor tdycdy = gmem_thr_copy_dy.partition_S(cdy);
        Tensor tNcN = gmem_thr_copy_N.partition_S(cN);
        Tensor tScS = gmem_thr_copy_S.partition_S(cS);
        
        // loaders and bumpers
        auto load_Q = [&]() {
            power_attention::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true, BlockQ>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQdYcQdY, std::min(BlockQ, params.chunk_seq_len - q_block * BlockQ));
            tQgQ.data() = tQgQ.data() + index_t(-BlockQ * params.seq_stride);
        };
        auto load_dY = [&]() {
            power_attention::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true, BlockQ>(gmem_tiled_copy_dY, tdYgdY, tdYsdY, tQdYcQdY, std::min(BlockQ, params.chunk_seq_len - q_block * BlockQ));
            tdYgdY.data() = tdYgdY.data() + index_t(-BlockQ * params.seq_stride);
        };
        auto load_S = [&]() {
            power_attention::copy</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true, BlockD>(gmem_tiled_copy_S, tSgS, tSsS, tScS, BlockD);
        };
        auto load_N = [&]() {
            power::copy1d</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true, BlockD>(gmem_tiled_copy_N, tNgN, tNsN, tNcN, BlockD);
        };
        auto load_dy = [&]() {
            power::copy1d</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true, BlockQ>(gmem_tiled_copy_dy, tdygdy, tdysdy, tdycdy, std::min(BlockQ, params.chunk_seq_len - q_block * BlockQ));
            tdygdy.data() = tdygdy.data() + index_t(-BlockQ * params.seq_stride_dy);
        };
        auto load_log_G = [&]() {
            if (params.gating) {
                power::copy1d</*Is_even_MN=*/false, /*Clear_OOB_MN=*/true, BlockQ>(gmem_tiled_copy_LogG, tlogGglogG, tlogGslogG, tdycdy, std::min(BlockQ, params.chunk_seq_len - q_block * BlockQ));
                tlogGglogG.data() = tlogGglogG.data() + index_t(-BlockQ * params.num_heads);
            }
        };
        auto save_Phi = [&]() {
            Tensor cP = make_identity_tensor(gPhi.shape());
            Tensor tPcP = gmem_thr_copy_PhiQ.partition_S(cP);
            power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_PhiQ, tPsP, tPgP, tPcP, std::min(BlockQ, params.chunk_seq_len - q_block * BlockQ));
            tPgP.data() = tPgP.data() + index_t(-BlockQ * params.num_heads * paddedExpandedDim);
        };


        // MMA stuff
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tid);
        // we use a different tiled mma to create the right shape of acc_dq
        // this is purely for the shape info
        auto tiled_mma_trans = TiledMMA<
            MMA_ATOM,
            Layout<Shape<_1, Int<NWarps>, _1>>,
            Tile<_16, Layout<Shape<_8, Int<NWarps>, _2>, Stride<_1, _16, _8>>, _16>>{};
        auto thr_mma_trans = tiled_mma_trans.get_thread_slice(tid);

        auto smem_tiled_copy_dYt = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_dYt = smem_tiled_copy_dYt.get_thread_slice(tid);
        Tensor tdSsdYt = smem_thr_copy_dYt.partition_S(sdYt);

        Tensor tdSrdYt = thr_mma.partition_fragment_B(sdYt); // (MMA_V, TILES_HEADDIM, TILES_T)
        Tensor tdPQrS = thr_mma_trans.partition_fragment_B(sS); // (MMA_V, TILES_D, TILES_HEADDIM)

        auto smem_tiled_copy_dY = make_tiled_copy_A(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma_trans);
        auto smem_thr_copy_dY = smem_tiled_copy_dY.get_thread_slice(tid);

        auto smem_tiled_copy_S = make_tiled_copy_B(
            typename Kernel_traits::SmemCopyAtom{}, tiled_mma_trans);
        auto smem_thr_copy_S = smem_tiled_copy_S.get_thread_slice(tid);

        Tensor tdPQsS = smem_thr_copy_S.partition_S(sS);
        Tensor tdPQsdY = smem_thr_copy_dY.partition_S(sdY);

        Tensor acc_ds = partition_fragment_C(tiled_mma, Shape<Int<BlockD>, Int<Headdim>>{});
        using regA_layout = decltype(thr_mma.partition_fragment_A(sPQt).layout());
        using regA_rowcol = decltype(power_attention::convert_layout_rA_rowcol(regA_layout{}));
        Tensor acc_dn = make_tensor<ElementAccum>(get<0>(regA_rowcol{}));

        power_attention::SymowStateExpanderBlockP2<Kernel_traits> sympow;

        // Prologue
        clear(acc_ds);
        clear(acc_dn);

        // start loading S and N
        if constexpr (S_in_regs) {
            load_S();
            load_N();
            cute::cp_async_fence();
        }

        // Initialize position for share memory pointers
        if (DoubleBuffer && q_block % 2 == 1) {
            sQ.data() = sQ.data() + int(size(sQ));
            sQt.data() = sQt.data() + int(size(sQ));
            tQsQ.data() = tQsQ.data() + int(size(sQ));
            tlogGslogG.data() = tlogGslogG.data() + int(size(slogG));
        }

        // start loading Q, log_G, dy and dY
        load_Q();
        load_log_G();
        if constexpr (DoubleBuffer) {
            tQsQ.data() = tQsQ.data() + int((q_block % 2 == 0) ? size(sQ) : -size(sQ));
            tlogGslogG.data() = tlogGslogG.data() + int((q_block % 2 == 0) ? size(slogG) : -size(slogG));
        }
        load_dy();
        load_dY();
        if constexpr (!S_in_regs) {
            load_S();
            load_N();
        }
        cute::cp_async_fence();

        // if S_in_regs, load them into registers
        if constexpr (S_in_regs) {
            power_attention::cp_async_wait<1>();
            __syncthreads();
            Tensor tdPQrS_copy_view = smem_thr_copy_S.retile_D(tdPQrS);
            cute::copy(smem_tiled_copy_S, tdPQsS, tdPQrS_copy_view);
            if constexpr (rescale) {
                power_attention::elementwise_product(tdPQrS, params.multiplier, tdPQrS);
            }
        }

        // Main loop
        const auto info = power_attention::binfo(did, OuterBlock, InnerBlock);
        const auto inner_bid = std::get<0>(info);
        const auto outer_bid = std::get<1>(info);
        const auto is_on_diagonal = std::get<2>(info);

        for (; q_block >= 0; --q_block)
        {
            // start loading next Q and logG, if double buffer is enabled
            if (DoubleBuffer && q_block > 0) {
                load_Q();
                tQsQ.data() = tQsQ.data() + int((q_block % 2 == 1) ? size(sQ) : -size(sQ));
                load_log_G();
                tlogGslogG.data() = tlogGslogG.data() + int((q_block % 2 == 1) ? size(slogG) : -size(slogG));
            }
            cute::cp_async_fence();

            // wait for Q, logG, dY, and dy
            if constexpr (DoubleBuffer) {
                power_attention::cp_async_wait<1>();
            } else {
                power_attention::cp_async_wait<0>();
            }
            __syncthreads();

#ifdef QUERY_STATE_BWD_DEBUG
            if (DEBUGGER_THREAD_QSBWD) {
                printf("q_block: %d, sQ: \n", q_block);
                print_tensor(sQ);
                printf("q_block %d, tdPQrS: \n", q_block);
                print_tensor(tdPQrS);
                printf("q_block %d, sS: \n", q_block);
                print_tensor(sS);
                printf("q_block %d, sN: \n", q_block);
                print_tensor(sN);
                printf("q_block %d, slogG: \n", q_block);
                print_tensor(slogG);
                printf("q_block %d, sdY: \n", q_block);
                print_tensor(sdY);
                printf("q_block %d, sdy: \n", q_block);
                print_tensor(sdy);
                printf("\n");
            }
#endif

            // expand states, also do ϕ(Q)^T @ dy -> dN
            auto tdSrPQt = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&]() {
                return sympow.expandState<INNER_BID>(sQt, sPQt, sdy, slogG, tiled_mma, acc_dn, outer_bid, is_on_diagonal, params);
            });

            if constexpr (DoubleBuffer) {
                sQt.data() = sQt.data() + int((q_block % 2 == 0) ? size(sQ) : -size(sQ));
                slogG.data() = slogG.data() + int((q_block % 2 == 0) ? size(slogG) : -size(slogG));
            }

            // if (params.return_phi) {
            //     Tensor tdSsPQt = thr_mma.partition_A(sPQt);
            //     cute::copy(tdSrPQt, tdSsPQt);
            //     __syncthreads();
            // }

            // ϕ(Q)^T @ dY -> dS
            power_attention::gemm_rs<rescale>(acc_ds, tdSrPQt, tdSrdYt, tdSsdYt, tiled_mma, smem_tiled_copy_dYt, smem_thr_copy_dYt, params.multiplier);

#ifdef QUERY_STATE_BWD_DEBUG
            if (DEBUGGER_THREAD_QSBWD) {
                printf("tid: %d, tdPQsdY: \n", tid);
                print_tensor(tdPQsdY);
                printf("\n");
            }
            __syncthreads();
#endif


            // dY @ S^T -> acc_dpq
            // (BlockQ x Headdim) @ (Headdim x BlockD) -> (BlockQ x BlockD)
            Tensor acc_dpq = partition_fragment_C(tiled_mma_trans, Shape<Int<BlockQ>, Int<BlockD>>{}); // ((2, 2), TILE_Q, TILED_D)
            clear(acc_dpq);
            Tensor tdPQrdY = thr_mma_trans.partition_fragment_A(sdY);
            power_attention::gemm</*A_in_regs=*/false, /*B_in_regs=*/S_in_regs, /*rescale_B=*/!S_in_regs, /*rescale_A=*/rescale>(acc_dpq, tdPQrdY, tdPQrS, tdPQsdY, tdPQsS, tiled_mma_trans, smem_tiled_copy_dY, smem_tiled_copy_S, smem_thr_copy_dY, smem_thr_copy_S, params.multiplier, params.multiplier);

#ifdef QUERY_STATE_BWD_DEBUG
            if (DEBUGGER_THREAD_QSBWD) {
                printf("q_block %d, acc_dpq before add_dyN: \n", q_block);
                print_tensor(acc_dpq);
                printf("\n");
            }
#endif

            // dy @ N^T -> acc_dpq
            sympow.add_dyN_trans(sdy, sN, acc_dpq, params.multiplier_squared);
            __syncthreads();

#ifdef QUERY_STATE_BWD_DEBUG
            if (DEBUGGER_THREAD_QSBWD) {
                printf("q_block %d, acc_dpq: \n", q_block);
                print_tensor(acc_dpq);
                printf("\n");
            }
            Tensor tdPQsdPQ = thr_mma_trans.partition_C(sPQ);
            cute::copy(acc_dpq, tdPQsdPQ);
            __syncthreads();
            if (DEBUGGER_THREAD_QSBWD) {
                printf("q_block %d, sdPQ: \n", q_block);
                print_tensor(sPQ);
                printf("\n");
                printf("q_block %d, outer_bid: %d\n", q_block, outer_bid);
            }
            __syncthreads();
#endif

            // load next dY, dy
            if (q_block > 0) {
                load_dY();
                load_dy();
                cute::cp_async_fence();
            }

            // backprop to dQ, or d(Q*exp(log_G/deg)) if gating is enabled
            // this handles atomicAdd to gmem
            sympow.gradQ(sQ, gdQaccum, sdQaccum, acc_dpq, tiled_mma_trans, inner_bid, outer_bid, is_on_diagonal);
            gdQaccum.data() = gdQaccum.data() + index_t(-BlockQ * params.seq_stride);
            if constexpr (DoubleBuffer) {
                sQ.data() = sQ.data() + int((q_block % 2 == 0) ? size(sQ) : -size(sQ));
            }

            if (params.return_phi) {
                save_Phi();
                __syncthreads();
            }
            __syncthreads();
        }

        // Epilogue
        // First reduce acc_dn within warps
        __syncthreads();
        power_attention::SumOp<ElementAccum> sum_op;
        CUTE_UNROLL
        for (int i = 0; i < size(acc_dn); i++) {
            acc_dn[i] = power_attention::Allreduce<4>::run(acc_dn[i], sum_op);
        }

        // write back to shared memory first
        constexpr static int TileShape_M = decltype(typename Kernel_traits::TiledMma{}.template tile_size_mnk<0>())::value;
        constexpr static int AtomShape_M = decltype(size<0>(typename Kernel_traits::TiledMma::AtomShape_MNK{}))::value;
        if (tid % 4 == 0) {
            for (int i = 0; i < size(acc_dn); i++) {
                int idx = (i / 2) * TileShape_M + (tid / 32) * AtomShape_M + (i % 2) * 8 + (tid % 32) / 4;
                if constexpr (std::is_same_v<Element, half_t>) {
                    sdN[idx] = acc_dn[i] * params.multiplier;
                } else {
                    sdN[idx] = acc_dn[i];
                }
            }
        }
        __syncthreads();

#ifdef QUERY_STATE_BWD_DEBUG
        if (DEBUGGER_THREAD_QSBWD) {
            printf("q_block %d, sdN: \n", q_block);
            print_tensor(sdN);
            printf("\n");
        }
#endif

        // Write back to global memory
        typename Kernel_traits::GmemCopyTileN gmem_tiled_copy_dN;
        auto gmem_thr_copy_dN = gmem_tiled_copy_dN.get_thread_slice(tid);
        Tensor tdNsdN = gmem_thr_copy_dN.partition_S(sdN);
        Tensor tdNgdN = gmem_thr_copy_dN.partition_D(gdN);
        Tensor cdN = make_identity_tensor(make_shape(size(gdN)));
        Tensor tdNcdN = gmem_thr_copy_dN.partition_S(cdN);

        power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dN, tdNsdN, tdNgdN, tdNcdN, std::min(BlockD, paddedExpandedDim - did * BlockD));

        // copy acc_ds back to shared memory first
        Tensor rdS = power_attention::convert_type<Element>(acc_ds);
        Tensor cdS = make_identity_tensor(make_shape(size<0>(gdS), size<1>(gdS)));
        auto smem_tiled_copy_dS = make_tiled_copy_C(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
        auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tid);
        Tensor tdScsdS = smem_thr_copy_dS.partition_S(cdS);
        Tensor taccdSrdS = smem_thr_copy_dS.retile_S(rdS);
        Tensor taccdSsdS = smem_thr_copy_dS.partition_D(sdS);

        power_attention::copy</*Is_even_MN=*/true>(smem_tiled_copy_dS, taccdSrdS, taccdSsdS, tdScsdS, BlockD);
        __syncthreads();


#ifdef QUERY_STATE_BWD_DEBUG
        if (DEBUGGER_THREAD_QSBWD) {
            printf("q_block %d, sdS: \n", q_block);
            print_tensor(sdS);
            printf("\n");
        }
#endif

        // copy back to global memory
        typename Kernel_traits::GmemCopyTiledS gmem_tiled_copy_dS;
        auto gmem_thr_copy_dS = gmem_tiled_copy_dS.get_thread_slice(tid);
        Tensor tdScdS = gmem_thr_copy_dS.partition_S(cdS);

        Tensor tdSsdS = gmem_thr_copy_dS.partition_S(sdS);
        Tensor tdSgdS = gmem_thr_copy_dS.partition_D(gdS);

        power_attention::copy</*Is_even_MN=*/false>(gmem_tiled_copy_dS, tdSsdS, tdSgdS, tdScdS, std::min(BlockD, paddedExpandedDim - did * BlockD));
    }

    template <typename Kernel_traits>
    __global__ void __launch_bounds__(Kernel_traits::NWarps *cutlass::NumThreadsPerWarp, 2) query_state_bwd_dSdNdQ(__grid_constant__ const Query_state_bwd_params params)
    {
        constexpr int PaddedExpandedDim = Kernel_traits::PaddedExpandedDim;
        constexpr int BlockD = Kernel_traits::BlockD;
        constexpr int numBlocksD = (PaddedExpandedDim + BlockD - 1) / BlockD;
        const int head_id = blockIdx.y;
        const int batch_id = blockIdx.z / params.num_chunks;
        const int chunk_id = blockIdx.z % params.num_chunks;
        
        for (int d_block = blockIdx.x; d_block < numBlocksD; d_block += gridDim.x) {
            query_state_bwd_dSdNdQ_impl<Kernel_traits>(params, d_block, head_id, batch_id, chunk_id);
        }
    }

} // namespace power_attention
