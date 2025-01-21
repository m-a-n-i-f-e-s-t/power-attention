#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "state.h"
#include "kernel_traits.h"
#include "utils.h"
#include "static_switch.h"
#include "power_utils.h"

// #define SYMPOW_DEBUG_CSFWD 1
// #define SYMPOW_DEBUG_CSBWD 1
// #define SYMPOW_DEBUG_QSFWD 1
// #define SYMPOW_DEBUG_QSBWD 1
#define DEBUGGER_THREAD (cute::thread(0, 0))

template <int DIM, typename Tensor, typename T>
__forceinline__ __device__ void static_add(Tensor &arr, T &&val)
{
    arr[DIM] += std::forward<T>(val);
}

namespace power_attention
{
    using namespace cute;

    // Helper to provide default value
    template <typename T>
    struct DefaultValue;

    template <>
    struct DefaultValue<std::nullptr_t>
    {
        static constexpr std::nullptr_t value = nullptr;
    };

    template <>
    struct DefaultValue<float>
    {
        static constexpr float value = 0.0f;
    };

    /**
     * @brief Cache caching arbitrary tensor values where the 
     * key indicates the start of a range of integer keys, whose 
     * corresponding values are cached.
     * 
     * 
     * @tparam ValTensorType 
     */
    template <typename ValTensorType, int CacheSize>
    struct RangeCache
    {
        using ValType = typename ValTensorType::value_type;
        using ValLayout = typename ValTensorType::layout_type;
        using KeyLayout = Layout<Shape<Int<CacheSize>>>;
        using CacheLayout = decltype(make_layout(KeyLayout{}, ValLayout{}));
        using CacheTensor = decltype(make_tensor<ValType>(CacheLayout{}));
        CacheTensor cache;
        int start;

        __forceinline__ __device__ RangeCache() : cache(make_tensor<ValType>(CacheLayout{})) {}

        __forceinline__ __device__ bool has(const int key)
        {
            return start <= key && key < start + CacheSize;
        }

        __forceinline__ __device__ auto get(const int key)
        {
            return BINARY_DIM_SWITCH(key - start, KEY, CacheSize, [&]() {
                return cache(KEY, _);
            });
            // return make_tensor<ValType>(ValLayout{});
        }

        template <typename SrcTensor, typename CopyAtom>
        __forceinline__ __device__ void put(const int key, const SrcTensor &src, CopyAtom)
        {
            start = (key / CacheSize) * CacheSize;
            cute::copy(CopyAtom{}, src, cache);
        }
    };


    /**
     * @brief Given block index, return the corresponding inner and outer block index, and whether to mask
     * 
     * The following layout is for when OuterBlock == InnerBlock
     * 
     * |+-------+----+----+----+----+----+----+----+
     * | Layout |           Outer Blocks           |
     * +--------+----+----+----+----+----+----+----+
     * |        | 0  |    |    |    |    |    |    |
     * |        | 1  | 2  |    |    |    |    |    |
     * | Inner  | 3  | 4  | 5  |    |    |    |    |
     * | Blocks | 6  | 7  | 8  | 9  |    |    |    |
     * |        | 10 | 11 | 12 | 13 | 14 |    |    |
     * |        | 15 | 16 | 17 | 18 | 19 | 20 |    |
     * |        | 21 | 22 | 23 | 24 | 25 | 26 | 27 |
     * +--------+----+----+----+----+----+----+----+
     * 
     * The following layout is for when InnerBlock == 2 * OuterBlock
     * 
     * +--------+----+----+----+----+----+----+----+----+
     * | Layout |            Outer Blocks               |
     * +--------+----+----+----+----+----+----+----+----+
     * |        | 0  | 1  |    |    |    |    |    |    |
     * | Inner  | 2  | 3  | 4  | 5  |    |    |    |    |
     * | Block  | 6  | 7  | 8  | 9  | 10 | 11 |    |    |
     * |        | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 |
     * +--------+----+----+----+----+----+----+----+----+
     * 
     * @return tuple of (inner_bid, outer_bid, need_mask)
     */
    __forceinline__ __device__ auto binfo(int did, int OuterBlock, int InnerBlock) {
        int inner_bid = 0; 
        int blocks_per_row = (InnerBlock / OuterBlock);
        while (did >= blocks_per_row) {
            did -= blocks_per_row;
            blocks_per_row += (InnerBlock / OuterBlock);
            inner_bid++;
        }
        return std::make_tuple(inner_bid, did, ((did + 1) * OuterBlock) > (inner_bid * InnerBlock));
    };

    /**
     * @brief An incremental version of binfo
     */
    template <int OuterBlock, int InnerBlock>
    __forceinline__ __device__ auto binfo_bump(int &inner_bid, int &outer_bid) {
        constexpr static int block_ratio = InnerBlock / OuterBlock;
        outer_bid += 1;
        if (outer_bid == (inner_bid + 1) * block_ratio) {
            outer_bid = 0;
            inner_bid += 1;
        }
    }

    /**
     * @brief return mask multiplier for the given inner and outer block id
     */
    template <int OuterBlock, int InnerBlock>
    __forceinline__ __device__ auto state_multiplier(int inner_bid, int outer_bid) {
        constexpr static int block_ratio = InnerBlock / OuterBlock;
        return (outer_bid + 1) > inner_bid * block_ratio ? 1.0f : 2.0f;
    }

    /**
     * @brief check if a specific block is on the diagonal
     */
    template <int OuterBlock, int InnerBlock>
    __forceinline__ __device__ auto on_diagonal(int inner_bid, int outer_bid) {
        constexpr static int block_ratio = InnerBlock / OuterBlock;
        return (outer_bid + 1) > inner_bid * block_ratio;
    }

    /**
     * @brief Similar to binfo, but for the transpose case. The transposed traversal is
     * helpful in reducing the number of shuffles during query_state_bwd_dq.
     * 
     * +--------+----+----+----+----+----+----+----+----+
     * | Layout |            Outer Blocks               |
     * +--------+----+----+----+----+----+----+----+----+
     * |        | 0  | 4  |    |    |    |    |    |    |
     * | Inner  | 1  | 5  | 8  | 11 |    |    |    |    |
     * | Block  | 2  | 6  | 9  | 12 | 14 | 16 |    |    |
     * |        | 3  | 7  | 10 | 13 | 15 | 17 | 18 | 19 |
     * +--------+----+----+----+----+----+----+----+----+
     * 
     * @return tuple of (inner_bid, outer_bid, need_mask)
     */
    __forceinline__ __device__ auto binfo_t(int did, int OuterBlock, int InnerBlock, int Headdim) {
        const int blocks_per_row = InnerBlock / OuterBlock;
        int outer_bid = 0;
        int blocks_per_col = Headdim / OuterBlock;
        int cols = 0;
        while (did >= blocks_per_col) {
            did -= blocks_per_col;
            outer_bid++;
            cols++;
            if (cols == blocks_per_row) {
                cols = 0;
                blocks_per_col -= 1;
            }
        }
        const int inner_bid = did + outer_bid * OuterBlock / InnerBlock;
        return std::make_tuple(inner_bid, outer_bid, (outer_bid + 1) * OuterBlock > inner_bid * InnerBlock);
    }


    template <typename Kernel_traits>
    struct SymowStateExpanderBlockP2 {
        using multiindex_t = typename Kernel_traits::multiindex_t;
        using C_type = typename Kernel_traits::C_type;
        using Element = typename Kernel_traits::Element;
        using ElementAccum = typename Kernel_traits::ElementAccum;
        using TiledMma = typename Kernel_traits::TiledMma;
        using MMA_ATOM = typename Kernel_traits::MMA_ATOM;

        constexpr static int Deg = Kernel_traits::Deg;
        constexpr static int Headdim = Kernel_traits::Headdim;
        constexpr static int BlockD = Kernel_traits::BlockD;
        constexpr static int BlockT = Kernel_traits::BlockT;
        constexpr static int NThreads = Kernel_traits::NThreads;
        constexpr static int NWarps = Kernel_traits::NWarps;
        constexpr static int TileShape_M = decltype(TiledMma{}.template tile_size_mnk<0>())::value;
        constexpr static int TileShape_K = decltype(TiledMma{}.template tile_size_mnk<2>())::value;
        constexpr static int AtomShape_M = decltype(size<0>(typename TiledMma::AtomShape_MNK{}))::value;
        constexpr static int AtomShape_K = decltype(size<2>(typename TiledMma::AtomShape_MNK{}))::value;
        constexpr static int InnerBlock = Kernel_traits::InnerBlock;
        constexpr static int OuterBlock = Kernel_traits::OuterBlock;
        constexpr static int numInnerBlocks = BlockD / InnerBlock;

        using LDMATRIX_TRANS = Copy_Atom<SM75_U16x8_LDSM_T, Element>;
        using LDMATRIX = Copy_Atom<SM75_U32x4_LDSM_N, Element>;

        template <int BID, typename TensorSKt>
        __forceinline__ __device__ auto get_inner_block_t(TensorSKt sKt) {
            auto l = sKt.layout();
            auto block_layout = make_layout(Shape<Shape<Int<InnerBlock>, Int<numInnerBlocks>>, Int<BlockT>>{}, Stride<Stride<_1, _0>, Int<Headdim>>{});
            return make_tensor(sKt.data(), composition(l, Int<BID * InnerBlock>{}, block_layout));
        }

        template <typename TensorSQKt>
        __forceinline__ __device__ auto read_row(TensorSQKt sQKt, const int bid, const int wid, const int tid) {
            Tensor rQKt_row = make_tensor<Element>(Shape<_2, _2, Int<BlockT/16>, Int<OuterBlock/NWarps>>{});

            CUTE_UNROLL
            for (int r = 0; r < OuterBlock / NWarps; r++) {
                CUTE_UNROLL
                for (int t = 0; t < BlockT / 16; t++) {
                    CUTE_UNROLL
                    for (int i = 0; i < 2; i++) {
                        CUTE_UNROLL
                        for (int j = 0; j < 2; j++) {
                            rQKt_row(i, j, t, r) = sQKt(bid * OuterBlock + r * NWarps + wid, i + (tid % 4) * 2 + j * 8 + t * 16);
                        }
                    }
                }
            }

            return rQKt_row;
        }

        /**
         * @brief Read a column of Q matrix, determined by outer block id only
         */
        template <typename TensorSQ>
        __forceinline__ __device__ auto read_outer(TensorSQ sQ, const int bid, const int tid) {
            Tensor rQ_col = make_tensor<Element>(Shape<_2, Int<BlockT / NWarps / 16>>{});
            constexpr static int stride = NWarps * 16;
            const int lane_id = tid % 32;
            const int warp_id = tid / 32;
            CUTE_UNROLL
            for (int t = 0; t < BlockT / stride; t++) {
                CUTE_UNROLL
                for (int i = 0; i < 2; i++) {
                    rQ_col(i, t) = sQ(i * 8 + lane_id / 4 + warp_id * 16 + t * stride, bid * OuterBlock);
                }
            }
            return rQ_col;
        }

        /**
         * @brief Read a column of Q matrix, determined by outer block id and warp id
         */
        template <typename TensorSQ>
        __forceinline__ __device__ auto read_outer(TensorSQ sQ, const int bid, const int wid, const int tid) {
            constexpr int N_Elements = BlockT / 16 * 2;
            Tensor rQ_col = make_tensor<Element>(Shape<Int<N_Elements>, Int<OuterBlock / NWarps>>{});

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("read_outer: bid: %d, wid: %d, tid: %d, outer_dim: %d\n", bid, wid, tid, bid * OuterBlock + wid);
            }
#endif  

            if constexpr (OuterBlock / NWarps > 1) {
                CUTE_UNROLL
                for (int r = 0; r < OuterBlock / NWarps; r++) {
                    CUTE_UNROLL
                    for (int i = 0; i < N_Elements; i++) {
                        rQ_col(i, r) = sQ(i * 8 + (tid % 32) / 4, bid * OuterBlock + wid + r * NWarps);
                    }
                }
            } else {
                CUTE_UNROLL
                for (int i = 0; i < N_Elements; i++) {
                    rQ_col(i, 0) = sQ(i * 8 + (tid % 32) / 4, bid * OuterBlock + wid);
                }
            }
            return rQ_col;
        }

        // state expansion for update_state_fwd
        template <int INNER_BID, typename TensorSKt, typename TensorSPKt, typename TiledMma>
        __forceinline__ __device__ auto expandState(TensorSKt sKt, TensorSPKt sPKt, TiledMma tiled_mma,  const int outer_bid, const bool is_on_diagonal) {
            const int tid = threadIdx.x;
            const int wid = tid / 32;

            // Steps:
            // 1. All warps read the same block of Kt matrix (inner block)
            // 2. Each warp reads a row of Kt, determined by outer block id and warp id
            // 3. Each warp expands on their own

            Tensor sKt_inner = get_inner_block_t<INNER_BID>(sKt); // (BlockD, BlockT)

            auto smem_tiled_copy_Kt_inner = make_tiled_copy_A(LDMATRIX_TRANS{}, tiled_mma);
            auto smem_thr_copy_Kt_inner = smem_tiled_copy_Kt_inner.get_thread_slice(tid);
            Tensor tKtsKt_inner = smem_thr_copy_Kt_inner.partition_S(sKt_inner);

            auto thr_mma = tiled_mma.get_thread_slice(tid);
            Tensor tSrPKt = thr_mma.partition_fragment_A(sPKt); // ((2, 2, 2), TILE_M, TILE_K)
            Tensor tSrPKt_copy_view = smem_thr_copy_Kt_inner.retile_D(tSrPKt); // ((8, 1), TILE_M, TILE_K)
            Tensor tSrPKt_rowcol = make_tensor(tSrPKt.data(), convert_layout_rA_rowcol(tSrPKt.layout()));

            // read inner block
            cute::copy(smem_tiled_copy_Kt_inner, tKtsKt_inner, tSrPKt_copy_view);
            // read outer row
            Tensor rKt_row = read_row(sKt, outer_bid, wid, tid); // ((2, 2, OuterBlock/NWarps), TILE_K)
            __syncthreads();

#ifdef SYMPOW_DEBUG_CSFWD
            if (DEBUGGER_THREAD) {
                printf("outer_bid: %d, inner_bid: %d, is_on_diagonal: %d, rKt_row: \n", outer_bid, INNER_BID, is_on_diagonal);
                print_tensor(rKt_row);
                printf("\n");
                printf("tSrPKt: \n");
                print_tensor(tSrPKt);
                printf("\n");
                printf("tSrPKt_rowcol: \n");
                print_tensor(tSrPKt_rowcol);
                printf("\n");
            }
            __syncthreads();
#endif

            // mask and expand
            if (is_on_diagonal) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tSrPKt_rowcol); n++) {
                    CUTE_UNROLL
                    for (int m = 0; m < size<0>(tSrPKt_rowcol); m++) {
                        auto tmp = fp32_mul(tSrPKt_rowcol(m, n), rKt_row[n]);
                        tSrPKt_rowcol(m, n) = static_cast<Element>(tmp);
                    }
                }
            } else {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tSrPKt_rowcol); n++) {
                    auto tmp0 = fp32_mul(rKt_row[n], 2.0f);
                    CUTE_UNROLL
                    for (int m = 0; m < size<0>(tSrPKt_rowcol); m++) {
                        auto tmp = fp32_mul(tSrPKt_rowcol(m, n), tmp0);
                        tSrPKt_rowcol(m, n) = static_cast<Element>(tmp);
                    }
                }
            }
            return tSrPKt;
        }

        template <int INNER_BID, typename TensorSQ, typename TiledMma>
        __forceinline__ __device__ auto read_inner(TensorSQ sQ, TiledMma tiled_mma, const int tid) {
            auto smem_tiled_copy_Q = make_tiled_copy_A(LDMATRIX{}, tiled_mma);
            auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
            Tensor tQsQ = smem_thr_copy_Q.partition_S(sQ); // ((8), TILE_Q, TILE_HEADDIM)
            Tensor tQsQ_inner_ = tQsQ(_, _, INNER_BID); // ((8), TILE_Q)
            auto l = tQsQ_inner_.layout();
            Tensor tQsQ_inner = make_tensor(tQsQ_inner_.data(), make_layout(get<0>(l), get<1>(l), Layout<Shape<_1>>{}));

            auto thr_mma = tiled_mma.get_thread_slice(tid);
            // simply for shape info
            Tensor sQinner = make_tensor(sQ.data(), Layout<Shape<Int<BlockT>, Int<InnerBlock>>>{});
            Tensor tSrPQ = thr_mma.partition_fragment_A(sQinner); // ((2, 2, 2), TILE_M, TILE_K)
            Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrPQ); // ((8, 1), TILE_M, TILE_K)

            cute::copy(smem_tiled_copy_Q, tQsQ_inner, tSrQ_copy_view);

            return tSrPQ;
        }

        // state expansion for update_state_bwd
        template <typename TensorSK, typename TiledMma>
        __forceinline__ __device__ auto expandState(TensorSK sK, TiledMma tiled_mma, const int inner_bid, const int outer_bid) {
            const int tid = threadIdx.x;
            const bool is_on_diagonal = on_diagonal<OuterBlock, InnerBlock>(inner_bid, outer_bid);
            static_assert(OuterBlock == 1, "OuterBlock must be 1 for update_state_bwd");
            static_assert(InnerBlock == BlockD, "InnerBlock must be BlockD for update_state_bwd");

            // Steps:
            // 1. All warps read the same block of K matrix (inner block)
            // 2. Each warp reads a col of K, determined by outer block id and warp id
            // 3. Each warp expands on their own

            // read inner block
            // collectively this is a BlockT x InnerBlock tensor spreaded across all threads
            // each thread handles ((2, 2, 2), Tile_T, Tile_dim)
            Tensor tdSrPK = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&](){
                return read_inner<INNER_BID>(sK, tiled_mma, tid); // ((2, 2, 2), Tile_T, Tile_dim)
            });
            Tensor tdSrPK_rowcol = make_tensor(tdSrPK.data(), convert_layout_rA_rowcol(tdSrPK.layout()));
            // read outer col
            // collectively this is a BlockT x OuterBlock tensor spreaded across all threads
            // each thread handles ((2), Tile_T)
            Tensor rK_col = read_outer(sK, outer_bid, tid); // ((2), TILE_M)

#ifdef SYMPOW_DEBUG_CSBWD
            if (DEBUGGER_THREAD) {
                printf("outer_bid: %d, inner_bid: %d, tdSrPK: \n", outer_bid, inner_bid);
                print_tensor(tdSrPK);
                printf("\n");
                printf("outer_bid: %d, inner_bid: %d, rK_col: \n", outer_bid, inner_bid);
                print_tensor(rK_col);
                printf("\n");
            }
#endif

            // mask and expand
            if (is_on_diagonal) {
                CUTE_UNROLL
                for (int m = 0; m < size<0>(tdSrPK_rowcol); m++) {
                    CUTE_UNROLL
                    for (int n = 0; n < size<1>(tdSrPK_rowcol); n++) {
                        tdSrPK_rowcol(m, n) = static_cast<Element>(fp32_mul(tdSrPK_rowcol(m, n), rK_col[m]));
                    }
                }
            } else {
                CUTE_UNROLL
                for (int m = 0; m < size<0>(tdSrPK_rowcol); m++) {
                    auto tmp = fp32_mul(rK_col[m], 2.0f);
                    CUTE_UNROLL
                    for (int n = 0; n < size<1>(tdSrPK_rowcol); n++) {
                        tdSrPK_rowcol(m, n) = static_cast<Element>(fp32_mul(tdSrPK_rowcol(m, n), tmp));
                    }
                }
            }

            return tdSrPK;
        }

        // state expansion for query_state_fwd
        template <typename TensorSQ, typename TensorSQP, typename TensorSLogG, typename TiledMma, typename Params>
        __forceinline__ __device__ auto expandState(TensorSQ sQ, TensorSQP sPQ, TensorSLogG sLogG, TiledMma tiled_mma, const int inner_bid, const int outer_bid, Params &params) {
            const int tid = threadIdx.x;

            // Steps:
            // 1. All warps read the same block of Q matrix (inner block)
            // 2. Each warp reads a col of Q, determined by outer block id and warp id
            // 3. Each warp expands on their own

            // read inner block
            Tensor tSrPQ = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&](){
                return read_inner<INNER_BID>(sQ, tiled_mma, tid); // ((2, 2, 2), 1, 1)
            });
            Tensor tSrPQ_rowcol = make_tensor(tSrPQ.data(), convert_layout_rA_rowcol(tSrPQ.layout()));
            // read outer col
            Tensor rQ_col = read_outer(sQ, outer_bid, tid); // ((2), TILE_M)
            // read sLogG
            Tensor rLogG_col = make_tensor<ElementAccum>(get<0>(tSrPQ_rowcol.layout()).shape());

            if (params.gating) {
                if constexpr (size(rLogG_col) > 2) {
                    CUTE_UNROLL
                    for (int i = 0; i < size(rLogG_col); i++) {
                        rLogG_col(i) = sLogG((i / 2) * NWarps * 16 + (i % 2) * 8 + (tid % 32) / 4 + (tid / 32) * 16);
                    }
                }
                else {
                    CUTE_UNROLL
                    for (int i = 0; i < size(rLogG_col); i++) {
                        rLogG_col(i) = sLogG(i * 8 + (tid % 32) / 4 + (tid / 32) * 16);
                    }
                }
            }

            // expand

#ifdef SYMPOW_DEBUG_QSFWD
            if (DEBUGGER_THREAD) {
                printf("tSrPQ: \n");
                print_tensor(tSrPQ);
                printf("\n");
                printf("rQ_col: \n");
                print_tensor(rQ_col);
                printf("\n");
                printf("rN_row: \n");
                print_tensor(rN_row);
                printf("\n");
                printf("rLogG_col: \n");
                print_tensor(rLogG_col);
                printf("\n");
                printf("multiplier: %f\n", params.multiplier);
                printf("std::is_same_v<Element, cutlass::half_t> : %d\n", std::is_same_v<Element, cutlass::half_t>);
            }
#endif

            if (params.use_multiplier) {
                CUTE_UNROLL
                for (int m = 0; m < size<0>(tSrPQ_rowcol); m++) {
                    ElementAccum tmp0 = fp32_mul(rQ_col[m], params.multiplier);
                    CUTE_UNROLL
                    for (int n = 0; n < size<1>(tSrPQ_rowcol); n++) {
                        tSrPQ_rowcol(m, n) = static_cast<Element>(fp32_mul(tSrPQ_rowcol(m, n), tmp0));
                    }
                }
            } else {
                CUTE_UNROLL
                for (int m = 0; m < size<0>(tSrPQ_rowcol); m++) {
                    CUTE_UNROLL
                    for (int n = 0; n < size<1>(tSrPQ_rowcol); n++) {
                        tSrPQ_rowcol(m, n) = static_cast<Element>(fp32_mul(tSrPQ_rowcol(m, n), rQ_col[m]));
                    }
                }
            }
            return tSrPQ;
        }

        template <typename TensorSdy, typename RowColLayout>
        __forceinline__ __device__ auto read_sdy(TensorSdy sdy, const int tid, RowColLayout) {
            Tensor rdy_row = make_tensor<ElementAccum>(get<1>(RowColLayout{}).shape()); // (2, (2, TILE_Q))
            using AccessType = cutlass::AlignedArray<ElementAccum, 2>;
            using CopyAtom = Copy_Atom<UniversalCopy<AccessType>, ElementAccum>;

            Tensor sdy_ = tiled_divide(sdy, Shape<Int<2>>{});

            CUTE_UNROLL
            for (int i = 0; i < size<1>(rdy_row); i++) {
                cute::copy(CopyAtom{}, sdy_(_, i * 4 + (tid % 4)), rdy_row(_, make_coord(i, 0)));
            }
            return rdy_row;
        }

        // state expansion for query_state_bwd_dsdndq & query_state_bwd_dsdn
        template <int INNER_BID, typename TensorSQt, typename TensorSPQt, typename TensorSRowmax, typename TensorSLogG, typename TiledMma, typename Params>
        __forceinline__ __device__ auto expandState(TensorSQt sQt, TensorSPQt sPQt, TensorSRowmax srowmax, TensorSLogG slogG, TiledMma tiled_mma, const int outer_bid, const bool need_mask, Params &params) {
            const int tid = threadIdx.x;
            const int wid = tid / 32;

            // Steps:
            // 1. All warps read the same block of Qt matrix (inner block)
            // 2. Each warp reads a row of Qt, determined by outer block id and warp id
            // 3. Each warp expands on their own

            Tensor sQt_inner = get_inner_block_t<INNER_BID>(sQt); // (BlockD, BlockT)

            auto smem_tiled_copy_Qt_inner = make_tiled_copy_A(LDMATRIX_TRANS{}, tiled_mma);
            auto smem_thr_copy_Qt_inner = smem_tiled_copy_Qt_inner.get_thread_slice(tid);
            Tensor tQtsQt_inner = smem_thr_copy_Qt_inner.partition_S(sQt_inner);

            auto thr_mma = tiled_mma.get_thread_slice(tid);
            Tensor tdSrPQt = thr_mma.partition_fragment_A(sPQt); // ((2, 2, 2), nTILE_D, nTILE_Q)
            Tensor tdSrPQt_copy_view = smem_thr_copy_Qt_inner.retile_D(tdSrPQt); // ((8, 1), nTILE_D, nTILE_Q)
            Tensor tdSrPQt_rowcol = make_tensor(tdSrPQt.data(), convert_layout_rA_rowcol(tdSrPQt.layout())); // ((2, nTILE_D), (2, 2, nTILE_Q))

            // read inner block
            cute::copy(smem_tiled_copy_Qt_inner, tQtsQt_inner, tdSrPQt_copy_view);
            // read outer row
            Tensor rQt_row = read_row(sQt, outer_bid, wid, tid); // ((2, 2, OuterBlock/NWarps), nTILE_Q)
            // slogG
            Tensor rLogG_row = make_tensor<ElementAccum>(get<1>(tdSrPQt_rowcol.layout()).shape()); // ((2, 2), nTILE_Q)

            if (params.gating) {
                // read slogG form smem
                if constexpr (size(rLogG_row) > 4) {
                    CUTE_UNROLL
                    for (int i = 0; i < size(rLogG_row); i++) {
                        rLogG_row(i) = slogG((i / 4) * 16 + ((i % 4) / 2) * 8 + i + (tid % 4) * 2);
                    }
                } else {
                    CUTE_UNROLL
                    for (int i = 0; i < size(rLogG_row); i++) {
                        rLogG_row(i) = slogG((i / 2) * 8 + i + (tid % 4) * 2);
                    }
                }
            }

            // read rowmax
            using R_ROWMAX = decltype(make_tensor<float>(Shape<Int<size<1>(tdSrPQt_rowcol)>>{}));
            R_ROWMAX r_rowmax;
            if (params.has_rowmax) {
                r_rowmax = power::read_col_summary<size<1>(tdSrPQt_rowcol), TiledMma>(srowmax);

                CUTE_UNROLL
                for (int i = 0; i < size(r_rowmax); i++) {
                    r_rowmax(i) = expf(-r_rowmax(i));
                }

#ifdef SYMPOW_DEBUG_QSBWD
                if (DEBUGGER_THREAD) {
                    printf("sQt_inner: \n");
                    print_tensor(sQt_inner);
                    printf("\n");
                    printf("tdSrPQt: \n");
                    print_tensor(tdSrPQt);
                    printf("\n");
                    printf("rLogG_row: \n");
                    print_tensor(rLogG_row);
                    printf("\n");
                    printf("r_rowmax: \n");
                    print_tensor(r_rowmax);
                    printf("\n");
                    printf("params.multiplier_squared: %f\n", params.multiplier_squared);
                }
#endif

                // expand with rowmax
                CUTE_UNROLL
                for (int n = 0; n < size<1>(tdSrPQt_rowcol); n++) {
                    if (params.multiplier_squared <= r_rowmax[n]) {
                        CUTE_UNROLL
                        for (int m = 0; m < size<0>(tdSrPQt_rowcol); m++) {
                            ElementAccum tmp = fp32_mul(tdSrPQt_rowcol(m, n), rQt_row[n]) * params.multiplier_squared;
                            tdSrPQt_rowcol(m, n) = static_cast<Element>(tmp);
                        }
                    } else { // in this case we need to scale dY, as we did in the forward pass
                        CUTE_UNROLL
                        for (int m = 0; m < size<0>(tdSrPQt_rowcol); m++) {
                            ElementAccum tmp = fp32_mul(tdSrPQt_rowcol(m, n), rQt_row[n]) * r_rowmax[n];
                            tdSrPQt_rowcol(m, n) = static_cast<Element>(tmp);
                        }
                    }
                }
            } else {
                // expand
                if (params.use_multiplier) {
                    CUTE_UNROLL
                    for (int m = 0; m < size<0>(tdSrPQt_rowcol); m++) {
                        CUTE_UNROLL
                        for (int n = 0; n < size<1>(tdSrPQt_rowcol); n++) {
                            tdSrPQt_rowcol(m, n) = static_cast<Element>(fp32_mul(tdSrPQt_rowcol(m, n), rQt_row[n]) * params.multiplier_squared);
                        }
                    }
                } else {
                    CUTE_UNROLL
                    for (int m = 0; m < size<0>(tdSrPQt_rowcol); m++) {
                        CUTE_UNROLL
                        for (int n = 0; n < size<1>(tdSrPQt_rowcol); n++) {
                            tdSrPQt_rowcol(m, n) = static_cast<Element>(fp32_mul(tdSrPQt_rowcol(m, n), rQt_row[n]));
                        }
                    }
                }
            }
            return tdSrPQt;
        }

        template <typename TensorSdy, typename TensorSN, typename ACC_DPQ>
        __forceinline__ __device__ auto add_dyN_trans(TensorSdy sdy, TensorSN sN, ACC_DPQ &acc_dpq, const float stabilizer=1.0f) {
            Tensor dpq_rowcol = make_tensor(acc_dpq.data(), power_attention::convert_layout_acc_rowcol(acc_dpq.layout())); // ((2, 1), (2, 2))
            Tensor rdy = make_tensor<ElementAccum>(get<0>(dpq_rowcol.layout())); // (2, 1)
            Tensor rN = make_tensor<ElementAccum>(make_layout(get<1, 0>(dpq_rowcol.layout()), get<1, 1>(dpq_rowcol.layout()))); // (2, 2)

            CUTE_STATIC_ASSERT_V(rank(rdy) == _2{}, "rdy has rank 2");
            CUTE_STATIC_ASSERT_V(rank(rN) == _2{}, "rN has rank 2");
            // CUTE_STATIC_ASSERT_V(size<1>(rdy) == _1{}, "rdy only has one iter");
            using AccessType = cutlass::AlignedArray<ElementAccum, 2>;
            using CopyAtom = Copy_Atom<UniversalCopy<AccessType>, ElementAccum>;
            
            const int tid = threadIdx.x;
            if constexpr (size<1>(rdy) == _1{}) {
                CUTE_UNROLL
                for (int i = 0; i < size(rdy); i++) {
                    rdy[i] = sdy[i * 8 + (tid % 32) / 4];
                }
            } else {
                CUTE_UNROLL
                for (int j = 0; j < size<1>(rdy); j++) {
                    CUTE_UNROLL
                    for (int i = 0; i < size<0>(rdy); i++) {
                        rdy(i, j) = sdy[i * 8 + (tid % 32) / 4 + j * 16];
                    }
                }
            }


            Tensor sN_ = tiled_divide(sN, Shape<Int<2>>{});
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rN); j++) {
                cute::copy(CopyAtom{}, sN_(_, j * 4 + (tid % 4) + (tid / 32) * 8), rN(_, j));
            }

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("sdy: \n");
                print_tensor(sdy);
                printf("\n");
                printf("sN: \n");
                print_tensor(sN);
                printf("\n");
                printf("rdy: \n");
                print_tensor(rdy);
                printf("\n");
                printf("rN: \n");
                print_tensor(rN);
                printf("\n");
            }
#endif

            CUTE_UNROLL
            for (int m = 0; m < size<0>(dpq_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(dpq_rowcol); n++) {
                    if constexpr (std::is_same_v<Element, cutlass::half_t>) {
                        dpq_rowcol(m, n) += rdy[m] * rN[n] * stabilizer;
                    } else {
                        dpq_rowcol(m, n) += rdy[m] * rN[n];
                    }
                }
            }
        }

        template <typename TensorRdy, typename TensorSN, typename ACC_DPQ, typename Params>
        __forceinline__ __device__ auto add_dyN(TensorRdy &rdy, TensorSN sN, ACC_DPQ &acc_dpq, const Params &params) {
            Tensor dpq_rowcol = make_tensor(acc_dpq.data(), power_attention::convert_layout_acc_rowcol(acc_dpq.layout())); // ((2, 1), (2, 2))
            Tensor rN = make_tensor<ElementAccum>(make_layout(get<1, 0>(dpq_rowcol.layout()), get<1, 1>(dpq_rowcol.layout()))); // (2, 2)

            CUTE_STATIC_ASSERT_V(rank(rdy) == _2{}, "rdy has rank 2");
            CUTE_STATIC_ASSERT_V(rank(rN) == _2{}, "rN has rank 2");
            // CUTE_STATIC_ASSERT_V(size<1>(rdy) == _1{}, "rdy only has one iter");
            using AccessType = cutlass::AlignedArray<ElementAccum, 2>;
            using CopyAtom = Copy_Atom<UniversalCopy<AccessType>, ElementAccum>;
            
            Tensor sN_ = tiled_divide(sN, Shape<Int<2>>{});
            const int tid = threadIdx.x;
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rN); j++) {
                cute::copy(CopyAtom{}, sN_(_, j * 4 + (tid % 4)), rN(_, j));
            }

            CUTE_UNROLL
            for (int m = 0; m < size<0>(dpq_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(dpq_rowcol); n++) {
                    dpq_rowcol(m, n) += rdy[m] * rN[n];
                }
            }
        }

        template <int InnerBlock, typename Tensor>
        __forceinline__ __device__ auto convert_layout_acc_inner_block_rowcol(Tensor &t) {
            auto l = t.layout(); // ((2, 2), Tile_T, Tile_D)
            constexpr static int n_inner_d = InnerBlock / 8;
            auto l1 = make_layout(get<0>(l), get<1>(l), tiled_divide(get<2>(l), Int<n_inner_d>{})); // ((2, 2), Tile_T, (n_inner_d, Tile_D/n_inner_d))
            auto l2 = make_layout(make_layout(get<0, 0>(l1), get<0, 1>(l1), get<2, 0>(l1)), get<1>(l1), Layout<Shape<_1>>{}, get<2, 1>(l1)); // ((2, 2, n_inner_d), Tile_T, 1, Tile_D/n_inner_d)
            auto l3 = convert_layout_rA_rowcol(take<0, 3>(l2));
            return make_tensor(t.data(), make_layout(get<0>(l3), get<1>(l3), get<3>(l2)));
        }


        /**
         * @brief Backpropagate gradient from Phi(Q) to Q, for query_state_bwd_dq and update_state_bwd
         */
        template <bool Adjust_Coefficient, typename TensorSQ, typename ACC_DPQ, typename ACC_DQ, typename TILED_MMA>
        __forceinline__ __device__ auto graddQK(TensorSQ sQ, ACC_DPQ &acc_dpq, ACC_DQ &acc_dq, TILED_MMA tiled_mma, const int inner_bid, const int outer_bid) {
            Tensor acc_dq_block_rowcol = convert_layout_acc_inner_block_rowcol<InnerBlock>(acc_dq); // ((2, TILE_T), (2, 2), TILE_HEADDIM/2)
            Tensor acc_dpq_block_rowcol = convert_layout_acc_inner_block_rowcol<InnerBlock>(acc_dpq); // ((2, TILE_T), (2, 2), TILE_D/2)

            const int tid = threadIdx.x;
            Tensor rQ_col = read_outer(sQ, outer_bid, tid); // ((2), TILE_M)

            // adjust for duplication
            if (Adjust_Coefficient) {
                const bool is_on_diagonal = on_diagonal<OuterBlock, InnerBlock>(inner_bid, outer_bid);
                if (!is_on_diagonal) {
                    CUTE_UNROLL
                    for (int m = 0; m < size<0>(acc_dpq_block_rowcol); m++) {
                        CUTE_UNROLL
                        for (int n = 0; n < size<1>(acc_dpq_block_rowcol); n++) {
                            acc_dpq_block_rowcol(m, n, 0) *= 2.0f;
                        }
                    }
                }
            }

            // backprop dQ inner
            CUTE_UNROLL
            for (int m = 0; m < size<0>(acc_dpq_block_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dpq_block_rowcol); n++) {
                    BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&](){
                        acc_dq_block_rowcol(m, n, INNER_BID) += fp32_mul(acc_dpq_block_rowcol(m, n, 0), rQ_col[m]);
                    });
                }
            }

            // backprop dQ outer
            Tensor rQ_inner = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&](){
                return read_inner<INNER_BID>(sQ, tiled_mma, tid); // ((2, 2, 2), 1, 1)
            });
            Tensor rQ_inner_rowcol = make_tensor(rQ_inner.data(), convert_layout_rA_rowcol(rQ_inner.layout())); // ((2, 1), (2, 2))
            Tensor dQ_outer_tmp = make_tensor<ElementAccum>(get<0>(acc_dq_block_rowcol.layout())); // (2, 1)
            clear(dQ_outer_tmp);

            SumOp<ElementAccum> sum_op;
            CUTE_UNROLL
            for (int m = 0; m < size<0>(acc_dpq_block_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dpq_block_rowcol); n++) {
                    dQ_outer_tmp[m] += fp32_mul(acc_dpq_block_rowcol(m, n, 0), rQ_inner_rowcol(m, n));
                }
                dQ_outer_tmp[m] = Allreduce<4>::run(dQ_outer_tmp[m], sum_op);
            }

            // add dQ_outer to acc_dq
            Tensor acc_dq_rowcol = make_tensor(acc_dq.data(), convert_layout_acc_rowcol(acc_dq.layout())); // ((2, 1), (2, TILE_HEADDIM))
            const int outer_dim = OuterBlock == 1 ? outer_bid : outer_bid * OuterBlock + (tid / 32);

            if (tid % 4 == (outer_dim % 8) / 2) {
                const int idx = get_col_idx<Headdim>(outer_dim);
                CUTE_UNROLL
                for (int m = 0; m < size(dQ_outer_tmp); m++) {
                    BINARY_DIM_SWITCH(idx, ID, Headdim / 4, [&]() {
                        acc_dq_rowcol(m, ID) += dQ_outer_tmp[m];
                    });
                }
            }
        }


        template <typename TensorSLOGG, typename TensorGDLOGG, typename TensorACCDPQ>
        __forceinline__ __device__ auto gradLogG(TensorSLOGG slogG, TensorGDLOGG gdlogG, TensorACCDPQ &acc_dpq) {
            const int tid = threadIdx.x;
            constexpr static int TILE_N_PER_INNER = 2;
            Tensor acc_dpq_block_ = make_tensor(acc_dpq.data(), make_layout(get<0>(acc_dpq.layout()), get<1>(acc_dpq.layout()), tiled_divide(get<2>(acc_dpq.layout()), Int<TILE_N_PER_INNER>{}))); // ((2, 2), TILE_Q, (TILE_N_PER_INNER, n_inner_d))
            Tensor dpq_block = acc_dpq_block_(_, _, make_coord(_, 0)); // ((2, 2), 1, 2)
            Tensor dpq_block_rowcol = make_tensor(dpq_block.data(), convert_layout_acc_rowcol(dpq_block.layout())); // ((2, 1), (2, 2))
            Tensor pq_block_rowcol = make_tensor_like(dpq_block_rowcol); // TODO (sean): get the right data
            Tensor rlogG = make_tensor<ElementAccum>(get<0>(convert_layout_acc_rowcol(acc_dpq.layout()))); // (2, 1)
            // read log_G and pre-scale it by deg, this assumes BlockT == 16
            CUTE_UNROLL
            for (int m = 0; m < size(rlogG); m++) {
                rlogG[m] = __expf(slogG[m * 8 + (tid % 32) / 4]);
            }
            Tensor dlogG = make_tensor_like(rlogG); // (2, 1)
            clear(dlogG);
            CUTE_UNROLL
            for (int m = 0; m < size<0>(dpq_block_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(dpq_block_rowcol); n++) {
                    dlogG[m] += fp32_mul(dpq_block_rowcol(m, n), pq_block_rowcol(m, n)) * rlogG[m];
                }
            }
            // intra-warp reduce dlogG
            SumOp<ElementAccum> sum_op;
            CUTE_UNROLL
            for (int i = 0; i < size(dlogG); i++) {
                dlogG[i] = Allreduce<4>::run(dlogG[i], sum_op);
                atomicAdd(&slogG[i * 8 + (tid % 32) / 4], dlogG[i]);
            }

            // // inter-warp reduce dlogG
            // __syncthreads(); // be careful, we are reusing slogG for sdlogG
            // CUTE_UNROLL
            // for (int w = 0; w < NWarps; w++) {
            //     if (tid / 32 == w && (tid % 4) == 0) {
            //         // note: this assumes BlockT == 16
            //         CUTE_UNROLL
            //         for (int i = 0; i < size(dlogG); i++) {
            //             const int offset = i * 8 + (tid % 32) / 4;
            //             slogG[offset] = w == 0 ? dlogG[i] : slogG[offset] + dlogG[i];
            //         }
            //     }
            //     __syncthreads();
            // }

            // // atomic add to global memory
            // auto gmem_tiled_copy_dlogG = make_tiled_copy(
            //     Copy_Atom<DefaultCopy, ElementAccum>{},
            //     Layout<Shape<Int<NThreads>>>{},
            //     Layout<Shape<_1>>{});
            // auto gmem_thr_copy_dlogG = gmem_tiled_copy_dlogG.get_thread_slice(tid);
            // Tensor cdlogG = make_identity_tensor(gdlogG.shape());
            // Tensor tdlogGcdlogG = gmem_thr_copy_dlogG.partition_D(cdlogG);
            // Tensor tdlogGgdlogG = gmem_thr_copy_dlogG.partition_D(gdlogG);
            // Tensor tdlogGslogG = gmem_thr_copy_dlogG.partition_S(slogG);
            // CUTE_UNROLL
            // for (int i = 0; i < size(tdlogGcdlogG); i++) {
            //     if (get<0>(tdlogGcdlogG(0, i)) < BlockT) {
            //         auto tmp = tdlogGslogG(i);
            //         atomicAdd(&tdlogGgdlogG[i], tmp);
            //     }
            // }
        }

        /**
         * @brief convert acc layout from ((2, 2), TILE_Q, TILE_D/H) to ((2, 2), TILE_Q, TILE_N_PER_INNER, n_innner)
         */
        template <typename TensorACC>
        __forceinline__ __device__ auto group_by_inner_block(TensorACC &acc) {
            constexpr static int TILN_SIZE_N = 16;
            constexpr static int TILE_N_PER_INNER = TILN_SIZE_N / 8; // 2
            // static_assert(size<0>(acc) == 4, "the value mode of acc should be 4");

            auto last_l = tiled_divide(get<2>(acc.layout()), Int<TILE_N_PER_INNER>{});
            Tensor acc_block = make_tensor(acc.data(), make_layout(get<0>(acc.layout()), get<1>(acc.layout()), get<0>(last_l), get<1>(last_l))); // ((2, 2), TILE_Q, TILE_N_PER_INNER, n_innner)
            return acc_block;
        }


        /**
         * @brief convert acc layout from ((2, 2), TILE_Q, TILE_D/H) to ((2, TILE_Q), (2, TILE_N_PER_INNER), n_innner)
         */
        template <typename TensorACC>
        __forceinline__ __device__ auto convert_acc_to_inner_rowcol(TensorACC &acc) {
            Tensor acc_block = group_by_inner_block(acc);
            auto per_inner_l = take<0, 3>(acc_block.layout());
            Tensor acc_block_rowcol = make_tensor(acc_block.data(), make_layout(convert_layout_acc_rowcol(per_inner_l), get<3>(acc_block.layout()))); // ((2, TILE_Q), (2, TILE_N_PER_INNER), n_innner)
            return acc_block_rowcol;
        }

        
        /**
         * @brief Given inner_bid, slice an acc tensor from [BlockT, Headdim/BlockD] to
         * an acc tensor from [BlockT, InnerBlock] where the provided inner_bid is used
         * to slice the Headdim/BlockD dimension.
         */
        template <int inner_bid, typename TensorACC>
        __forceinline__ __device__ auto slice_acc_to_inner(TensorACC &acc) {
            Tensor acc_block = group_by_inner_block(acc); // ((2, 2), TILE_Q, TILE_N_PER_INNER, n_innner)
            Tensor inner_block = acc_block(_, _, _, inner_bid); // ((2, 2), 1, 2)
            Tensor inner_block_rowcol = make_tensor(inner_block.data(), convert_layout_acc_rowcol(inner_block.layout())); // ((2, 1), (2, 2))
            return inner_block_rowcol;
        }


        /**
         * @brief Dynamic version of slice_acc_to_inner
         */
        template <typename TensorACC>
        __forceinline__ __device__ auto slice_acc_to_inner(TensorACC &acc, const int inner_bid) {
            // static_assert(size<0>(acc) == 4, "the value mode of acc should be 4");

            Tensor acc_block = group_by_inner_block(acc);
            Tensor inner_block = acc_block(_, _, _, inner_bid); // ((2, 2), 1, 2)
            Tensor inner_block_rowcol = make_tensor(inner_block.data(), convert_layout_acc_rowcol(inner_block.layout())); // ((2, 1), (2, 2))
            return inner_block_rowcol;
        }

        /**
         * @brief Backpropagate gradient from Phi(Q) to Q, for query_state_bwd_dsdndq
         * 
         * This method is only correct when BLOCK_D == InnerBlock
         */
        template <typename TensorSQ, typename TensorGDQ_ACCUM, typename TensorSDQ_ACCUM, typename ACC_DPQ, typename Tile_MMA>
        __forceinline__ __device__ auto gradQ(TensorSQ sQ, TensorGDQ_ACCUM gdQaccum, TensorSDQ_ACCUM sdQaccum, ACC_DPQ &acc_dpq, Tile_MMA tiled_mma, const int inner_bid, const int outer_bid, const bool is_on_diagonal) {
            // this assumes InnerBlock == 16
            // static_assert(size<0>(acc_dpq) == 4 && size<2>(acc_dpq) == 2, "acc_dpq should only cover one Inner_block in the BlockD dimension");
            const int tid = threadIdx.x;
            Tensor dpq_block_rowcol = slice_acc_to_inner<0>(acc_dpq); // ((2, TILE_Q), (2, 2))

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("acc_dpq: \n");
                print_tensor(acc_dpq);
                printf("\n");
            }
#endif


            // adjust for duplication
            if (!is_on_diagonal) {
                CUTE_UNROLL
                for (int m = 0; m < size<0>(dpq_block_rowcol); m++) {
                    CUTE_UNROLL
                    for (int n = 0; n < size<1>(dpq_block_rowcol); n++) {
                        dpq_block_rowcol(m, n) *= 2.0f;
                    }
                }
            }

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("is_on_diagonal: %d\n", is_on_diagonal);
                printf("dpq_block_rowcol: \n");
                print_tensor(dpq_block_rowcol);
                printf("\n");
            }
#endif

            Tensor acc_dq_inner = make_tensor<ElementAccum>(
                Shape<Shape<_2, _2, _2>, Int<BlockT / 16>, _1>{});
            Tensor acc_dq_inner_rowcol = make_tensor(acc_dq_inner.data(), convert_layout_rA_rowcol(acc_dq_inner.layout())); // ((2, BlockT / 16 * NWarps), (2, 2, 1)))
            clear(acc_dq_inner);

            Tensor rQ_outer = read_outer(sQ, outer_bid, tid / 32, tid); // ((2), TILE_M)

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("rQ_outer: \n");
                print_tensor(rQ_outer);
                printf("\n");
            }
#endif

            // mask and backprop dQ_inner
            CUTE_UNROLL
            for (int m = 0; m < size<0>(dpq_block_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(dpq_block_rowcol); n++) {
                    acc_dq_inner_rowcol(m, n) += fp32_mul(dpq_block_rowcol(m, n), rQ_outer[m]);
                }
            }

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("acc_dq_inner_rowcol: \n");
                print_tensor(acc_dq_inner_rowcol);
                printf("\n");
            }
#endif

            // the dQaccum layout is one where each warp will map to the same [BlockT, Headdim] region
            // the following fake tiled mma is used to achieve this
            using MMA_ATOM = typename Tile_MMA::Atom;
            auto fake_tiled_mma = TiledMMA<
                MMA_ATOM,
                Layout<Shape<_1, _1, _1>>,
                Tile<_1, _1, _1>>{};
            auto thr_fake_mma = fake_tiled_mma.get_thread_slice(tid % 32);

            // atomic add to global memory
            Tensor acc_gdq = thr_fake_mma.partition_C(gdQaccum); // ((2, 2), TILE_Q, TILE_Headdim)
            Tensor acc_gdq_inner_rowcol = slice_acc_to_inner(acc_gdq, inner_bid); // ((2, TILE_Q), (2, TILE_N_PER_INNER))
            // Tensor acc_sdq = thr_fake_mma.partition_C(sdQaccum); // ((2, 2), TILE_Q, 2)
            // Tensor acc_sdq_inner_rowcol = slice_acc_to_inner<0>(acc_sdq); // ((2, TILE_Q), (2, TILE_N_PER_INNER))

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("acc_gdq: \n");
                print_tensor(acc_gdq);
                printf("\n");
                printf("acc_gdq_inner_rowcol: \n");
                print_tensor(acc_gdq_inner_rowcol);
                printf("\n");
            }
#endif

            CUTE_UNROLL
            for (int m = 0; m < size<0>(acc_gdq_inner_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_gdq_inner_rowcol); n++) {
                    atomicAdd(&acc_gdq_inner_rowcol(m, n), acc_dq_inner_rowcol(m, n));
                }
            }

            // // clear sdq
            // for (int i = tid; i < size(sdQaccum); i += blockDim.x) {
            //     sdQaccum(i) = 0.0f;
            // }
            // __syncthreads();

            // for (int w = 0; w < NWarps; w++) {
            //     if (tid / 32 == w) {
            //         CUTE_UNROLL
            //         for (int m = 0; m < size<0>(acc_gdq_inner_rowcol); m++) {
            //             CUTE_UNROLL
            //             for (int n = 0; n < size<1>(acc_gdq_inner_rowcol); n++) {
            //                 acc_sdq_inner_rowcol(m, n) = (w == 0) ? acc_dq_inner_rowcol(m, n) : acc_sdq_inner_rowcol(m, n) + acc_dq_inner_rowcol(m, n);
            //             }
            //         }
            //     }
            //     __syncthreads();
            // }

            // // backprop to gdQaccum
            // if (tid / 32 == 0) {
            //     CUTE_UNROLL
            //     for (int m = 0; m < size<0>(acc_sdq_inner_rowcol); m++) {
            //         CUTE_UNROLL
            //         for (int n = 0; n < size<1>(acc_sdq_inner_rowcol); n++) {
            //             atomicAdd(&acc_gdq_inner_rowcol(m, n), acc_sdq_inner_rowcol(m, n));
            //         }
            //     }
            // }


            // mask and backprop dQ_outer
            // read Q_inner
            // here we assume BLOCK_D == InnerBlock, meaning TILE_D == 1
            Tensor rQ_inner = BINARY_DIM_SWITCH(inner_bid, INNER_BID, Headdim/InnerBlock, [&](){
                return read_inner<INNER_BID>(sQ, tiled_mma, tid); // ((2, 2, 2), TILE_Q, TILE_D)
            });
            Tensor acc_dq_outer = make_tensor<ElementAccum>(
                Shape<_2>{});
            clear(acc_dq_outer);

            static_assert(size<2>(rQ_inner.layout()) == 1, "TILE_D must be 1");

            Tensor rQ_inner_rowcol = make_tensor(rQ_inner.data(), convert_layout_rA_rowcol(rQ_inner.layout())); // ((2, 1), (2, 2))

            SumOp<ElementAccum> sum_op;

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("rQ_inner_rowcol: \n");
                print_tensor(rQ_inner_rowcol);
                printf("\n");
            }
#endif

            CUTE_UNROLL
            for (int m = 0; m < size<0>(dpq_block_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(dpq_block_rowcol); n++) {
                    acc_dq_outer[m] += fp32_mul(dpq_block_rowcol(m, n), rQ_inner_rowcol(m, n));
                }
                acc_dq_outer[m] = Allreduce<4>::run(acc_dq_outer[m], sum_op);
            }

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("acc_dq_outer: \n");
                print_tensor(acc_dq_outer);
                printf("\n");
            }
#endif

            // atomic add outer grad to global memory
            Tensor acc_gdq_rowcol = make_tensor(acc_gdq.data(), convert_layout_acc_rowcol(acc_gdq.layout())); // ((2, 1), (2, 2, Headdim/BlockD))
            const int outer_dim = outer_bid * OuterBlock + tid / 32;

#ifdef SYMPOW_DEBUG_QSBWD
            if (DEBUGGER_THREAD) {
                printf("acc_gdq_rowcol: \n");
                print_tensor(acc_gdq_rowcol);
                printf("\n");
                printf("outer_dim: %d\n", outer_dim);
            }
#endif

            if (tid % 4 == 0) {
                const int row_idx_offset = (tid % 32) / 4;
                CUTE_UNROLL
                for (int m = 0; m < size<0>(acc_dq_outer); m++) {
                    atomicAdd(&gdQaccum(m * 8 + row_idx_offset, outer_dim), acc_dq_outer(m));
                }
            }

            // backprop to dlog_G
            // if (gating) {
            //     grad_gating(dpq_block_rowcol, acc_dq_inner_rowcol, acc_dq_outer, rQ_inner, rQ_outer, slogG, gdlogG, inner_bid, outer_bid, tid);
            // }

        };

        /**
         * @brief backprop gradient from d(Q * exp(log_G) / deg) to dlog_G
         * and dQ
         * 
         * dQ_inner: ((2, 2), TILE_Q, 2)
         * acc_dq_rowcol: ((2, TILE_Q), (2, 2, HEADDIM/InnerBlock))
         * rQ_inner: ((2, 2, 2), TILE_Q, 1) contains the inner block of Q
         * rQ_outer: ((2), TILE_Q) contains the outer col of Q
         * slogG: shared memory buffer for logG
         * outer_bid: outer block index
         * 
         * dQ = d(Q * exp(log_G) / deg) * exp(log_G / deg)
         * dlog_G = d(Q * exp(log_G) / deg) * Q * exp(log_G / deg) / deg
         */
        template <typename TensorDPQ_ROWCOL, typename TensorDQ_INNER_ROWCOL, typename TensorACC_DQ_OUTER, typename TensorRQ_INNER, typename TensorRQ_OUTER, typename TensorSLOGG, typename TensorGDLOGG>
        __forceinline__ __device__ auto grad_gating(TensorDPQ_ROWCOL &dpq_rowcol, TensorDQ_INNER_ROWCOL &acc_dq_inner_rowcol, TensorACC_DQ_OUTER &acc_dq_outer, TensorRQ_INNER &rQ_inner, TensorRQ_OUTER &rQ_outer, TensorSLOGG slogG, TensorGDLOGG gdlogG, const int inner_bid,const int outer_bid, const int tid) {
            Tensor rQ_inner_rowcol = make_tensor(rQ_inner.data(), convert_layout_rA_rowcol(rQ_inner.layout())); // ((2, 1), (2, 2))
            Tensor dlogG = make_tensor<ElementAccum>(get<0>(acc_dq_inner_rowcol.layout())); // (2, 1)
            Tensor rlogG = make_tensor<ElementAccum>(get<0>(acc_dq_inner_rowcol.layout())); // (2, 1)
            static_assert(size<0>(acc_dq_inner_rowcol) == size<0>(rQ_inner_rowcol), "acc_dq_inner_rowcol and rQ_inner_rowcol must have the same size in the first dimension");
            static_assert(size<0>(acc_dq_inner_rowcol) == size<0>(rQ_outer), "acc_dq_inner_rowcol and rQ_outer must have the same size in the first dimension");
            clear(rlogG);
            clear(dlogG);
            constexpr static ElementAccum deg = static_cast<ElementAccum>(2.0f);
            const int outer_dim = outer_bid * OuterBlock + tid / 32;

            // read log_G and pre-scale it by deg, this assumes BlockT == 16
            CUTE_UNROLL
            for (int m = 0; m < size(rlogG); m++) {
                rlogG[m] = __expf(slogG[m * 8 + (tid % 32) / 4] / deg);
            }

            // backprop dq_inner to dlog_G
            CUTE_UNROLL
            for (int m = 0; m < size(dlogG); m++) {
                const float tmp = rlogG[m] / deg;
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dq_inner_rowcol); n++) {
                    dlogG[m] += static_cast<ElementAccum>(acc_dq_inner_rowcol(m, n) * rQ_inner_rowcol(m, n))* tmp;
                }
                // backprop dq_outer to dlog_G
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dq_outer); n++) {
                    dlogG[m] += static_cast<ElementAccum>(acc_dq_outer(m, n) * rQ_outer[m]) * tmp;
                }
            }


            // backprop to dq_inner
            CUTE_UNROLL
            for (int m = 0; m < size<0>(acc_dq_inner_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dq_inner_rowcol); n++) {
                    acc_dq_inner_rowcol(m, n) = static_cast<Element>(fp32_mul(rlogG[m], acc_dq_inner_rowcol(m, n)));
                }
                // backprop to dq_outer
                CUTE_UNROLL
                for (int n = 0; n < size<1>(acc_dq_outer); n++) {
                    acc_dq_outer(m, n) *= rlogG[m];
                }
            }


            // intra-warp reduce dlogG
            SumOp<ElementAccum> sum_op;
            CUTE_UNROLL
            for (int i = 0; i < size(dlogG); i++) {
                dlogG[i] = Allreduce<4>::run(dlogG[i], sum_op);
            }

            // inter-warp reduce dlogG
            __syncthreads(); // be careful, we are reusing slogG for sdlogG
            CUTE_UNROLL
            for (int w = 0; w < NWarps; w++) {
                if (tid / 32 == w && (tid % 4) == 0) {
                    // note: this assumes BlockT == 16
                    CUTE_UNROLL
                    for (int i = 0; i < size(dlogG); i++) {
                        const int offset = i * 8 + (tid % 32) / 4;
                        slogG[offset] = w == 0 ? dlogG[i] : slogG[offset] + dlogG[i];
                    }
                }
                __syncthreads();
            }

            // atomic add to global memory
            auto gmem_tiled_copy_dlogG = make_tiled_copy(
                Copy_Atom<DefaultCopy, ElementAccum>{},
                Layout<Shape<Int<NThreads>>>{},
                Layout<Shape<_1>>{});
            auto gmem_thr_copy_dlogG = gmem_tiled_copy_dlogG.get_thread_slice(tid);
            Tensor cdlogG = make_identity_tensor(gdlogG.shape());
            Tensor tdlogGcdlogG = gmem_thr_copy_dlogG.partition_D(cdlogG);
            Tensor tdlogGgdlogG = gmem_thr_copy_dlogG.partition_D(gdlogG);
            Tensor tdlogGslogG = gmem_thr_copy_dlogG.partition_S(slogG);
            CUTE_UNROLL
            for (int i = 0; i < size(tdlogGcdlogG); i++) {
                if (get<0>(tdlogGcdlogG(0, i)) < BlockT) {
                    auto tmp = tdlogGslogG(i);
                    atomicAdd(&tdlogGgdlogG[i], tmp);
                }
            }
        }


        template <typename TensorSds, typename ACC_DPK>
        __forceinline__ __device__ auto add_1ds(TensorSds sds, ACC_DPK &acc_dpk) {
            Tensor dpk_rowcol = make_tensor(acc_dpk.data(), convert_layout_acc_rowcol(acc_dpk.layout())); // ((2, 1), (2, Tile_D))
            Tensor rds = make_tensor<ElementAccum>(get<1>(dpk_rowcol.layout())); // (2, Tile_D)
            Tensor sds_ = tiled_divide(sds, Shape<Int<2>>{});

            using AccessType = cutlass::AlignedArray<ElementAccum, 2>;
            using CopyAtom = Copy_Atom<UniversalCopy<AccessType>, ElementAccum>;

            const int tid = threadIdx.x;
            CUTE_UNROLL
            for (int j = 0; j < size<1>(rds); j++) {
                cute::copy(CopyAtom{}, sds_(_, j * 4 + (tid % 4)), rds(_, j));
            }

            CUTE_UNROLL
            for (int m = 0; m < size<0>(dpk_rowcol); m++) {
                CUTE_UNROLL
                for (int n = 0; n < size<1>(dpk_rowcol); n++) {
                    dpk_rowcol(m, n) += rds[n];
                }
            }
        }

    };


} // namespace power_attention