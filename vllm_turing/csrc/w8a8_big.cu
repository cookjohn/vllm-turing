// Large-M INT8 W8A8 GEMM for Turing (prefill), CUTLASS 2.x tensor-op kernel with vLLM's scaling epilogue
// (out[m, n] = (A[m, :] . W[n, :]) * sa[m] * sw[n], per-row activation and per-column weight scales, fp16 / bf16 out)
// at the tile configuration that measured fastest on the 2080 Ti for the Qwen3.8-27B TP2 shapes: 256 x 128 x 64
// threadblocks, 64 x 64 x 64 warps, raster swizzle 4 (vLLM's sm75 kernel: 128 x 128 x 64, swizzle 1; the 2-stage
// register-staged mainloop of sm75 caps every configuration near 55 % of the tensor-core peak, the tile choice is
// worth ~10 %). Built against the CUTLASS headers shipped in vllm/third_party with the system toolkit's nvcc.
// extern "C" int w8a8_big(A, W, sa, sw, out, M, N, K, dtype (0 fp16 / 1 bf16), stream) -> 0 ok, 1 cannot implement,
//   2 init failed, 3 run failed. A [M, K] and W [N, K] int8 row-major, sa [M] and sw [N] float32.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"

namespace tb = cutlass::epilogue::threadblock;

template <typename ElementD>
struct BigGemm {
    using TileShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstShape = cutlass::gemm::GemmShape<8, 8, 16>;
    using ThreadMap = tb::OutputTileThreadLayout<TileShape, WarpShape, float, 4, 1>;
    using Accum = tb::VisitorAccFetch;
    using ScaleA = tb::VisitorColBroadcast<ThreadMap, float, cute::Stride<cute::_1, cute::_0, cute::_0>>;
    using ScaleB = tb::VisitorRowBroadcast<ThreadMap, float, cute::Stride<cute::_0, cute::_1, cute::_0>>;
    using Compute0 = tb::VisitorCompute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT0 = tb::Sm80EVT<Compute0, ScaleB, Accum>;
    using Compute1 = tb::VisitorCompute<cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using EVT1 = tb::Sm80EVT<Compute1, ScaleA, EVT0>;
    using Store = tb::VisitorAuxStore<ThreadMap, ElementD, cutlass::FloatRoundStyle::round_to_nearest,
                                      cute::Stride<int64_t, cute::_1, cute::_0>>;
    using EVTD = tb::Sm80EVT<Store, EVT1>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        float, cutlass::layout::RowMajor, 4,
        int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        TileShape, WarpShape, InstShape, EVTD,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2,
        cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    static int run(const void* A, const void* W, const float* sa, const float* sw, void* out, int M, int N, int K,
                   cudaStream_t s) {
        typename ScaleA::Arguments a_args{sa, 0.0f, {}};
        typename ScaleB::Arguments b_args{sw, 0.0f, {}};
        typename EVT0::Arguments evt0{b_args, {}, {}};        // children (ScaleB, Accum), then the node (Compute0)
        typename EVT1::Arguments evt1{a_args, evt0, {}};
        typename Store::Arguments d_args{(ElementD*) out, {(int64_t) N, cute::_1{}, cute::_0{}}};
        typename EVTD::Arguments evt{evt1, d_args};
        typename Op::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, evt, A, W, nullptr, nullptr,
                                    0, 0, 0, 0, (int64_t) K, (int64_t) K, (int64_t) N, (int64_t) N);
        Op op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return 1;
        size_t ws = Op::get_workspace_size(args);
        if (ws != 0) return 1;                      // no split-K: no workspace expected
        if (op.initialize(args, nullptr, s) != cutlass::Status::kSuccess) return 2;
        if (op(s) != cutlass::Status::kSuccess) return 3;
        return 0;
    }
};

extern "C" int w8a8_big(const void* A, const void* W, const void* sa, const void* sw, void* out, int M, int N, int K,
                        int dtype, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    if (dtype == 0) {
        return BigGemm<cutlass::half_t>::run(A, W, (const float*) sa, (const float*) sw, out, M, N, K, s);
    }
    return BigGemm<cutlass::bfloat16_t>::run(A, W, (const float*) sa, (const float*) sw, out, M, N, K, s);
}
