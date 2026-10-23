// CUTLASS 2.x int8 tensor-op GEMM (Turing, mma.m8n8k16) tile-config sweep for the prefill shapes of Qwen3.8-27B TP2:
// out[M, N] (fp16) = A[M, K] (int8, row-major) . W[N, K]^T (int8, row-major = column-major B), alpha = 1.
// Built against the CUTLASS headers shipped in vllm/third_party (no scaling epilogue here: raw GEMM throughput per
// config; vLLM's sm75 kernel is config 0 with a per-row / per-column scaling visitor epilogue).
// extern "C" int cutlass_gemm_cfg(int cfg, const void* A, const void* W, void* C, int M, int N, int K, void* stream)
//   -> 0 ok, 1 cannot implement, 2 init failed, 3 run failed, -1 unknown cfg
#include <cuda_fp16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

template <int TM, int TN, int TK, int WM, int WN, int WK, typename Swizzle>
struct Cfg {
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t, cutlass::layout::RowMajor,
        int8_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<TM, TN, TK>, cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<8, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, int32_t, float>,
        Swizzle, 2>;
    static int run(const void* A, const void* W, void* C, int M, int N, int K, cudaStream_t s) {
        typename Gemm::Arguments args({M, N, K}, {(const int8_t*) A, K}, {(const int8_t*) W, K},
                                      {(cutlass::half_t*) C, N}, {(cutlass::half_t*) C, N}, {1.0f, 0.0f});
        Gemm op;
        if (op.can_implement(args) != cutlass::Status::kSuccess) return 1;
        if (op.initialize(args, nullptr, s) != cutlass::Status::kSuccess) return 2;
        if (op(s) != cutlass::Status::kSuccess) return 3;
        return 0;
    }
};

using Sw1 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using Sw4 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
using Sw8 = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

extern "C" int cutlass_gemm_cfg(int cfg, const void* A, const void* W, void* C, int M, int N, int K, void* stream) {
    cudaStream_t s = (cudaStream_t) stream;
    switch (cfg) {
        case 0: return Cfg<128, 128, 64, 64, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 1: return Cfg<256, 128, 64, 64, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 2: return Cfg<128, 256, 64, 64, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 3: return Cfg<256, 128, 32, 64, 64, 32, Sw1>::run(A, W, C, M, N, K, s);
        case 4: return Cfg<128, 128, 128, 64, 64, 128, Sw1>::run(A, W, C, M, N, K, s);
        case 5: return Cfg<64, 128, 64, 32, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 6: return Cfg<128, 64, 64, 64, 32, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 7: return Cfg<256, 128, 64, 64, 64, 64, Sw4>::run(A, W, C, M, N, K, s);
        case 8: return Cfg<256, 128, 64, 64, 64, 64, Sw8>::run(A, W, C, M, N, K, s);
        case 9: return Cfg<128, 128, 64, 64, 64, 64, Sw8>::run(A, W, C, M, N, K, s);
        case 10: return Cfg<256, 128, 64, 128, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 11: return Cfg<128, 128, 64, 32, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
        case 12: return Cfg<256, 64, 64, 64, 64, 64, Sw1>::run(A, W, C, M, N, K, s);
    }
    return -1;
}
