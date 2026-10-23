// Small-M INT8 W8A8 GEMM for Turing (sm75), the decode / speculative-verify shapes (M <= 16 rows):
//   out[M, N] = (a[M, K] . W[N, K]^T) * scale_a * scale_w (+ bias), a and W int8, int32 accumulation (dp4a)
// CUTLASS pads a few rows to a full 128-row tile and reaches ~50 % of the memory bandwidth there; this reads every W row
// once with 16-byte loads (one warp per R rows, lanes stride over K), the activations (a few KB) stay in L1.
// W is the row-major [N, K] int8 weight (vLLM passes it as the transposed [K, N] view, stride (1, K)).
// Exact: the same int32 dot products and float scaling as cutlass_scaled_mm.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename T> __device__ __forceinline__ T     to_out(float x);
template <> __device__ __forceinline__ half           to_out<half>(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16  to_out<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }
template <typename T> __device__ __forceinline__ float from_t(T x);
template <> __device__ __forceinline__ float           from_t<half>(half x) { return __half2float(x); }
template <> __device__ __forceinline__ float           from_t<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

// weights are read once: no L1 allocation, keep L1 for the activations
static __device__ __forceinline__ int4 ld_stream(const int4 * p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

template <typename T, int M, int R>
__global__ void __launch_bounds__(256) w8a8_gemv_kernel(const int8_t * __restrict__ a, const int8_t * __restrict__ w,
                                                        const float * __restrict__ sa, const float * __restrict__ sw,
                                                        const T * __restrict__ bias, T * __restrict__ out,
                                                        const int N, const int K, const int sa_rows, const int sw_rows) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = warp * R;
    if (n0 >= N) {
        return;
    }
    const int    K16 = K >> 4;
    const int4 * a4  = reinterpret_cast<const int4 *>(a);
    const int4 * w4[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        w4[r] = reinterpret_cast<const int4 *>(w) + (size_t) min(n0 + r, N - 1) * K16;
    }
    int acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[r][m] = 0;
        }
    }
#pragma unroll 4
    for (int k = lane; k < K16; k += 32) {
        int4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wv[r] = ld_stream(w4[r] + k);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const int4 av = __ldg(a4 + (size_t) m * K16 + k);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r][m] = __dp4a(wv[r].x, av.x, acc[r][m]);
                acc[r][m] = __dp4a(wv[r].y, av.y, acc[r][m]);
                acc[r][m] = __dp4a(wv[r].z, av.z, acc[r][m]);
                acc[r][m] = __dp4a(wv[r].w, av.w, acc[r][m]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                acc[r][m] += __shfl_xor_sync(0xffffffff, acc[r][m], o);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int n = n0 + r;
            if (n < N) {
                const float swn = sw[sw_rows ? n : 0];
                const float b   = bias ? from_t<T>(bias[n]) : 0.0f;
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    out[(size_t) m * N + n] = to_out<T>((float) acc[r][m] * sa[sa_rows ? m : 0] * swn + b);
                }
            }
        }
    }
}

template <typename T, int M>
static void launch_m(const int8_t * a, const int8_t * w, const float * sa, const float * sw, const T * bias, T * out,
                     int N, int K, int sa_rows, int sw_rows, cudaStream_t s) {
    // rows per warp (loads in flight and activation reuse vs registers), measured per M range: one layer of each
    // shape M = 10: 498 us (R 2) / 518 (1); M = 16: 598 (4) / 668 (2) / 745 (1); CUTLASS ~787 throughout
    constexpr int R      = M <= 4 ? 2 : (M <= 8 ? 1 : (M <= 12 ? 2 : 4));
    constexpr int rows_b = 8 * R;                     // 8 warps per block
    const int     nb     = (N + rows_b - 1) / rows_b;
    w8a8_gemv_kernel<T, M, R><<<nb, 256, 0, s>>>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows);
}

template <typename T>
static void launch(const int8_t * a, const int8_t * w, const float * sa, const float * sw, const T * bias, T * out,
                   int M, int N, int K, int sa_rows, int sw_rows, cudaStream_t s) {
    switch (M) {
        case 1: launch_m<T, 1>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 2: launch_m<T, 2>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 3: launch_m<T, 3>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 4: launch_m<T, 4>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 5: launch_m<T, 5>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 6: launch_m<T, 6>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 7: launch_m<T, 7>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 8: launch_m<T, 8>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 9: launch_m<T, 9>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 10: launch_m<T, 10>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 11: launch_m<T, 11>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 12: launch_m<T, 12>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 13: launch_m<T, 13>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 14: launch_m<T, 14>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 15: launch_m<T, 15>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
        case 16: launch_m<T, 16>(a, w, sa, sw, bias, out, N, K, sa_rows, sw_rows, s); break;
    }
}

// ---- M = 13..32 rows (MTP verify of 3..6 concurrent requests): int8 tensor cores --------------------------------------
// CUTLASS pads these to 128-row tiles (~300 GB/s, 775 us for one layer of each shape); the dp4a kernel above runs out of
// issue slots past M = 12. Here out^T = W . a^T with mma.m8n8k16.s8 (Turing): W rows are the MMA's A operand (8 rows per
// tile), activation rows its B operand (8 per tile). Lane (g, t) loads 32 bytes of row g at k = 128 s + 32 t for both
// operands and feeds 32-bit word c of them to MMA c: the same permutation of k on both sides, so each MMA sums a
// consistent set of 16 k values and the 8 MMAs of a step cover all 128 (every row is read a full 128-byte line per step:
// 64-byte halves ran at ~300 GB/s). Every warp streams 16 W rows past L1 and reads all M activation rows (L1 hits,
// shared by the warps of a block); KS warps split K for the same rows (enough blocks for small N) and their partial sums
// meet in shared memory. One layer of each shape: M = 13..16 525-530 us (dp4a 526-576), 20: 549, 25: 572, 30: 592.
// Past M = 32 (5+ activation tiles) it is no faster than CUTLASS.
static __device__ __forceinline__ void mma_s8(int (&c)[2], int a, int b) {
    // not volatile: lets the compiler move the next step's loads above the MMAs
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%0, %1};"
        : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}

template <typename T, int MT, int KS>
__global__ void __launch_bounds__(256) w8a8_mma_kernel(const int8_t * __restrict__ a, const int8_t * __restrict__ w,
                                                       const float * __restrict__ sa, const float * __restrict__ sw,
                                                       const T * __restrict__ bias, T * __restrict__ out,
                                                       const int M, const int N, const int K, const int sa_rows,
                                                       const int sw_rows) {
    constexpr int WR = 8 / KS;                      // row groups per block (16 W rows each)
    __shared__ int red[KS > 1 ? WR * (KS - 1) * 32 * 4 * MT : 1];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int rg = warp % WR, ks = warp / WR;
    const int g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * WR + rg) * 16;
    const int4 * w0 = reinterpret_cast<const int4 *>(w + (size_t) min(n0 + g, N - 1) * K);
    const int4 * w1 = reinterpret_cast<const int4 *>(w + (size_t) min(n0 + 8 + g, N - 1) * K);
    const int4 * ap[MT];
    bool av[MT];
#pragma unroll
    for (int j = 0; j < MT; ++j) {
        av[j] = 8 * j + g < M;
        ap[j] = reinterpret_cast<const int4 *>(a + (size_t) min(8 * j + g, M - 1) * K);
    }
    int acc[2][MT][2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            acc[i][j][0] = acc[i][j][1] = 0;
        }
    }
    // 128-byte k steps: lane t holds bytes [128 s + 32 t, + 32) of its rows (two int4), so every row is read a full
    // 128-byte line per step (64-byte halves, one per instruction, ran at ~300 GB/s)
    const int  K16   = K >> 4;
    const int  steps = (K16 + 7) >> 3;
    const int4 z     = make_int4(0, 0, 0, 0);
    // weights of the next step are loaded before this step's MMAs (two steps of loads in flight per warp); with MT = 4
    // (2 blocks per SM either way) the activations of the next step too: loaded just in time, their L1 misses stalled
    // every step (M = 30 at 66% of the DRAM peak). Two steps of weights ahead did not help.
    constexpr bool XPF = MT >= 4 && KS <= 2;   // KS = 8 (out_proj): 1 block per SM at 136 registers was slower
    int  idx = ks * 8 + 2 * t;
    int4 na0 = idx < K16 ? ld_stream(w0 + idx) : z, na1 = idx + 1 < K16 ? ld_stream(w0 + idx + 1) : z;
    int4 nb0 = idx < K16 ? ld_stream(w1 + idx) : z, nb1 = idx + 1 < K16 ? ld_stream(w1 + idx + 1) : z;
    int4 nx[XPF ? MT : 1][2];
    if constexpr (XPF) {
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            nx[j][0] = av[j] && idx < K16 ? __ldg(ap[j] + idx) : z;
            nx[j][1] = av[j] && idx + 1 < K16 ? __ldg(ap[j] + idx + 1) : z;
        }
    }
    for (int s = ks; s < steps; s += KS) {
        const int  ci = idx;
        const int4 wa0 = na0, wa1 = na1, wb0 = nb0, wb1 = nb1;
        int4 cx[XPF ? MT : 1][2];
        if constexpr (XPF) {
#pragma unroll
            for (int j = 0; j < MT; ++j) {
                cx[j][0] = nx[j][0];
                cx[j][1] = nx[j][1];
            }
        }
        idx += 8 * KS;
        if (s + KS < steps) {
            na0 = idx < K16 ? ld_stream(w0 + idx) : z;
            na1 = idx + 1 < K16 ? ld_stream(w0 + idx + 1) : z;
            nb0 = idx < K16 ? ld_stream(w1 + idx) : z;
            nb1 = idx + 1 < K16 ? ld_stream(w1 + idx + 1) : z;
            if constexpr (XPF) {
#pragma unroll
                for (int j = 0; j < MT; ++j) {
                    nx[j][0] = av[j] && idx < K16 ? __ldg(ap[j] + idx) : z;
                    nx[j][1] = av[j] && idx + 1 < K16 ? __ldg(ap[j] + idx + 1) : z;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < MT; ++j) {
            const int4 b0 = XPF ? cx[j][0] : (av[j] && ci < K16 ? __ldg(ap[j] + ci) : z);
            const int4 b1 = XPF ? cx[j][1] : (av[j] && ci + 1 < K16 ? __ldg(ap[j] + ci + 1) : z);
            mma_s8(acc[0][j], wa0.x, b0.x);
            mma_s8(acc[0][j], wa0.y, b0.y);
            mma_s8(acc[0][j], wa0.z, b0.z);
            mma_s8(acc[0][j], wa0.w, b0.w);
            mma_s8(acc[0][j], wa1.x, b1.x);
            mma_s8(acc[0][j], wa1.y, b1.y);
            mma_s8(acc[0][j], wa1.z, b1.z);
            mma_s8(acc[0][j], wa1.w, b1.w);
            mma_s8(acc[1][j], wb0.x, b0.x);
            mma_s8(acc[1][j], wb0.y, b0.y);
            mma_s8(acc[1][j], wb0.z, b0.z);
            mma_s8(acc[1][j], wb0.w, b0.w);
            mma_s8(acc[1][j], wb1.x, b1.x);
            mma_s8(acc[1][j], wb1.y, b1.y);
            mma_s8(acc[1][j], wb1.z, b1.z);
            mma_s8(acc[1][j], wb1.w, b1.w);
        }
    }
    if constexpr (KS > 1) {
        if (ks > 0) {
            int * r = red + (rg * (KS - 1) + ks - 1) * 32 * 4 * MT + lane;     // [element][lane]: no bank conflicts
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < MT; ++j) {
                    r[((i * MT + j) * 2) * 32]     = acc[i][j][0];
                    r[((i * MT + j) * 2 + 1) * 32] = acc[i][j][1];
                }
            }
        }
        __syncthreads();
        if (ks > 0) {
            return;
        }
#pragma unroll
        for (int q = 0; q < KS - 1; ++q) {
            const int * r = red + (rg * (KS - 1) + q) * 32 * 4 * MT + lane;
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int j = 0; j < MT; ++j) {
                    acc[i][j][0] += r[((i * MT + j) * 2) * 32];
                    acc[i][j][1] += r[((i * MT + j) * 2 + 1) * 32];
                }
            }
        }
    }
    // C fragment: lane (g, t) holds W row g, activation rows 2 t and 2 t + 1 of each 8 x 8 tile
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int n = n0 + 8 * i + g;
        if (n >= N) {
            continue;
        }
        const float swn = sw[sw_rows ? n : 0];
        const float bn  = bias ? from_t<T>(bias[n]) : 0.0f;
#pragma unroll
        for (int j = 0; j < MT; ++j) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int m = 8 * j + 2 * t + r;
                if (m < M) {
                    out[(size_t) m * N + n] = to_out<T>((float) acc[i][j][r] * sa[sa_rows ? m : 0] * swn + bn);
                }
            }
        }
    }
}

template <typename T, int MT>
static void launch_mma_m(const int8_t * a, const int8_t * w, const float * sa, const float * sw, const T * bias, T * out,
                         int M, int N, int K, int sa_rows, int sw_rows, cudaStream_t s) {
    // KS warps of a block split K for the same 16-row groups. Measured at the Qwen3.8 TP2 shapes (10-23, CUDA graphs;
    // N x K = 17408 x 5120 gate_up, 8192 / 7168 x 5120 GDN / attention in_proj, 5120 x 8704 down, 5120 x 3072 out_proj):
    // MT <= 3 (M <= 24) is fastest with KS = 4 (gate_up 179 vs 221 us at M = 16), except the 7-8k-column in_proj with
    // KS = 2; MT = 4 (M 25..32, two registers more per thread: 2 blocks per SM) with KS = 2, except short-K out_proj
    // with KS = 8 (K / 8 fits few 128-byte steps per warp otherwise)
    int ks;
    if (MT >= 4) {
        ks = (N < 7000 && K < 4096) ? 8 : 2;
    } else {
        ks = (N >= 7000 && N < 16384) ? 2 : 4;
    }
    if (ks == 2) {
        w8a8_mma_kernel<T, MT, 2><<<(N + 63) / 64, 256, 0, s>>>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows);
    } else if (ks == 4) {
        w8a8_mma_kernel<T, MT, 4><<<(N + 31) / 32, 256, 0, s>>>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows);
    } else {
        w8a8_mma_kernel<T, MT, 8><<<(N + 15) / 16, 256, 0, s>>>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows);
    }
}

template <typename T>
static void launch_mma(const int8_t * a, const int8_t * w, const float * sa, const float * sw, const T * bias, T * out,
                       int M, int N, int K, int sa_rows, int sw_rows, cudaStream_t s) {
    switch ((M + 7) / 8) {
        case 1: launch_mma_m<T, 1>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows, s); break;
        case 2: launch_mma_m<T, 2>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows, s); break;
        case 3: launch_mma_m<T, 3>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows, s); break;
        case 4: launch_mma_m<T, 4>(a, w, sa, sw, bias, out, M, N, K, sa_rows, sw_rows, s); break;
    }
}

// M = 1..32; dtype as below. Returns the CUDA error code of the launch (0 = success).
extern "C" int w8a8_mma(const void * a, const void * w, const void * sa, const void * sw, const void * bias, void * out,
                        int M, int N, int K, int sa_rows, int sw_rows, int dtype, void * stream) {
    if (M < 1 || M > 32 || (K & 15) != 0) {
        return -1;
    }
    cudaStream_t s = (cudaStream_t) stream;
    if (dtype == 0) {
        launch_mma<half>((const int8_t *) a, (const int8_t *) w, (const float *) sa, (const float *) sw,
                         (const half *) bias, (half *) out, M, N, K, sa_rows, sw_rows, s);
    } else {
        launch_mma<__nv_bfloat16>((const int8_t *) a, (const int8_t *) w, (const float *) sa, (const float *) sw,
                                  (const __nv_bfloat16 *) bias, (__nv_bfloat16 *) out, M, N, K, sa_rows, sw_rows, s);
    }
    return (int) cudaGetLastError();
}

// dtype: 0 = float16, 1 = bfloat16. Returns the CUDA error code of the launch (0 = success).
extern "C" int w8a8_gemv(const void * a, const void * w, const void * sa, const void * sw, const void * bias, void * out,
                         int M, int N, int K, int sa_rows, int sw_rows, int dtype, void * stream) {
    if (M < 1 || M > 16 || (K & 15) != 0) {
        return -1;
    }
    cudaStream_t s = (cudaStream_t) stream;
    if (dtype == 0) {
        launch<half>((const int8_t *) a, (const int8_t *) w, (const float *) sa, (const float *) sw, (const half *) bias,
                     (half *) out, M, N, K, sa_rows, sw_rows, s);
    } else {
        launch<__nv_bfloat16>((const int8_t *) a, (const int8_t *) w, (const float *) sa, (const float *) sw,
                              (const __nv_bfloat16 *) bias, (__nv_bfloat16 *) out, M, N, K, sa_rows, sw_rows, s);
    }
    return (int) cudaGetLastError();
}
