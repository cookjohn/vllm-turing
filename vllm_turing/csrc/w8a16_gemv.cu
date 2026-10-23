// INT8-weight / fp16-activation GEMV for the LM head (and other unquantized projections) on Turing, M <= 16 rows:
//   out[M, N] = (x[M, K] . (W[N, K] * s[N])^T), W int8 row-major (per-row scale s, fp32), x fp16 / bf16, fp32 sums.
// Same structure as w8a8_gemv.cu (one warp per R rows, 16 weight bytes per lane per step, weights streamed past L1);
// the activations stay fp16 (no activation quantization: logits keep fp16-input accuracy).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename T> __device__ __forceinline__ float2 t2f(uint32_t v);
template <> __device__ __forceinline__ float2 t2f<half>(uint32_t v) { return __half22float2(*reinterpret_cast<half2 *>(&v)); }
template <> __device__ __forceinline__ float2 t2f<__nv_bfloat16>(uint32_t v) {
    return __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 *>(&v));
}
template <typename T> __device__ __forceinline__ T f2t(float x);
template <> __device__ __forceinline__ half f2t<half>(float x) { return __float2half_rn(x); }
template <> __device__ __forceinline__ __nv_bfloat16 f2t<__nv_bfloat16>(float x) { return __float2bfloat16_rn(x); }

static __device__ __forceinline__ int4 ld_stream(const int4 * p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// 4 int8 (one 32-bit word) times 4 activations (two 32-bit words of fp16 pairs)
template <typename T>
static __device__ __forceinline__ float dot4(uint32_t w, uint32_t x01, uint32_t x23, float acc) {
    const float2 a = t2f<T>(x01), b = t2f<T>(x23);
    acc = fmaf((float) (int8_t) (w & 0xff), a.x, acc);
    acc = fmaf((float) (int8_t) ((w >> 8) & 0xff), a.y, acc);
    acc = fmaf((float) (int8_t) ((w >> 16) & 0xff), b.x, acc);
    acc = fmaf((float) (int8_t) (w >> 24), b.y, acc);
    return acc;
}

template <typename T, int M, int R>
__global__ void __launch_bounds__(256) w8a16_kernel(const T * __restrict__ x, const int8_t * __restrict__ w,
                                                   const float * __restrict__ s, const T * __restrict__ bias,
                                                   T * __restrict__ out, const int N, const int K,
                                                   const int * __restrict__ rows) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int n0 = warp * R;
    if (n0 >= N) {
        return;
    }
    const int K16 = K >> 4;
    const int4 * w4[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int nr = min(n0 + r, N - 1);
        w4[r] = reinterpret_cast<const int4 *>(w) + (size_t) (rows ? rows[nr] : nr) * K16;
    }
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[r][m] = 0.0f;
        }
    }
    const uint4 * x4 = reinterpret_cast<const uint4 *>(x);   // 8 activations per uint4
#pragma unroll 2
    for (int k = lane; k < K16; k += 32) {
        int4 wv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wv[r] = ld_stream(w4[r] + k);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 xa = __ldg(x4 + ((size_t) m * K16 + k) * 2);       // activations 16k .. 16k+7
            const uint4 xb = __ldg(x4 + ((size_t) m * K16 + k) * 2 + 1);   // 16k+8 .. 16k+15
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float a = acc[r][m];
                a = dot4<T>((uint32_t) wv[r].x, xa.x, xa.y, a);
                a = dot4<T>((uint32_t) wv[r].y, xa.z, xa.w, a);
                a = dot4<T>((uint32_t) wv[r].z, xb.x, xb.y, a);
                a = dot4<T>((uint32_t) wv[r].w, xb.z, xb.w, a);
                acc[r][m] = a;
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
                const int   ns = rows ? rows[n] : n;   // source row (rows: gather a subset, output compact)
                const float b  = bias ? (float) bias[ns] : 0.0f;
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    out[(size_t) m * N + n] = f2t<T>(acc[r][m] * s[ns] + b);
                }
            }
        }
    }
}

template <typename T, int M>
static void launch_m(const T * x, const int8_t * w, const float * s, const T * bias, T * out, int N, int K,
                     const int * rows, cudaStream_t st) {
    // rows per warp: each lane's activation loads (32 bytes per m per 16 weight bytes) serve R weight rows; with one row
    // per warp they bound the kernel on L1 from M ~ 4 (M = 7: 243 GB/s, the speculative verify / DFlash candidates)
    constexpr int R = M <= 2 ? 2 : 4;   // 8 spills registers at M > 8 (M = 16: 10 ms vs 2.9)
    const int nb = (N + 8 * R - 1) / (8 * R);
    w8a16_kernel<T, M, R><<<nb, 256, 0, st>>>(x, w, s, bias, out, N, K, rows);
}

template <typename T>
static void launch(const T * x, const int8_t * w, const float * s, const T * bias, T * out, int M, int N, int K,
                   const int * rows, cudaStream_t st) {
    switch (M) {
        case 1: launch_m<T, 1>(x, w, s, bias, out, N, K, rows, st); break;
        case 2: launch_m<T, 2>(x, w, s, bias, out, N, K, rows, st); break;
        case 3: launch_m<T, 3>(x, w, s, bias, out, N, K, rows, st); break;
        case 4: launch_m<T, 4>(x, w, s, bias, out, N, K, rows, st); break;
        case 5: launch_m<T, 5>(x, w, s, bias, out, N, K, rows, st); break;
        case 6: launch_m<T, 6>(x, w, s, bias, out, N, K, rows, st); break;
        case 7: launch_m<T, 7>(x, w, s, bias, out, N, K, rows, st); break;
        case 8: launch_m<T, 8>(x, w, s, bias, out, N, K, rows, st); break;
        case 9: launch_m<T, 9>(x, w, s, bias, out, N, K, rows, st); break;
        case 10: launch_m<T, 10>(x, w, s, bias, out, N, K, rows, st); break;
        case 11: launch_m<T, 11>(x, w, s, bias, out, N, K, rows, st); break;
        case 12: launch_m<T, 12>(x, w, s, bias, out, N, K, rows, st); break;
        case 13: launch_m<T, 13>(x, w, s, bias, out, N, K, rows, st); break;
        case 14: launch_m<T, 14>(x, w, s, bias, out, N, K, rows, st); break;
        case 15: launch_m<T, 15>(x, w, s, bias, out, N, K, rows, st); break;
        case 16: launch_m<T, 16>(x, w, s, bias, out, N, K, rows, st); break;
    }
}

// dtype: 0 = float16, 1 = bfloat16. rows (optional, int32 [N]): compute only these weight rows (a gathered subset,
// e.g. a pruned draft vocabulary), out is [M, N] over the subset.
extern "C" int w8a16_gemv_rows(const void * x, const void * w, const void * s, const void * bias, void * out, int M,
                               int N, int K, int dtype, const void * rows, void * stream);

extern "C" int w8a16_gemv(const void * x, const void * w, const void * s, const void * bias, void * out, int M, int N,
                          int K, int dtype, void * stream) {
    return w8a16_gemv_rows(x, w, s, bias, out, M, N, K, dtype, nullptr, stream);
}

extern "C" int w8a16_gemv_rows(const void * x, const void * w, const void * s, const void * bias, void * out, int M,
                               int N, int K, int dtype, const void * rows, void * stream) {
    if (M < 1 || M > 16 || (K & 15) != 0) {
        return -1;
    }
    if (dtype == 0) {
        launch<half>((const half *) x, (const int8_t *) w, (const float *) s, (const half *) bias, (half *) out, M, N, K,
                     (const int *) rows, (cudaStream_t) stream);
    } else {
        launch<__nv_bfloat16>((const __nv_bfloat16 *) x, (const int8_t *) w, (const float *) s,
                              (const __nv_bfloat16 *) bias, (__nv_bfloat16 *) out, M, N, K, (const int *) rows,
                              (cudaStream_t) stream);
    }
    return (int) cudaGetLastError();
}
