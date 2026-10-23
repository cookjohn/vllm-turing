// INT8-weight / fp16-activation GEMM on Turing tensor cores for 17..64 rows (the LM head of a speculative verify with
// several concurrent requests: 6 x 5 MTP tokens = 30 rows). out[M, N] = (x[M, K] . W[N, K]^T) * s[N] (+ bias).
// The W8A16 GEMV (w8a16_gemv.cu) runs fp32 FMAs on CUDA cores: its cost grows with the rows (16 rows: 3.8 ms for the
// 124160 x 5120 head, memory floor ~1.15 ms). Here the weights stream once: each lane loads 16 int8 of one weight row
// (ld.global.nc, past L1), converts them to fp16 in registers (byte permute into 0x64xx + subtraction: exact) and feeds
// 8 mma.m16n8k8 (fp16 operands, fp32 accumulation; products exact) with the activations from shared memory. A k
// permutation shared by both operands lets one 16-byte load serve 8 MMAs: MMA j of a 64-wide k segment takes, in lane
// (g, t), weight bytes 16t + 2j, +1 and the activations at the same k. Per-row weight scales and the bias are applied to
// the fp32 sums (exact: the scale factors out of the sum). Block: 8 warps x 32 output columns, x staged in 128-wide k
// tiles (row stride 65 words: conflict-free fragment reads); the next tile's weights are loaded before the MMAs.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

constexpr int MW_THREADS = 256;

static __device__ __forceinline__ int4 ld_nc(const void * p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

// 4 int8 (one word, bytes k .. k+3) -> 2 half2 {k, k+1}, {k+2, k+3}
static __device__ __forceinline__ void i8x4_to_h2x2(uint32_t w, uint32_t & lo, uint32_t & hi) {
    w ^= 0x80808080u;                                               // signed -> biased unsigned
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(lo) : "r"(w), "n"(0x64646464), "n"(0x5150));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(hi) : "r"(w), "n"(0x64646464), "n"(0x5352));
    asm("sub.f16x2 %0, %1, %2;" : "=r"(lo) : "r"(lo), "r"(0x64806480u));   // (1024 + u) - 1152 = u - 128
    asm("sub.f16x2 %0, %1, %2;" : "=r"(hi) : "r"(hi), "r"(0x64806480u));
}

static __device__ __forceinline__ void hmma(float * d, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}

// MT m-tiles of 16 rows (rows >= M read as zero), NT n-tiles of 8 columns per warp, KT-wide k tiles (each weight row
// is read in KT-byte runs per tile: longer runs keep DRAM pages open)
template <int MT, int NT, int KT>
__global__ void __launch_bounds__(MW_THREADS, 2) w8a16_mma_kernel(const half * __restrict__ x, const int8_t * __restrict__ w,
                                                                  const float * __restrict__ s, const half * __restrict__ bias,
                                                                  half * __restrict__ out, const int M, const int N,
                                                                  const int K) {
    constexpr int XLD = KT + 2, NSG = KT / 64;   // x row stride in halves (odd word count: conflict-free reads)
    __shared__ __align__(16) half sx[MT * 16 * XLD];
    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int n0 = (blockIdx.x * 8 + wp) * 8 * NT;           // the warp's NT n-tiles: n0 + 8 nt + g
    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[mt][nt][e] = 0.0f;
            }
        }
    }
    // the lane's weight rows (one per n-tile; rows past N are clamped and their sums dropped)
    const int8_t * wrow[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        wrow[nt] = w + (int64_t) min(n0 + 8 * nt + g, N - 1) * K + 16 * t;
    }
    int4 wb[NT][NSG];                                       // a k tile: NSG segments of 64 per n-tile
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int sg = 0; sg < NSG; ++sg) {
            wb[nt][sg] = ld_nc(wrow[nt] + 64 * sg);
        }
    }
    for (int k0 = 0; k0 < K; k0 += KT) {
        __syncthreads();                                    // previous tile's x reads done
        for (int i = tid; i < MT * 16 * (KT / 2); i += MW_THREADS) {   // x tile, one word (2 halves) per step
            const int r = i / (KT / 2), c = (i % (KT / 2)) * 2;
            uint32_t v = 0;
            if (r < M) {
                v = *reinterpret_cast<const uint32_t *>(x + (int64_t) r * K + k0 + c);
            }
            *reinterpret_cast<uint32_t *>(sx + r * XLD + c) = v;
        }
        __syncthreads();
#pragma unroll
        for (int sg = 0; sg < NSG; ++sg) {
            // B: lane (g, t) of n-tile nt holds weight bytes [64 sg + 16 t, +16) of row n0 + 8 nt + g; converted
            // first, then the same registers take the next tile's segment (in flight during the MMAs)
            uint32_t b[NT][8];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint32_t ww[4] = {(uint32_t) wb[nt][sg].x, (uint32_t) wb[nt][sg].y, (uint32_t) wb[nt][sg].z,
                                        (uint32_t) wb[nt][sg].w};
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    i8x4_to_h2x2(ww[q], b[nt][2 * q], b[nt][2 * q + 1]);
                }
            }
            if (k0 + KT < K) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    wb[nt][sg] = ld_nc(wrow[nt] + k0 + KT + 64 * sg);
                }
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {                   // MMA j: k = 64 sg + 16 t + 2 j, +1 in lane (g, t)
                const int kc = 64 * sg + 16 * t + 2 * j;
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const uint32_t a0 = *reinterpret_cast<const uint32_t *>(sx + (16 * mt + g) * XLD + kc);
                    const uint32_t a1 = *reinterpret_cast<const uint32_t *>(sx + (16 * mt + g + 8) * XLD + kc);
#pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        hmma(acc[mt][nt], a0, a1, b[nt][j]);
                    }
                }
            }
        }
    }
    // D fragment: rows 16 mt + g (+8), columns n0 + 8 nt + 2 t (+1)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int c = n0 + 8 * nt + 2 * t;
        if (c >= N) {
            continue;
        }
        const bool c1 = c + 1 < N;
        const float s0 = s[c], s1 = c1 ? s[c + 1] : 0.0f;
        const float b0 = bias ? __half2float(bias[c]) : 0.0f, b1 = (bias && c1) ? __half2float(bias[c + 1]) : 0.0f;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int r = 16 * mt + g + 8 * hh;
                if (r < M) {
                    half * o = out + (int64_t) r * N + c;
                    const float v0 = acc[mt][nt][2 * hh] * s0 + b0, v1 = acc[mt][nt][2 * hh + 1] * s1 + b1;
                    if (c1) {
                        *reinterpret_cast<half2 *>(o) = __floats2half2_rn(v0, v1);
                    } else {
                        o[0] = __float2half_rn(v0);
                    }
                }
            }
        }
    }
}

// the x tile is static shared memory (<= 48 KB): wide k tiles take fewer m-tiles
template <int MT, int NT, int KT>
static int run(dim3 grid, const half * x, const int8_t * w, const float * s, const half * bias, half * out, int M, int N,
               int K, cudaStream_t st) {
    if constexpr (MT * 16 * (KT + 2) * 2 <= 48 * 1024) {
        w8a16_mma_kernel<MT, NT, KT><<<grid, MW_THREADS, 0, st>>>(x, w, s, bias, out, M, N, K);
        return (int) cudaGetLastError();
    } else {
        return -2;
    }
}

template <int NT, int KT>
static int launch(const half * x, const int8_t * w, const float * s, const half * bias, half * out, int M, int N, int K,
                  cudaStream_t st) {
    if (K % KT != 0) {
        return -1;
    }
    const dim3 grid((N + 64 * NT - 1) / (64 * NT));
    switch ((M + 15) / 16) {
        case 1: return run<1, NT, KT>(grid, x, w, s, bias, out, M, N, K, st);
        case 2: return run<2, NT, KT>(grid, x, w, s, bias, out, M, N, K, st);
        case 3: return run<3, NT, KT>(grid, x, w, s, bias, out, M, N, K, st);
        default: return run<4, NT, KT>(grid, x, w, s, bias, out, M, N, K, st);
    }
}

// x [M, K] fp16, w [N, K] int8 (row-major), s [N] fp32, bias [N] fp16 or null, out [M, N] fp16 (N even for the
// paired stores). 1 <= M <= 64; cfg: 0 = 4 n-tiles x 128-wide k tiles, 1 = 2 x 256, 2 = 1 x 512
extern "C" int w8a16_mma(const void * x, const void * w, const void * s, const void * bias, void * out, int M, int N,
                         int K, int cfg, void * stream) {
    if (M < 1 || M > 64 || N % 2 != 0) {
        return -1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    const half * xh = (const half *) x;
    const int8_t * wq = (const int8_t *) w;
    switch (cfg) {
        case 0: return launch<4, 128>(xh, wq, (const float *) s, (const half *) bias, (half *) out, M, N, K, st);
        case 1: return launch<2, 256>(xh, wq, (const float *) s, (const half *) bias, (half *) out, M, N, K, st);
        default: return launch<1, 512>(xh, wq, (const float *) s, (const half *) bias, (half *) out, M, N, K, st);
    }
}
