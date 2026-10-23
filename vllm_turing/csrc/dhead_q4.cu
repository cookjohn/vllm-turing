// Greedy draft head over a pruned vocabulary, 4-bit weights, fused argmax (Turing, M <= 8 rows):
//   best[m] = argmax_n (x[m, :] . W[n, :]),  W: Q4_0-style blocks of 32 (fp16 scale d, q in 0..15, w = d * (q - 8)).
// Weight layout per row: K / 32 chunks of 16 bytes (4 words of 8 nibbles); word j of chunk c holds elements
// 32c + 8j .. 32c + 8j + 7 with element 2i at bits [4i, 4i + 3] and element 2i + 1 at bits [4i + 16, 4i + 19] (i = 0..3),
// so (word >> 4i) & 0x000F000F | 0x64006400 is the half2 (1024 + q_2i, 1024 + q_2i+1) (the usual fp16 magic-number
// dequant); subtracting 1032 gives q - 8. Per word the 8 products are summed in fp16 (half2 FMAs), then per chunk
// scaled and accumulated in fp32. One warp per R rows, 16 weight bytes per lane per step, weights streamed past L1.
// Pass 1 writes the per-block (value, local row) maximum, pass 2 reduces the blocks and writes the exchange slot of this
// rank: ex[m, rank, 0..3] = (value, gid / 2048, gid % 2048, 0) as fp16 (integers < 2048 are exact in fp16), zeros in the
// other ranks' slots, so a sum all-reduce over ranks gathers every rank's candidate.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ int4 ld_stream(const int4 * p) {
    int4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

static __device__ __forceinline__ half2 u2h2(uint32_t v) { return *reinterpret_cast<half2 *>(&v); }

// 8 weights (one word) times 8 activations (4 half2) into a half2 partial sum
static __device__ __forceinline__ half2 dot8(uint32_t w, const uint4 & xa, half2 acc) {
    const half2 off = __float2half2_rn(1032.0f);
    acc = __hfma2(__hsub2(u2h2((w & 0x000F000Fu) | 0x64006400u), off), u2h2(xa.x), acc);
    acc = __hfma2(__hsub2(u2h2(((w >> 4) & 0x000F000Fu) | 0x64006400u), off), u2h2(xa.y), acc);
    acc = __hfma2(__hsub2(u2h2(((w >> 8) & 0x000F000Fu) | 0x64006400u), off), u2h2(xa.z), acc);
    acc = __hfma2(__hsub2(u2h2(((w >> 12) & 0x000F000Fu) | 0x64006400u), off), u2h2(xa.w), acc);
    return acc;
}

template <int M, int R>
__global__ void __launch_bounds__(256) dhead_kernel(const half * __restrict__ x, const uint8_t * __restrict__ w,
                                                   const half * __restrict__ d, const int N, const int K,
                                                   float * __restrict__ pval, int * __restrict__ pidx) {
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * 8 + wid) * R;
    const int KC = K >> 5;   // 32-element chunks per row
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[r][m] = 0.0f;
        }
    }
    if (n0 < N) {
        const int4 * w4[R];
        const half * dr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int nr = min(n0 + r, N - 1);
            w4[r] = reinterpret_cast<const int4 *>(w) + (size_t) nr * KC;
            dr[r] = d + (size_t) nr * KC;
        }
        const uint4 * x4 = reinterpret_cast<const uint4 *>(x);   // 8 activations per uint4, 4 per chunk
#pragma unroll 2
        for (int c = lane; c < KC; c += 32) {
            int4  wv[R];
            float dv[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                wv[r] = ld_stream(w4[r] + c);
                dv[r] = __half2float(dr[r][c]);
            }
            // dequantized once per chunk (q - 8 pairs, as dot8 forms them), not once per activation row
            half2 wq[R][16];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t ww[4] = {(uint32_t) wv[r].x, (uint32_t) wv[r].y, (uint32_t) wv[r].z, (uint32_t) wv[r].w};
                const half2 off = __float2half2_rn(1032.0f);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        wq[r][4 * j + i] = __hsub2(u2h2(((ww[j] >> (4 * i)) & 0x000F000Fu) | 0x64006400u), off);
                    }
                }
            }
#pragma unroll
            for (int m = 0; m < M; ++m) {
                const uint4 * xm = x4 + ((size_t) m * KC + c) * 4;
                const uint4   xv[4] = {__ldg(xm), __ldg(xm + 1), __ldg(xm + 2), __ldg(xm + 3)};
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    // fp16 sums of 8 products per word (|q - 8| <= 8: no overflow below |x| ~ 2000), fp32 beyond;
                    // the same FMA order as dot8
                    float sum = 0.0f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        half2 p = __float2half2_rn(0.0f);
                        p = __hfma2(wq[r][4 * j + 0], u2h2(xv[j].x), p);
                        p = __hfma2(wq[r][4 * j + 1], u2h2(xv[j].y), p);
                        p = __hfma2(wq[r][4 * j + 2], u2h2(xv[j].z), p);
                        p = __hfma2(wq[r][4 * j + 3], u2h2(xv[j].w), p);
                        const float2 pf = __half22float2(p);
                        sum = j == 0 ? (pf.x + pf.y) : sum + (pf.x + pf.y);
                    }
                    acc[r][m] = fmaf(sum, dv[r], acc[r][m]);
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
    }
    // block argmax per m over 8 warps x R rows
    __shared__ float sv[8][M];
    __shared__ int   si[8][M];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float bv = -INFINITY;
            int   bi = -1;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                if (n0 + r < N && acc[r][m] > bv) {
                    bv = acc[r][m];
                    bi = n0 + r;
                }
            }
            sv[wid][m] = bv;
            si[wid][m] = bi;
        }
    }
    __syncthreads();
    if (threadIdx.x < M) {
        const int m  = threadIdx.x;
        float     bv = sv[0][m];
        int       bi = si[0][m];
        for (int i = 1; i < 8; ++i) {
            if (sv[i][m] > bv) {
                bv = sv[i][m];
                bi = si[i][m];
            }
        }
        pval[(size_t) m * gridDim.x + blockIdx.x] = bv;
        pidx[(size_t) m * gridDim.x + blockIdx.x] = bi;
    }
}

// one block per m: reduce the per-block maxima, write this rank's exchange slot (other slots zero)
__global__ void __launch_bounds__(256) dhead_reduce(const float * __restrict__ pval, const int * __restrict__ pidx,
                                                   const int nb, const int64_t * __restrict__ gids, half * __restrict__ ex,
                                                   const int rank, const int tp) {
    const int m = blockIdx.x;
    float     bv = -INFINITY;
    int       bi = -1;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        const float v = pval[(size_t) m * nb + i];
        if (v > bv) {
            bv = v;
            bi = pidx[(size_t) m * nb + i];
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffff, bv, o);
        const int   oi = __shfl_xor_sync(0xffffffff, bi, o);
        if (ov > bv || (ov == bv && oi >= 0 && (bi < 0 || oi < bi))) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[8];
    __shared__ int   si[8];
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) {
        sv[wid] = bv;
        si[wid] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 1; i < nw; ++i) {
            if (sv[i] > sv[0] || (sv[i] == sv[0] && si[i] >= 0 && (si[0] < 0 || si[i] < si[0]))) {
                sv[0] = sv[i];
                si[0] = si[i];
            }
        }
        const int64_t g = si[0] >= 0 ? gids[si[0]] : 0;
        half *        e = ex + (size_t) m * tp * 4;
        for (int r = 0; r < tp; ++r) {
            const bool me = r == rank;
            // clamp to the fp16 range (a logit beyond +-65504 would only happen with a broken model)
            e[r * 4 + 0] = __float2half_rn(me ? fminf(fmaxf(sv[0], -65000.0f), 65000.0f) : 0.0f);
            e[r * 4 + 1] = __float2half_rn(me ? (float) (g >> 11) : 0.0f);
            e[r * 4 + 2] = __float2half_rn(me ? (float) (g & 2047) : 0.0f);
            e[r * 4 + 3] = __float2half_rn(0.0f);
        }
    }
}

// weight rows per warp: 2 for one activation row; 4 above, so each lane's activation loads (64 bytes per row
// per 16-byte weight chunk, from L1) serve four weight rows (one row per warp: 1.06 ms per call at M = 6)
#define DHEAD_R(M) ((M) <= 1 ? 2 : 4)

template <int M>
static int launch_m(const half * x, const uint8_t * w, const half * d, int N, int K, float * pval, int * pidx,
                    cudaStream_t st) {
    constexpr int R  = DHEAD_R(M);
    const int     nb = (N + 8 * R - 1) / (8 * R);
    dhead_kernel<M, R><<<nb, 256, 0, st>>>(x, w, d, N, K, pval, pidx);
    return nb;
}

// Number of pass-1 blocks for N rows and M activation rows (size of the pval / pidx scratch per m).
extern "C" int dhead_blocks(int M, int N) {
    const int R = DHEAD_R(M);
    return (N + 8 * R - 1) / (8 * R);
}

// x fp16 [M, K]; w [N, K / 2] bytes; d fp16 [N, K / 32]; gids int64 [N]; pval / pidx scratch [M, dhead_blocks];
// ex fp16 [M, tp, 4] output (this rank's candidate in slot rank).
extern "C" int dhead_argmax(const void * x, const void * w, const void * d, const void * gids, void * pval, void * pidx,
                            void * ex, int M, int N, int K, int rank, int tp, void * stream) {
    if (M < 1 || M > 8 || (K & 31) != 0 || N < 1) {
        return -1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    const half * xh = (const half *) x;
    const uint8_t * wb = (const uint8_t *) w;
    const half * dh = (const half *) d;
    float * pv = (float *) pval;
    int * pi = (int *) pidx;
    int nb = 0;
    switch (M) {
        case 1: nb = launch_m<1>(xh, wb, dh, N, K, pv, pi, st); break;
        case 2: nb = launch_m<2>(xh, wb, dh, N, K, pv, pi, st); break;
        case 3: nb = launch_m<3>(xh, wb, dh, N, K, pv, pi, st); break;
        case 4: nb = launch_m<4>(xh, wb, dh, N, K, pv, pi, st); break;
        case 5: nb = launch_m<5>(xh, wb, dh, N, K, pv, pi, st); break;
        case 6: nb = launch_m<6>(xh, wb, dh, N, K, pv, pi, st); break;
        case 7: nb = launch_m<7>(xh, wb, dh, N, K, pv, pi, st); break;
        case 8: nb = launch_m<8>(xh, wb, dh, N, K, pv, pi, st); break;
    }
    dhead_reduce<<<M, 256, 0, st>>>(pv, pi, nb, (const int64_t *) gids, (half *) ex, rank, tp);
    return (int) cudaGetLastError();
}

// Logits over the 4-bit rows (no argmax): out[m, n] = x[m, :] . W[n, :] as fp16, M <= 16 (DFlash2 candidate head: 7 query
// rows per request). Four rows per warp, so each lane's activation loads serve four weight rows: with one row per warp
// the activation reads (64 bytes per m per 16-byte weight chunk) bound the kernel on L1 at M > 2.
template <int M>
__global__ void __launch_bounds__(256) dhead_logits_kernel(const half * __restrict__ x, const uint8_t * __restrict__ w,
                                                          const half * __restrict__ d, const int N, const int K,
                                                          half * __restrict__ out) {
    constexpr int R = 4;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * 8 + wid) * R;
    if (n0 >= N) {
        return;
    }
    const int KC = K >> 5;
    float acc[R][M];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int m = 0; m < M; ++m) {
            acc[r][m] = 0.0f;
        }
    }
    const int4 * w4[R];
    const half * dr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int nr = min(n0 + r, N - 1);
        w4[r] = reinterpret_cast<const int4 *>(w) + (size_t) nr * KC;
        dr[r] = d + (size_t) nr * KC;
    }
    const uint4 * x4 = reinterpret_cast<const uint4 *>(x);
    for (int c = lane; c < KC; c += 32) {
        int4  wv[R];
        float dv[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            wv[r] = ld_stream(w4[r] + c);
            dv[r] = __half2float(dr[r][c]);
        }
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const uint4 * xm = x4 + ((size_t) m * KC + c) * 4;
            const uint4   x0 = __ldg(xm), x1 = __ldg(xm + 1), x2 = __ldg(xm + 2), x3 = __ldg(xm + 3);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const half2  z  = __float2half2_rn(0.0f);
                const float2 p0 = __half22float2(dot8((uint32_t) wv[r].x, x0, z));
                const float2 p1 = __half22float2(dot8((uint32_t) wv[r].y, x1, z));
                const float2 p2 = __half22float2(dot8((uint32_t) wv[r].z, x2, z));
                const float2 p3 = __half22float2(dot8((uint32_t) wv[r].w, x3, z));
                acc[r][m] = fmaf((p0.x + p0.y) + (p1.x + p1.y) + (p2.x + p2.y) + (p3.x + p3.y), dv[r], acc[r][m]);
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
            if (n0 + r < N) {
#pragma unroll
                for (int m = 0; m < M; ++m) {
                    out[(size_t) m * N + n0 + r] = __float2half_rn(fminf(fmaxf(acc[r][m], -65000.0f), 65000.0f));
                }
            }
        }
    }
}

template <int M>
static void launch_logits(const half * x, const uint8_t * w, const half * d, int N, int K, half * out, cudaStream_t st) {
    const int nb = (N + 31) / 32;   // 8 warps x 4 rows
    dhead_logits_kernel<M><<<nb, 256, 0, st>>>(x, w, d, N, K, out);
}

// x fp16 [M, K], M <= 16; w / d as for dhead_argmax; out fp16 [M, N].
extern "C" int dhead_logits(const void * x, const void * w, const void * d, void * out, int M, int N, int K,
                            void * stream) {
    if (M < 1 || M > 16 || (K & 31) != 0 || N < 1) {
        return -1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    const half * xh = (const half *) x;
    const uint8_t * wb = (const uint8_t *) w;
    const half * dh = (const half *) d;
    half * o = (half *) out;
    switch (M) {
        case 1: launch_logits<1>(xh, wb, dh, N, K, o, st); break;
        case 2: launch_logits<2>(xh, wb, dh, N, K, o, st); break;
        case 3: launch_logits<3>(xh, wb, dh, N, K, o, st); break;
        case 4: launch_logits<4>(xh, wb, dh, N, K, o, st); break;
        case 5: launch_logits<5>(xh, wb, dh, N, K, o, st); break;
        case 6: launch_logits<6>(xh, wb, dh, N, K, o, st); break;
        case 7: launch_logits<7>(xh, wb, dh, N, K, o, st); break;
        case 8: launch_logits<8>(xh, wb, dh, N, K, o, st); break;
        case 9: launch_logits<9>(xh, wb, dh, N, K, o, st); break;
        case 10: launch_logits<10>(xh, wb, dh, N, K, o, st); break;
        case 11: launch_logits<11>(xh, wb, dh, N, K, o, st); break;
        case 12: launch_logits<12>(xh, wb, dh, N, K, o, st); break;
        case 13: launch_logits<13>(xh, wb, dh, N, K, o, st); break;
        case 14: launch_logits<14>(xh, wb, dh, N, K, o, st); break;
        case 15: launch_logits<15>(xh, wb, dh, N, K, o, st); break;
        case 16: launch_logits<16>(xh, wb, dh, N, K, o, st); break;
    }
    return (int) cudaGetLastError();
}
