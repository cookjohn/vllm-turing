// Turing (sm_75) attention for speculative verify / draft decode steps over a long FP8 (e4m3) paged KV cache: q_len
// (1..16) query tokens per request, D = 256, 6 Q heads per KV head (Qwen3.8-27B per TP2 rank), FlashInfer's paged
// layout with any page / token / head strides and 16-token pages (fdec.py hooks the verify / decode wrappers vLLM's
// FlashInfer backend plans; lengths and page ids come from the wrapper's device buffers, so the fixed grid is
// CUDA-graph safe). FlashInfer's FA2 kernel runs a 5-token verify as a 64-row Q tile per block (30 rows used) and
// dequantizes every FP8 KV tile once per warp: at 120k context ~535 us per call (230 GB/s of 616). Here:
//  - one block = (request, KV head, KV split, 16-row tile of the q_len x 6 query rows); its 4 warps take interleaved
//    8-key steps of the split with their own online softmax (lazy rescale: the running max only moves when a row's
//    max grows by more than 2^8), merged in shared memory at the end; fd_merge combines the splits.
//  - K and V go from global memory straight into MMA fragments (no shared-memory staging), FP8 -> f16 by moving the
//    e4m3 bits into the f16 exponent / mantissa (value x 2^-8, folded into the scales): a K row's 16-byte load is the QK
//    B operand under a k permutation Q (staged in shared memory, 4-dim groups reordered) shares; a thread's two V keys
//    are interleaved with prmt into the PV B operand, the output dims permuted (fd_merge writes them in order).
//  - the sibling row tiles of a split are adjacent blocks: their second read of the KV comes from L2.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static constexpr int FD_D    = 256;
static constexpr int FD_G    = 6;      // Q heads per KV head
static constexpr int FD_PAGE = 16;
static constexpr int FD_NW   = 4;
static constexpr int FD_NT   = FD_NW * 32;
static constexpr int FD_PCH  = 256;    // page ids staged in shared memory at a time
static constexpr int FD_MINP = 16;     // minimum pages per split (256 keys)
static constexpr float FD_TH = 8.0f;   // lazy rescale threshold (log2 units): P <= 2^8, exact in f16

struct __align__(16) fd_smem {
    union {
        struct {
            uint4 q[16 * FD_D / 8];   // 16 rows x 256 halves, 4-dim groups as (0 2 1 3), 16-byte units swizzled by row
            int pages[FD_PCH];
        } a;
        float o[32 * 128];            // one warp's O accumulators in register order (cross-warp merge)
    } u;
    float m[FD_NW][16];
    float l[FD_NW][16];
};

static __device__ __forceinline__ void fd_mma(float * d, const uint32_t a0, const uint32_t a1, const uint32_t b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
static __device__ __forceinline__ float fd_ex2(const float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// the low bytes of the two 16-bit lanes (e4m3) -> half2 = values x 2^-8 (the lanes' high bytes are ignored)
static __device__ __forceinline__ uint32_t fd_cvt(const uint32_t u) {
    return ((u << 7) & 0x3F803F80u) | ((u << 8) & 0x80008000u);
}
static __device__ __forceinline__ uint32_t fd_h2u(const half2 h) {
    return *(const uint32_t *) &h;
}
static __device__ __forceinline__ uint4 fd_ld(const uint8_t * p) {
    return __ldg((const uint4 *) p);
}

// KV length of request b from FlashInfer's paged metadata (the last-page length may be <= 0: an upper-bound plan whose
// exact length ends a page earlier, vllm_turing nosync.py) and its split size in pages
static __device__ __forceinline__ void fd_split(const int * indptr, const int * last, const int b, const int S,
                                                int & kv_len, int & npe, int & pps) {
    const int npg = indptr[b + 1] - indptr[b];
    kv_len = npg > 0 ? max((npg - 1) * FD_PAGE + last[b], 0) : 0;
    npe = (kv_len + FD_PAGE - 1) / FD_PAGE;
    pps = max((npe + S - 1) / S, FD_MINP);
}

// grid: ((b * H + kh) * S + s) * MT + mt; partial pidx = ((b * H + kh) * MT + mt) * S + s
static __global__ void __launch_bounds__(FD_NT, 2) fd_attn(
        const half * __restrict__ q, const int64_t qs_t, const int64_t qs_h, const uint8_t * __restrict__ kv,
        const int64_t s_page, const int64_t s_tok, const int64_t s_head, const int64_t v_off,
        const int * __restrict__ indptr, const int * __restrict__ indices, const int * __restrict__ last,
        const int q_len, const int causal, const int H, const int MT, const int S, const float qscale,
        const float oscale, uint32_t * __restrict__ ws_o, float * __restrict__ ws_lse) {
    __shared__ fd_smem sm;
    int bid = blockIdx.x;
    const int mt = bid % MT;
    bid /= MT;
    const int s = bid % S;
    bid /= S;
    const int kh = bid % H, b = bid / H;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;

    int kv_len, npe, pps;
    fd_split(indptr, last, b, S, kv_len, npe, pps);
    const int p0 = s * pps;
    if (p0 >= npe) {
        return;
    }
    const int p1 = min(npe, p0 + pps);
    const int pbeg = indptr[b];
    const int rows = q_len * FD_G;
    const int pidx = ((b * H + kh) * MT + mt) * S + s;

    // Q rows of this tile -> shared memory (rows past q_len x 6: zeros)
    for (int i = threadIdx.x; i < 16 * 32; i += FD_NT) {
        const int r = i / 32, un = i % 32, rg = mt * 16 + r;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (rg < rows) {
            const uint4 x = *(const uint4 *) (q + (int64_t) (b * q_len + rg / FD_G) * qs_t +
                                              (int64_t) (kh * FD_G + rg % FD_G) * qs_h + un * 8);
            v.x = __byte_perm(x.x, x.y, 0x5410);
            v.y = __byte_perm(x.x, x.y, 0x7632);
            v.z = __byte_perm(x.z, x.w, 0x5410);
            v.w = __byte_perm(x.z, x.w, 0x7632);
        }
        sm.u.a.q[r * 32 + (un ^ (r & 1))] = v;
    }

    // causal limits of this thread's rows g and g + 8 (keys >= limit masked); padded rows see everything
    const int qi0 = (mt * 16 + g) / FD_G, qi1 = (mt * 16 + g + 8) / FD_G;
    const int lim0 = causal ? kv_len - (q_len - 1 - qi0) : kv_len;
    const int lim1 = causal ? kv_len - (q_len - 1 - qi1) : kv_len;
    const int limmin = causal ? kv_len - (q_len - 1 - (mt * 16) / FD_G) : kv_len;

    float o[32][4];
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
    }
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    const uint8_t * kvh = kv + kh * s_head;
    const int qsw = g & 1;   // row swizzle of rows g and g + 8

    for (int c0 = p0; c0 < p1; c0 += FD_PCH) {
        const int c1 = min(p1, c0 + FD_PCH);
        __syncthreads();   // the previous chunk's page ids are no longer read (first chunk: Q stores done)
        for (int i = threadIdx.x; i < c1 - c0; i += FD_NT) {
            sm.u.a.pages[i] = indices[pbeg + c0 + i];
        }
        __syncthreads();
        const int kb0 = c0 * FD_PAGE;
        const int nst = (min(c1 * FD_PAGE, kv_len) - kb0 + 7) / 8;
        uint4 kr[4], vr[4];
        int st = warp;
        if (st < nst) {
            const uint8_t * pb = kvh + (int64_t) sm.u.a.pages[st / 2] * s_page + (int64_t) ((st & 1) * 8) * s_tok;
            const uint8_t * kp = pb + g * s_tok + 16 * t;
            const uint8_t * vp = pb + v_off + (2 * t) * s_tok + 16 * g;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                kr[c] = fd_ld(kp + 64 * c);
            }
            vr[0] = fd_ld(vp);
            vr[1] = fd_ld(vp + 128);
            vr[2] = fd_ld(vp + s_tok);
            vr[3] = fd_ld(vp + s_tok + 128);
        }
        for (; st < nst; st += FD_NW) {
            const int nx = st + FD_NW;
            const uint8_t * pbn = nx < nst ? kvh + (int64_t) sm.u.a.pages[nx / 2] * s_page +
                                                 (int64_t) ((nx & 1) * 8) * s_tok : nullptr;
            // S = Q K^T for the step's 8 keys (k permuted: thread t's dims 64c + 16t + 4w + {0,2} / {1,3})
            float sc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 qa = sm.u.a.q[g * 32 + ((8 * c + 2 * t) ^ qsw)];
                const uint4 qb = sm.u.a.q[g * 32 + ((8 * c + 2 * t + 1) ^ qsw)];
                const uint4 qc = sm.u.a.q[(g + 8) * 32 + ((8 * c + 2 * t) ^ qsw)];
                const uint4 qd = sm.u.a.q[(g + 8) * 32 + ((8 * c + 2 * t + 1) ^ qsw)];
                const uint32_t A0[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
                const uint32_t A1[8] = {qc.x, qc.y, qc.z, qc.w, qd.x, qd.y, qd.z, qd.w};
                const uint32_t kw[4] = {kr[c].x, kr[c].y, kr[c].z, kr[c].w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    fd_mma(sc, A0[2 * w], A1[2 * w], fd_cvt(kw[w]));
                    fd_mma(sc, A0[2 * w + 1], A1[2 * w + 1], fd_cvt(kw[w] >> 8));
                }
            }
            if (pbn) {   // next step's K (registers free now)
                const uint8_t * kp = pbn + g * s_tok + 16 * t;
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    kr[c] = fd_ld(kp + 64 * c);
                }
            }
            // online softmax over columns 2t, 2t + 1 (keys kb, kb + 1) of rows g, g + 8
            const int kb = kb0 + st * 8 + 2 * t;
            float x0 = sc[0] * qscale, x1 = sc[1] * qscale, x2 = sc[2] * qscale, x3 = sc[3] * qscale;
            if (kb0 + st * 8 + 8 > limmin) {
                x0 = kb < lim0 ? x0 : -INFINITY;
                x1 = kb + 1 < lim0 ? x1 : -INFINITY;
                x2 = kb < lim1 ? x2 : -INFINITY;
                x3 = kb + 1 < lim1 ? x3 : -INFINITY;
            }
            float r0 = fmaxf(x0, x1), r1 = fmaxf(x2, x3);
            r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFF, r0, 1));
            r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFF, r1, 1));
            r0 = fmaxf(r0, __shfl_xor_sync(0xFFFFFFFF, r0, 2));
            r1 = fmaxf(r1, __shfl_xor_sync(0xFFFFFFFF, r1, 2));
            if (__any_sync(0xFFFFFFFF, r0 > m0 + FD_TH || r1 > m1 + FD_TH)) {
                const float n0 = fmaxf(m0, r0), n1 = fmaxf(m1, r1);
                const float f0 = fd_ex2(m0 - n0), f1 = fd_ex2(m1 - n1);
                m0 = n0;
                m1 = n1;
                l0 *= f0;
                l1 *= f1;
#pragma unroll
                for (int nt = 0; nt < 32; ++nt) {
                    o[nt][0] *= f0;
                    o[nt][1] *= f0;
                    o[nt][2] *= f1;
                    o[nt][3] *= f1;
                }
            }
            const float e0 = fd_ex2(x0 - m0), e1 = fd_ex2(x1 - m0), e2 = fd_ex2(x2 - m1), e3 = fd_ex2(x3 - m1);
            l0 += e0 + e1;
            l1 += e2 + e3;
            const uint32_t pa = fd_h2u(__floats2half2_rn(e0, e1)), pb = fd_h2u(__floats2half2_rn(e2, e3));
            // O += P V: n-tile (h, j) column n = dim 128h + 16n + j; V keys 2t (vr[h]) and 2t + 1 (vr[2 + h])
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint32_t xw[4] = {vr[h].x, vr[h].y, vr[h].z, vr[h].w};
                const uint32_t yw[4] = {vr[2 + h].x, vr[2 + h].y, vr[2 + h].z, vr[2 + h].w};
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    const uint32_t u = __byte_perm(xw[w], yw[w], 0x5410), v = __byte_perm(xw[w], yw[w], 0x7632);
                    fd_mma(o[h * 16 + 4 * w + 0], pa, pb, fd_cvt(u));
                    fd_mma(o[h * 16 + 4 * w + 1], pa, pb, fd_cvt(u >> 8));
                    fd_mma(o[h * 16 + 4 * w + 2], pa, pb, fd_cvt(v));
                    fd_mma(o[h * 16 + 4 * w + 3], pa, pb, fd_cvt(v >> 8));
                }
            }
            if (pbn) {   // next step's V
                const uint8_t * vp = pbn + v_off + (2 * t) * s_tok + 16 * g;
                vr[0] = fd_ld(vp);
                vr[1] = fd_ld(vp + 128);
                vr[2] = fd_ld(vp + s_tok);
                vr[3] = fd_ld(vp + s_tok + 128);
            }
        }
    }

    // merge the 4 warps: common max per row, O summed through shared memory (register order: conflict-free)
    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 1);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 1);
    l0 += __shfl_xor_sync(0xFFFFFFFF, l0, 2);
    l1 += __shfl_xor_sync(0xFFFFFFFF, l1, 2);
    __syncthreads();   // Q / page ids no longer read: the union is reused
    if (t == 0) {
        sm.m[warp][g] = m0;
        sm.m[warp][g + 8] = m1;
    }
    __syncthreads();
    float M0 = sm.m[0][g], M1 = sm.m[0][g + 8];
#pragma unroll
    for (int w = 1; w < FD_NW; ++w) {
        M0 = fmaxf(M0, sm.m[w][g]);
        M1 = fmaxf(M1, sm.m[w][g + 8]);
    }
    const float f0 = fd_ex2(m0 - M0), f1 = fd_ex2(m1 - M1);
    if (t == 0) {
        sm.l[warp][g] = l0 * f0;
        sm.l[warp][g + 8] = l1 * f1;
    }
#pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        o[nt][0] *= f0;
        o[nt][1] *= f0;
        o[nt][2] *= f1;
        o[nt][3] *= f1;
    }
    float * buf = sm.u.o;
    if (warp == FD_NW - 1) {
#pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                buf[(nt * 4 + e) * 32 + lane] = o[nt][e];
            }
        }
    }
    __syncthreads();
    for (int w = FD_NW - 2; w >= 0; --w) {
        if (warp == w) {
#pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    o[nt][e] += buf[(nt * 4 + e) * 32 + lane];
                    if (w > 0) {
                        buf[(nt * 4 + e) * 32 + lane] = o[nt][e];
                    }
                }
            }
        }
        __syncthreads();
    }
    if (warp == 0) {
        float L0 = 0.0f, L1 = 0.0f;
#pragma unroll
        for (int w = 0; w < FD_NW; ++w) {
            L0 += sm.l[w][g];
            L1 += sm.l[w][g + 8];
        }
        const float i0 = L0 > 0.0f ? oscale / L0 : 0.0f, i1 = L1 > 0.0f ? oscale / L1 : 0.0f;
        uint32_t * wo = ws_o + (int64_t) pidx * 2048;
#pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            wo[(nt * 2 + 0) * 32 + lane] = fd_h2u(__floats2half2_rn(o[nt][0] * i0, o[nt][1] * i0));
            wo[(nt * 2 + 1) * 32 + lane] = fd_h2u(__floats2half2_rn(o[nt][2] * i1, o[nt][3] * i1));
        }
        if (t == 0) {
            ws_lse[(int64_t) pidx * 16 + g] = L0 > 0.0f ? M0 + __log2f(L0) : -INFINITY;
            ws_lse[(int64_t) pidx * 16 + g + 8] = L1 > 0.0f ? M1 + __log2f(L1) : -INFINITY;
        }
    }
}

// combine a request's splits; half2 element e of a partial: n-tile nt = e / 64 (h = nt / 16, j = nt % 16), row half
// (e / 32) % 2, lane e % 32 (g = lane / 4, t = lane % 4) -> row 8 * half + g, dims 128h + 32t + j and + 16
static __global__ void __launch_bounds__(256) fd_merge(
        const uint32_t * __restrict__ ws_o, const float * __restrict__ ws_lse, const int * __restrict__ indptr,
        const int * __restrict__ last, const int q_len, const int H, const int MT, const int S,
        half * __restrict__ out, const int64_t os_t, const int64_t os_h) {
    int bid = blockIdx.x;
    const int part = bid % 8;
    bid /= 8;
    const int mt = bid % MT;
    bid /= MT;
    const int kh = bid % H, b = bid / H;
    const int e = part * 256 + threadIdx.x;
    const int lane = e % 32, rh = (e / 32) % 2, nt = e / 64, g = lane / 4, t = lane % 4;
    const int row = rh * 8 + g, rg = mt * 16 + row;
    if (rg >= q_len * FD_G) {
        return;
    }
    int kv_len, npe, pps;
    fd_split(indptr, last, b, S, kv_len, npe, pps);
    const int ns = (npe + pps - 1) / pps;
    const int64_t p0 = ((int64_t) (b * H + kh) * MT + mt) * S;
    float M = -INFINITY;
    for (int i = 0; i < ns; ++i) {
        M = fmaxf(M, ws_lse[(p0 + i) * 16 + row]);
    }
    float a0 = 0.0f, a1 = 0.0f, ws = 0.0f;
    if (M > -INFINITY) {
        for (int i = 0; i < ns; ++i) {
            const float w = fd_ex2(ws_lse[(p0 + i) * 16 + row] - M);
            const uint32_t u = ws_o[(p0 + i) * 2048 + e];
            const float2 f = __half22float2(*(const half2 *) &u);
            a0 += w * f.x;
            a1 += w * f.y;
            ws += w;
        }
    }
    const float inv = ws > 0.0f ? 1.0f / ws : 0.0f;
    const int d0 = 128 * (nt / 16) + 32 * t + nt % 16;
    half * op = out + (int64_t) (b * q_len + rg / FD_G) * os_t + (int64_t) (kh * FD_G + rg % FD_G) * os_h;
    op[d0] = __float2half_rn(a0 * inv);
    op[d0 + 16] = __float2half_rn(a1 * inv);
}

// q [B * q_len, H * 6, 256] f16 (strides qs_t, qs_h; dims contiguous); kv: K plane base, V plane at + v_off bytes; FP8
// element (page p, slot s, head h) at p * s_page + s * s_tok + h * s_head bytes; indptr [B + 1] / indices / last [B]:
// FlashInfer's paged metadata (16-token pages); query token j of request b sees kv_len - (q_len - 1 - j) keys (causal)
// or all; qscale = softmax scale x k_scale x 256 x log2(e), oscale = v_scale x 256; S splits per (request, KV head,
// row tile); ws_o: B * H * MT * S * 2048 u32, ws_lse: B * H * MT * S * 16 f32; out [B * q_len, H * 6, 256] f16.
extern "C" int fd_run(const void * q, int64_t qs_t, int64_t qs_h, const void * kv, int64_t s_page, int64_t s_tok,
                      int64_t s_head, int64_t v_off, const int * indptr, const int * indices, const int * last, int B,
                      int q_len, int causal, int H, int S, float qscale, float oscale, void * ws_o, void * ws_lse,
                      void * out, int64_t os_t, int64_t os_h, cudaStream_t stream) {
    if (B <= 0 || q_len <= 0 || q_len > 16 || S <= 0) {
        return -1;
    }
    const int MT = (q_len * FD_G + 15) / 16;
    fd_attn<<<B * H * S * MT, FD_NT, 0, stream>>>((const half *) q, qs_t, qs_h, (const uint8_t *) kv, s_page, s_tok,
                                                    s_head, v_off, indptr, indices, last, q_len, causal, H, MT, S,
                                                    qscale, oscale, (uint32_t *) ws_o, (float *) ws_lse);
    fd_merge<<<B * H * MT * 8, 256, 0, stream>>>((const uint32_t *) ws_o, (const float *) ws_lse, indptr, last, q_len,
                                                 H, MT, S, (half *) out, os_t, os_h);
    return (int) cudaGetLastError();
}
