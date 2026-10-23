// Turing (sm_75) causal prefill attention for FP8 (e4m3) paged KV, D = 256, 6 Q heads per KV head (Qwen3.8-27B per
// TP2 rank: 12 Q heads, 2 KV heads). Port of the int8-KQ kernel of our llama.cpp fork (ggml-cuda/fattn-int8.cu) to
// vLLM's paged cache (fiattn8.py hooks FlashInfer's prefill wrapper):
//  - KQ on int8 tensor cores (mma.m8n8k16.s8, 4x the rate of the f16 MMA with f32 accumulation FlashInfer uses on
//    GeForce Turing): K converted per KV chunk from FP8 to int8 with a per-row scale (fa8_convert_k, gathers the
//    request's pages), Q quantized per row in the kernel prologue; the scores are exact int32 dot products, scaled in f32.
//  - PV with the f16 MMA, f16 accumulation (V converted per chunk from FP8 to f16, fa8_convert_v).
//  - one block = one KV head x 16 tokens x its 6 Q heads = 96 query columns, 12 warps, warp-specialized (4 KQ warps:
//    MMAs + online softmax; 8 PV warps: PV + all K/V staging), named-barrier double buffering (see the llama.cpp file).
//  - causal mask from positions (query token i of the chunk sits at kv position q_pos0 + i): tiles past a block's last
//    visible position are skipped, partial tiles are masked per element.
//  - KV chunks of up to 16384 positions bound the converted K/V size; a row's partial results are merged in the kernel
//    epilogue into a running (unnormalized sum, max, sum) in f32, and the chunk holding the row block's last visible
//    position writes the normalized f16 output.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

static constexpr int FI_D   = 256;
static constexpr int FI_G   = 6;             // Q heads per KV head
static constexpr int FI_TPB = 16;            // tokens per block
static constexpr int FI_NC  = FI_TPB * FI_G; // 96 query columns
static constexpr int FI_NW  = 12;
static constexpr int FI_NT  = FI_NW * 32;
static constexpr int FI_BKV = 32;

struct __align__(16) fi_smem {
    int8_t k[2][FI_BKV * FI_D];              // int8 K tiles (KQ group), 16-byte chunks XOR-swizzled by row
    half   v[2][FI_BKV * FI_D];              // f16 V tiles (PV group), swizzled
    half   p[2][FI_NC * FI_BKV];             // P [col][kv], 16-byte chunks swizzled by (col/2)&3, double-buffered
    float  ksc[2][FI_BKV];
    float  alpha[2][FI_NC];
    float  ml[2][FI_NC];                     // final max (log2 units), sum per column
    float  prev[2][FI_NC];                   // merge: running max (log2 units), sum of earlier chunks
};

static __device__ __forceinline__ void fi_bar_sync(const int id, const int n) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory");
}
static __device__ __forceinline__ void fi_bar_arrive(const int id, const int n) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory");
}
static __device__ __forceinline__ void fi_ldsm_x4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
static __device__ __forceinline__ void fi_ldsm_x4_t(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
static __device__ __forceinline__ void fi_imma(int & d0, int & d1, uint32_t a, uint32_t b) {
    asm("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};" : "+r"(d0), "+r"(d1) : "r"(a), "r"(b));
}
static __device__ __forceinline__ void fi_hmma(uint32_t & d0, uint32_t & d1, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};" : "+r"(d0), "+r"(d1) : "r"(a0), "r"(a1), "r"(b0));
}
static __device__ __forceinline__ float fi_ex2(const float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 8 e4m3 bytes -> 8 floats
static __device__ __forceinline__ void fp8x8_to_float(const uint2 u, float * v) {
    const uint32_t w[2] = {u.x, u.y};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __nv_fp8_storage_t b = (__nv_fp8_storage_t) ((w[i / 4] >> (8 * (i % 4))) & 0xFF);
        const __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
        v[i] = __half2float(*(const half *) &h);
    }
}

// paged FP8 cache element (page p, slot s, head kh) at base + p*s_page + s*s_tok + kh*s_head (bytes); the K and V planes
// are separate base pointers. Rows over (kh, kv) for kv positions kv0 .. kv0+len of one request (pages[]).
// K: int8 row with one scale (amax / 127, after k_scale); one warp per (kh, kv) row, lane = 8 values
static __global__ void fa8_convert_k(const uint8_t * __restrict__ base, const int64_t s_page, const int64_t s_tok,
                                     const int64_t s_head, const int page_size, const int * __restrict__ pages,
                                     const int64_t kv0, const int len, const int64_t pos_max, const int nkv_heads,
                                     const float k_scale, int8_t * __restrict__ K8, float * __restrict__ KS) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // over nkv_heads * len
    if (row >= (int64_t) nkv_heads * len) {
        return;
    }
    const int kh = row / len, kv = row % len, lane = threadIdx.x % 32;
    const int64_t pos = min(kv0 + kv, pos_max);   // padding of the last chunk
    const uint8_t * src = base + (int64_t) pages[pos / page_size] * s_page + (pos % page_size) * s_tok + kh * s_head;
    float v[8];
    fp8x8_to_float(*(const uint2 *) (src + 8 * lane), v);
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v[i] *= k_scale;
        amax = fmaxf(amax, fabsf(v[i]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    }
    const float s  = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float is = 1.0f / s;
    uint32_t lo = 0, hi = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        lo |= (uint32_t) (__float2int_rn(v[i]     * is) & 0xFF) << (8 * i);
        hi |= (uint32_t) (__float2int_rn(v[4 + i] * is) & 0xFF) << (8 * i);
    }
    *(uint2 *) (K8 + row * FI_D + 8 * lane) = make_uint2(lo, hi);
    if (lane == 0) {
        KS[row] = s;
    }
}

// V: f16 row (after v_scale); one warp per (kh, kv) row
static __global__ void fa8_convert_v(const uint8_t * __restrict__ base, const int64_t s_page, const int64_t s_tok,
                                     const int64_t s_head, const int page_size, const int * __restrict__ pages,
                                     const int64_t kv0, const int len, const int64_t pos_max, const int nkv_heads,
                                     const float v_scale, half * __restrict__ V16) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= (int64_t) nkv_heads * len) {
        return;
    }
    const int kh = row / len, kv = row % len, lane = threadIdx.x % 32;
    const int64_t pos = min(kv0 + kv, pos_max);   // padding of the last chunk
    const uint8_t * src = base + (int64_t) pages[pos / page_size] * s_page + (pos % page_size) * s_tok + kh * s_head;
    float v[8];
    fp8x8_to_float(*(const uint2 *) (src + 8 * lane), v);
    half h[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        h[i] = __float2half_rn(v[i] * v_scale);
    }
    *(uint4 *) (V16 + row * FI_D + 8 * lane) = *(const uint4 *) h;
}

// Q element (tok, h, d) at Q + tok*qs_t + h*qs_h + d (f16); K8/KS/V16 head-major [kh][len][D] for kv positions
// kv0 .. kv0+len; query token i at kv position q_pos0 + i (causal: sees kv <= that). Running merge state per row
// (tok*NH + h): A [D] f32 (unnormalized), MS (max in log2 units, sum); O [tok][NH][D] f16 output (row stride os_t).
// first: no earlier chunk for any row (kv0 == 0).
static __global__ void __launch_bounds__(FI_NT, 1) fa8_kernel(
        const half * __restrict__ Q, const int64_t qs_t, const int64_t qs_h,
        const int8_t * __restrict__ K8, const float * __restrict__ KS, const half * __restrict__ V16,
        const int64_t kv0, const int64_t q_pos0, float * __restrict__ A, float2 * __restrict__ MS,
        half * __restrict__ O, const int64_t os_t, const int ntok, const int NH, const int len, const float scale) {
    extern __shared__ __align__(16) uint8_t fi_raw[];
    fi_smem & sm = *(fi_smem *) fi_raw;
    constexpr int NKQ = 4, NPV = FI_NW - NKQ;    // warps per group
    constexpr int CPW = FI_NC / NKQ;             // 24 columns per KQ warp
    constexpr int NTN = CPW / 8;                 // 3 n-tiles
    constexpr int NPVT = NPV * 32;
    constexpr int B_FULL = 3, B_EMPTY = 5, B_MMA = 7;
    const int tid = threadIdx.x, lane = tid % 32, w = tid / 32;
    const int g = lane / 4, t = lane % 4;
    const int kh = blockIdx.y, t0 = blockIdx.x * FI_TPB;
    const int last_tok = min(t0 + FI_TPB, ntok) - 1;
    // causal: tiles holding a kv position visible to the block's last token; positions past the chunk end are later
    // chunks, positions before kv0 earlier ones
    const int64_t lim = q_pos0 + last_tok;                   // last visible kv position of the block
    if (lim < kv0) {
        return;                                              // the whole chunk is in the block's future
    }
    const int ntile = (int) min((int64_t) (len / FI_BKV), (lim - kv0) / FI_BKV + 1);
    const bool final_chunk = lim < kv0 + len;                // this chunk holds the block's last visible position
    const bool first = kv0 == 0;
    const int64_t full_lim = q_pos0 + t0;                    // tiles ending at or before this: no mask for any token

    const int8_t * kbase = K8 + (int64_t) kh * len * FI_D;
    const half   * vbase = V16 + (int64_t) kh * len * FI_D;
    const float  * sbase = KS + (int64_t) kh * len;
    auto kld = [&](const int j, const int i) -> int4 {     // int4 i of tile j: row i/16, chunk i%16
        return *(const int4 *) (kbase + (int64_t) (j*FI_BKV + i/16) * FI_D + 16*(i % 16));
    };
    auto kst = [&](const int buf, const int i, const int4 v) {
        const int row = i / 16, ch = i % 16;
        *(int4 *) (sm.k[buf] + row*FI_D + 16*(ch ^ (row & 7))) = v;
    };
    auto vld = [&](const int j, const int i) -> int4 {     // row i/32, chunk i%32
        return *(const int4 *) (vbase + (int64_t) (j*FI_BKV + i/32) * FI_D + 8*(i % 32));
    };
    auto vst = [&](const int buf, const int i, const int4 v) {
        const int row = i / 32, ch = i % 32;
        *(int4 *) (sm.v[buf] + row*FI_D + 8*(ch ^ (row & 7))) = v;
    };
    // prologue: K(0), K(1), V(0) by all threads; the merge state of earlier chunks
    for (int i = tid; i < FI_BKV * 16; i += FI_NT) {
        kst(0, i, kld(0, i));
        if (ntile > 1) {
            kst(1, i, kld(1, i));
        }
    }
    for (int i = tid; i < FI_BKV * 32; i += FI_NT) {
        vst(0, i, vld(0, i));
    }
    if (tid < FI_BKV) {
        sm.ksc[0][tid] = sbase[tid];
        if (ntile > 1) {
            sm.ksc[1][tid] = sbase[FI_BKV + tid];
        }
    }
    if (tid < FI_NC) {
        const int tok = min(t0 + tid / FI_G, ntok - 1);
        const float2 ms = first ? make_float2(-INFINITY, 0.0f) : MS[(int64_t) tok * NH + kh*FI_G + tid % FI_G];
        sm.prev[0][tid] = ms.x;
        sm.prev[1][tid] = ms.y;
    }
    __syncthreads();

    if (w < NKQ) {
        // ================= KQ group: MMAs + softmax =================
        const int ws = w;
        uint32_t qB[NTN][16];
        float qsc_own[NTN];
#pragma unroll
        for (int nt = 0; nt < NTN; ++nt) {
            const int c = CPW*ws + 8*nt + g;
            const int tok = min(t0 + c / FI_G, ntok - 1), h = kh*FI_G + c % FI_G;
            const half * q = Q + tok*qs_t + h*qs_h;
            float4 qv[16];
            float amax = 0.0f;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const uint2 u = *(const uint2 *) (q + 16*ks + 4*t);
                const float2 a = __half22float2(*(const half2 *) &u.x), b = __half22float2(*(const half2 *) &u.y);
                qv[ks] = make_float4(a.x, a.y, b.x, b.y);
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y))));
            }
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, 2));
            const float qs = amax > 0.0f ? amax / 127.0f : 1.0f;
            const float iq = 1.0f / qs;
#pragma unroll
            for (int ks = 0; ks < 16; ++ks) {
                const int x0 = __float2int_rn(qv[ks].x * iq), x1 = __float2int_rn(qv[ks].y * iq);
                const int x2 = __float2int_rn(qv[ks].z * iq), x3 = __float2int_rn(qv[ks].w * iq);
                qB[nt][ks] = (uint32_t) (x0 & 0xFF) | ((uint32_t) (x1 & 0xFF) << 8) | ((uint32_t) (x2 & 0xFF) << 16) | ((uint32_t) (x3 & 0xFF) << 24);
            }
            qsc_own[nt] = qs * scale * 1.4426950408889634f;   // softmax scale and log2(e) folded in
        }
        float qsc[NTN], m[NTN], l[NTN];
        int64_t cpos[NTN];                                    // kv position of each column's token
#pragma unroll
        for (int nt = 0; nt < NTN; ++nt) {
            qsc[nt]  = qsc_own[nt];
            cpos[nt] = q_pos0 + min(t0 + (CPW*ws + 8*nt + g) / FI_G, ntok - 1);
            m[nt]    = -INFINITY;
            l[nt]    = 0.0f;
        }

        for (int j = 0; j < ntile; ++j) {
            const int buf  = j & 1;
            if (j >= 2) {
                fi_bar_sync(B_EMPTY + buf, FI_NT);           // P[buf] free and K(j) in K[buf]
            }
            const bool partial = kv0 + (int64_t) j*FI_BKV + FI_BKV - 1 > full_lim;
            {
                int s[4][NTN][2];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) for (int nt = 0; nt < NTN; ++nt) { s[mt][nt][0] = 0; s[mt][nt][1] = 0; }
                const int8_t * kb = sm.k[buf];
#pragma unroll
                for (int kq = 0; kq < 4; ++kq) {
                    uint32_t a[4][4];
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        const int row = 8*mt + (lane % 8);
                        fi_ldsm_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3], kb + row*FI_D + 16*((4*kq + lane/8) ^ (row & 7)));
                    }
#pragma unroll
                    for (int x = 0; x < 4; ++x)
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                    for (int nt = 0; nt < NTN; ++nt) {
                        fi_imma(s[mt][nt][0], s[mt][nt][1], qB[nt][4*kq + x], a[mt][x]);
                    }
                }
                if (j >= 1) {
                    fi_bar_arrive(B_MMA, FI_NT);                 // KQ MMAs of tile j issued: PV(j-1) may run
                }
                half * pb = sm.p[buf];
                float2 ksv[4];
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    ksv[mt] = *(const float2 *) &sm.ksc[buf][8*mt + 2*t];
                }
                const int64_t tpos = kv0 + (int64_t) j*FI_BKV + 2*t;   // kv position of element (mt = 0, e = 0)
#pragma unroll
                for (int nt = 0; nt < NTN; ++nt) {
                    float sf[4][2];
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        // exact int -> float: 1.5*2^23 + s is an exact float for |s| < 2^22 (256 int8 products)
                        sf[mt][0] = (__int_as_float(s[mt][nt][0] + 0x4B400000) - 12582912.0f) * (qsc[nt] * ksv[mt].x);
                        sf[mt][1] = (__int_as_float(s[mt][nt][1] + 0x4B400000) - 12582912.0f) * (qsc[nt] * ksv[mt].y);
                    }
                    if (partial) {
#pragma unroll
                        for (int mt = 0; mt < 4; ++mt) {
                            if (tpos + 8*mt + 0 > cpos[nt]) sf[mt][0] = -INFINITY;
                            if (tpos + 8*mt + 1 > cpos[nt]) sf[mt][1] = -INFINITY;
                        }
                    }
                    float mx = fmaxf(fmaxf(sf[0][0], sf[0][1]), fmaxf(sf[1][0], sf[1][1]));
                    mx = fmaxf(mx, fmaxf(fmaxf(sf[2][0], sf[2][1]), fmaxf(sf[3][0], sf[3][1])));
                    mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 1));
                    mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, 2));
                    const float mn = fmaxf(m[nt], mx);
                    const float rb = mn == -INFINITY ? 0.0f : mn;   // column fully masked so far
                    const float al = fi_ex2(m[nt] - rb);
                    m[nt] = mn;
                    float ps = 0.0f;
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        sf[mt][0] = fi_ex2(sf[mt][0] - rb);
                        sf[mt][1] = fi_ex2(sf[mt][1] - rb);
                        ps += sf[mt][0] + sf[mt][1];
                    }
                    l[nt] = l[nt]*al + ps;
                    const int c = CPW*ws + 8*nt + g;
                    if (t == 0) {
                        sm.alpha[buf][c] = al;
                    }
#pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        *(half2 *) (pb + c*FI_BKV + 8*(mt ^ ((c >> 1) & 3)) + 2*t) = __floats2half2_rn(sf[mt][0], sf[mt][1]);
                    }
                }
            }
            fi_bar_arrive(B_FULL + buf, FI_NT);              // P[buf] ready, K[buf] no longer read
        }
#pragma unroll
        for (int nt = 0; nt < NTN; ++nt) {
            float ls = l[nt];
            ls += __shfl_xor_sync(0xFFFFFFFF, ls, 1);
            ls += __shfl_xor_sync(0xFFFFFFFF, ls, 2);
            if (t == 0) {
                sm.ml[0][CPW*ws + 8*nt + g] = m[nt];
                sm.ml[1][CPW*ws + 8*nt + g] = ls;
            }
        }
        __syncthreads();
        // running max / sum of the rows (written by the KQ group: the PV group read the old values from sm.prev)
        if (!final_chunk && tid < FI_NC && t0 + tid / FI_G < ntok) {
            const float m0 = sm.prev[0][tid], s0 = sm.prev[1][tid];
            const float mc = sm.ml[0][tid], lc = sm.ml[1][tid];
            const float m1 = fmaxf(m0, mc);
            const float a  = s0 > 0.0f ? fi_ex2(m0 - m1) : 0.0f;
            const float b  = lc > 0.0f ? fi_ex2(mc - m1) : 0.0f;
            MS[(int64_t) (t0 + tid / FI_G) * NH + kh*FI_G + tid % FI_G] = make_float2(m1, s0*a + lc*b);
        }
    } else {
        // ================= PV group: PV + all K/V staging =================
        const int pw = w - NKQ, gtid = tid - NKQ*32;
        const int cg = pw / 4, dg = pw % 4;      // 48 columns (3 m-tiles) x 64 dims (8 n-tiles)
        uint32_t o[3][8][2];
#pragma unroll
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 8; ++j) { o[i][j][0] = 0u; o[i][j][1] = 0u; }
        // (the K/V loads stay right before their staging stores: issuing them one iteration earlier keeps 24 more
        // registers live through the PV MMAs: 168 registers with spills, 2% slower in interleaved A/B runs)
        int4 kr[2], vr[4];
        float ksr = 0.0f;
        for (int j = 0; j < ntile; ++j) {
            const int buf = j & 1;
            fi_bar_sync(B_FULL + buf, FI_NT);                // P(j) ready; V(j) stores of all PV warps visible
            {
                const int jk = min(j + 2, ntile - 1), jv = min(j + 1, ntile - 1);
#pragma unroll
                for (int r = 0; r < 2; ++r) {
                    kr[r] = kld(jk, gtid + r*NPVT);
                }
                ksr = sbase[jk*FI_BKV + gtid % FI_BKV];
#pragma unroll
                for (int r = 0; r < 4; ++r) {
                    vr[r] = vld(jv, gtid + r*NPVT);
                }
            }
            {
                float alf[3][2];
                bool same = true;
#pragma unroll
                for (int mt = 0; mt < 3; ++mt) for (int hh = 0; hh < 2; ++hh) {
                    alf[mt][hh] = sm.alpha[buf][48*cg + 16*mt + 8*hh + g];
                    same &= alf[mt][hh] == 1.0f;
                }
                if (!__all_sync(0xFFFFFFFF, same)) {
#pragma unroll
                    for (int mt = 0; mt < 3; ++mt) {
                        const half2 a0 = __float2half2_rn(alf[mt][0]), a1 = __float2half2_rn(alf[mt][1]);
#pragma unroll
                        for (int nt = 0; nt < 8; ++nt) {
                            *(half2 *) &o[mt][nt][0] = __hmul2(*(half2 *) &o[mt][nt][0], a0);
                            *(half2 *) &o[mt][nt][1] = __hmul2(*(half2 *) &o[mt][nt][1], a1);
                        }
                    }
                }
                if (j + 2 < ntile) {
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        kst(buf, gtid + r*NPVT, kr[r]);
                    }
                    if (gtid < FI_BKV) {
                        sm.ksc[buf][gtid] = ksr;
                    }
                }
                if (j + 1 < ntile) {
#pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        vst((j + 1) & 1, gtid + r*NPVT, vr[r]);
                    }
                }
                if (j + 1 < ntile) {
                    fi_bar_sync(B_MMA, FI_NT);                   // run the PV MMAs during the KQ softmax of tile j+1
                }
                const half * pb = sm.p[buf];
                const half * vb = sm.v[buf];
                const int jm = lane / 8, i8 = lane % 8;
#pragma unroll
                for (int k16 = 0; k16 < 2; ++k16) {
                    uint32_t pa[3][4];
#pragma unroll
                    for (int mt = 0; mt < 3; ++mt) {
                        const int c = 48*cg + 16*mt + 8*(jm % 2) + i8, ch = 2*k16 + jm/2;
                        fi_ldsm_x4(pa[mt][0], pa[mt][1], pa[mt][2], pa[mt][3], pb + c*FI_BKV + 8*(ch ^ ((c >> 1) & 3)));
                    }
                    uint32_t b[4][4];   // (k 0-7, n 0-7) (k 8-15, n 0-7) (k 0-7, n 8-15) (k 8-15, n 8-15)
#pragma unroll
                    for (int n16 = 0; n16 < 4; ++n16) {
                        const int row = 16*k16 + 8*(jm % 2) + i8;
                        const int ch  = 8*dg + 2*n16 + jm / 2;
                        fi_ldsm_x4_t(b[n16][0], b[n16][1], b[n16][2], b[n16][3], vb + row*FI_D + 8*(ch ^ (row & 7)));
                    }
#pragma unroll
                    for (int st = 0; st < 2; ++st)
#pragma unroll
                    for (int mt = 0; mt < 3; ++mt)
#pragma unroll
                    for (int n16 = 0; n16 < 4; ++n16) {
                        fi_hmma(o[mt][2*n16 + 0][0], o[mt][2*n16 + 0][1], pa[mt][2*st], pa[mt][2*st + 1], b[n16][st]);
                        fi_hmma(o[mt][2*n16 + 1][0], o[mt][2*n16 + 1][1], pa[mt][2*st], pa[mt][2*st + 1], b[n16][2 + st]);
                    }
                }
            }
            if (j + 2 < ntile) {
                fi_bar_arrive(B_EMPTY + buf, FI_NT);         // P[buf] free and K(j+2) in K[buf]
            }
        }
        __syncthreads();
        // epilogue: merge with the earlier chunks' running state (A, sm.prev) and write the running state or the
        // normalized f16 output (final chunk of the row block)
#pragma unroll
        for (int mt = 0; mt < 3; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int c = 48*cg + 16*mt + 8*hh + g;
            const int tok = t0 + c / FI_G;
            if (tok >= ntok) {
                continue;
            }
            const int64_t row = (int64_t) tok * NH + kh*FI_G + c % FI_G;
            const float mc = sm.ml[0][c], lc = sm.ml[1][c];
            const float m0 = sm.prev[0][c], s0 = sm.prev[1][c];
            const float m1 = fmaxf(m0, mc);
            const float a  = s0 > 0.0f ? fi_ex2(m0 - m1) : 0.0f;    // first chunk: s0 = 0
            const float b  = lc > 0.0f ? fi_ex2(mc - m1) : 0.0f;
            const float s1 = s0*a + lc*b;
            float * arow = A + row * FI_D + 64*dg + 2*t;
            if (final_chunk) {
                const float il = s1 > 0.0f ? 1.0f / s1 : 0.0f;
                half * orow = O + (int64_t) tok * os_t + (int64_t) (kh*FI_G + c % FI_G) * FI_D + 64*dg + 2*t;
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const float2 f = __half22float2(*(half2 *) &o[mt][nt][hh]);
                    float2 r = make_float2(f.x * b, f.y * b);
                    if (a > 0.0f) {
                        const float2 p = *(const float2 *) (arow + 8*nt);
                        r.x += p.x * a;
                        r.y += p.y * a;
                    }
                    *(half2 *) (orow + 8*nt) = __floats2half2_rn(r.x * il, r.y * il);
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < 8; ++nt) {
                    const float2 f = __half22float2(*(half2 *) &o[mt][nt][hh]);
                    float2 r = make_float2(f.x * b, f.y * b);
                    if (a > 0.0f) {
                        const float2 p = *(const float2 *) (arow + 8*nt);
                        r.x += p.x * a;
                        r.y += p.y * a;
                    }
                    *(float2 *) (arow + 8*nt) = r;
                }
            }
        }
    }
}

// One prefill request: q [ntok, NH, D] f16 (strides qs_t, qs_h), query i at kv position q_pos0 + i, kv length
// q_pos0 + ntok; pages[] its page ids; K / V planes of the FP8 paged cache at k_base / v_base with byte strides s_page,
// s_tok, s_head; out [ntok, NH, D] f16 (row stride os_t). Workspace: K8 (NKV*C*D int8), KS (NKV*C f32), V16
// (NKV*C*D f16), A (ntok*NH*D f32) and MS (ntok*NH float2), C = chunk (multiple of 32).
extern "C" int fa8_prefill(const void * q, long long qs_t, long long qs_h, const void * k_base, const void * v_base,
                           long long s_page, long long s_tok, long long s_head, int page_size, const int * pages,
                           long long q_pos0, int ntok, int NH, int NKV, float scale, float k_scale, float v_scale,
                           void * out, long long os_t, void * K8, void * KS, void * V16, void * A, void * MS, int C,
                           void * stream_) {
    if (ntok <= 0) {
        return 0;
    }
    if (NH != FI_G * NKV || C % FI_BKV != 0 || C <= 0) {
        return 1;
    }
    cudaStream_t stream = (cudaStream_t) stream_;
    static bool attr_set[64] = {false};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!attr_set[dev]) {
        if (cudaFuncSetAttribute(fa8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sizeof(fi_smem)) != cudaSuccess) {
            return 2;
        }
        attr_set[dev] = true;
    }
    const int64_t n_kv = q_pos0 + ntok;
    const int nblk = (ntok + FI_TPB - 1) / FI_TPB;
    for (int64_t c0 = 0; c0 < n_kv; c0 += C) {
        // the converted length is padded to a multiple of 32 with positions the mask hides (valid pages: the last
        // page is fully allocated)
        const int64_t rem = n_kv - c0;
        const int len = (int) (rem < C ? (rem + FI_BKV - 1) / FI_BKV * FI_BKV : C);
        const int64_t nrow_kv = (int64_t) NKV * len;
        fa8_convert_k<<<(nrow_kv + 7) / 8, 256, 0, stream>>>((const uint8_t *) k_base, s_page, s_tok, s_head, page_size,
            pages, c0, len, n_kv - 1, NKV, k_scale, (int8_t *) K8, (float *) KS);
        fa8_convert_v<<<(nrow_kv + 7) / 8, 256, 0, stream>>>((const uint8_t *) v_base, s_page, s_tok, s_head, page_size,
            pages, c0, len, n_kv - 1, NKV, v_scale, (half *) V16);
        fa8_kernel<<<dim3(nblk, NKV), FI_NT, sizeof(fi_smem), stream>>>((const half *) q, qs_t, qs_h, (const int8_t *) K8,
            (const float *) KS, (const half *) V16, c0, q_pos0, (float *) A, (float2 *) MS, (half *) out, os_t, ntok, NH,
            len, scale);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 3;
}
