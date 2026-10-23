// Gated DeltaNet chunked prefill on Turing tensor cores: replacements for the two FLA Triton kernels that dominate the
// prefill (vllm.third_party.flash_linear_attention: chunk_gated_delta_rule_fwd_kernel_h_blockdim64 and
// chunk_fwd_kernel_o). Triton 3.x does not use the sm75 tensor cores (tl.dot runs on FMA, ~2 TFLOPS); these use
// mma.sync m16n8k8 (fp16 operands, fp32 accumulation) with ldmatrix from shared memory.
// Same math and the same fp16 rounding points as FLA: fp32 state, converted to the input dtype for the w.S product,
// v_new stored before gating, the gated v_new converted before the k^T.v_new update, chunk states h stored in the input
// dtype, o = (q.h^T * e^g + causal(q.k^T * e^(g_t - g_s)) . v_new) * scale. Chunk size 64, K = 128, natural exp.
// Layouts (token-major, B = 1 with cu_seqlens or B sequences of length T):
//   q, k [tokens, Hg, K]; w [tokens, H, K]; u, v_new, o [tokens, H, V]; g [tokens, H] fp32 (chunk-local cumsum);
//   h [chunks, H, V, K]; initial / final state [N, H, V, K] fp32.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define BT 64
#define KD 128
#define PADH 8   // shared rows padded by 8 halves (16 B): conflict-free ldmatrix

template <typename T> struct vec2;
template <> struct vec2<half>          { using t = half2; };
template <> struct vec2<__nv_bfloat16> { using t = __nv_bfloat162; };
template <typename T> __device__ __forceinline__ typename vec2<T>::t f2v(float a, float b);
template <> __device__ __forceinline__ half2 f2v<half>(float a, float b) { return __floats2half2_rn(a, b); }
template <> __device__ __forceinline__ __nv_bfloat162 f2v<__nv_bfloat16>(float a, float b) { return __floats2bfloat162_rn(a, b); }
template <typename T> __device__ __forceinline__ float2 v2f(typename vec2<T>::t x);
template <> __device__ __forceinline__ float2 v2f<half>(half2 x) { return __half22float2(x); }
template <> __device__ __forceinline__ float2 v2f<__nv_bfloat16>(__nv_bfloat162 x) { return __bfloat1622float2(x); }

static __device__ __forceinline__ uint32_t smem_u32(const void * p) { return (uint32_t) __cvta_generic_to_shared(p); }
static __device__ __forceinline__ void ldsm_x4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
static __device__ __forceinline__ void ldsm_x4_t(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const void * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_u32(p)));
}
// D += A.B, m16n8k8, A row (2 regs), B col (1 reg), fp32 accumulators; f16 or bf16 operands (bf16: sm80+ only)
template <typename T> __device__ __forceinline__ void mma1688(float * d, uint32_t a0, uint32_t a1, uint32_t b0);
template <> __device__ __forceinline__ void mma1688<half>(float * d, uint32_t a0, uint32_t a1, uint32_t b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}

// copy a [rows][KD] tile (row stride ld elements in global) into shared [rows][KD + PADH], zero rows >= nvalid
template <typename T, int ROWS, int NT>
static __device__ __forceinline__ void load_tile(T * s, const T * g, const int64_t ld, const int nvalid, const int tid) {
    constexpr int CH = KD / 8;   // 16-byte chunks per row
#pragma unroll
    for (int i = tid; i < ROWS * CH; i += NT) {
        const int r = i / CH, c = (i % CH) * 8;
        int4 v = make_int4(0, 0, 0, 0);
        if (r < nvalid) {
            v = *reinterpret_cast<const int4 *>(g + (int64_t) r * ld + c);
        }
        *reinterpret_cast<int4 *>(s + r * (KD + PADH) + c) = v;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// chunk state recurrence: grid (N * H, V / BV), 4 warps. State S[v][k] (BV x 128) in mma accumulators, warp w owns
// k in [32w, 32w + 32). Per chunk: store S -> h; b_v = w.S^T (64 x BV); v_new = u - b_v; S = S e^gl + k^T (v_new e^(gl-g))
// BV = 64 (dynamic shared memory, 61.7 KB): 48 blocks for 24 heads, all resident on the 68 SMs. With BV = 32 the 96
// blocks did not fit (one 47.8 KB block per SM): 28 ran in a second round, and every block reloads the same w / k tiles.
template <typename T, int BV>
__global__ void __launch_bounds__(128) gdn_h_kernel(const T * __restrict__ k, const T * __restrict__ w,
                                                    const T * __restrict__ u, const float * __restrict__ g,
                                                    const float * __restrict__ h0, T * __restrict__ h,
                                                    T * __restrict__ v_new, float * __restrict__ ht,
                                                    const int * __restrict__ cu_seqlens, const int * __restrict__ chunk_offsets,
                                                    const int T_fixed, const int H, const int Hg, const int V) {
    extern __shared__ __align__(16) unsigned char smem[];
    T (*sW)[KD + PADH] = reinterpret_cast<T (*)[KD + PADH]>(smem);
    T (*sK)[KD + PADH] = sW + BT;
    T (*sS)[KD + PADH] = sK + BT;                                    // [BV][KD]
    T (*sV)[BV + PADH] = reinterpret_cast<T (*)[BV + PADH]>(sS + BV);
    float * sG = reinterpret_cast<float *>(sV + BT);

    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int gq = lane >> 2, tq = lane & 3;          // mma group / thread-in-group
    const int i_nh = blockIdx.x, n = i_nh / H, hh = i_nh % H, v0 = blockIdx.y * BV;
    const int hk = hh / (H / Hg);
    int bos, Tn, boh;
    if (cu_seqlens) {
        bos = cu_seqlens[n];
        Tn  = cu_seqlens[n + 1] - bos;
        boh = chunk_offsets[n];
    } else {
        bos = n * T_fixed;
        Tn  = T_fixed;
        boh = n * ((T_fixed + BT - 1) / BT);
    }
    const int NTn = (Tn + BT - 1) / BT;

    // state accumulators: [m-tile (v 16-block)][n-tile (k 8-block within the warp's 32)][4]
    float S[BV / 16][4][4];
#pragma unroll
    for (int mt = 0; mt < BV / 16; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float x = 0.0f;
                if (h0) {
                    const int vv = v0 + mt * 16 + gq + (e >> 1) * 8;
                    const int kk = wp * 32 + nt * 8 + tq * 2 + (e & 1);
                    x = h0[((int64_t) i_nh * V + vv) * KD + kk];
                }
                S[mt][nt][e] = x;
            }
        }
    }

    for (int it = 0; it < NTn; ++it) {
        const int t0 = it * BT, nval = min(BT, Tn - t0);
        const int64_t tok0 = (int64_t) bos + t0;
        load_tile<T, BT, 128>(&sW[0][0], w + (tok0 * H + hh) * KD, (int64_t) H * KD, nval, tid);
        load_tile<T, BT, 128>(&sK[0][0], k + (tok0 * Hg + hk) * KD, (int64_t) Hg * KD, nval, tid);
        if (tid < BT) {
            sG[tid] = tid < nval ? g[(tok0 + tid) * H + hh] : 0.0f;
        }
        // S -> shared (input dtype)
#pragma unroll
        for (int mt = 0; mt < BV / 16; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int vv = mt * 16 + gq, kk = wp * 32 + nt * 8 + tq * 2;
                *reinterpret_cast<typename vec2<T>::t *>(&sS[vv][kk])     = f2v<T>(S[mt][nt][0], S[mt][nt][1]);
                *reinterpret_cast<typename vec2<T>::t *>(&sS[vv + 8][kk]) = f2v<T>(S[mt][nt][2], S[mt][nt][3]);
            }
        }
        __syncthreads();
        // S -> h[boh + it] (coalesced)
        {
            T * hp = h + (((int64_t) (boh + it) * H + hh) * V + v0) * KD;
#pragma unroll
            for (int i = tid; i < BV * KD / 8; i += 128) {
                const int r = i / (KD / 8), c = (i % (KD / 8)) * 8;
                *reinterpret_cast<int4 *>(hp + (int64_t) r * KD + c) = *reinterpret_cast<const int4 *>(&sS[r][c]);
            }
        }
        // b_v[t][v] = sum_k w[t][k] S[v][k]: warp w takes t rows [16w, 16w + 16), 4 n-tiles of v, 16 k-steps
        float bv[BV / 8][4];
#pragma unroll
        for (int nt = 0; nt < BV / 8; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                bv[nt][e] = 0.0f;
            }
        }
        const int mrow = wp * 16;
#pragma unroll
        for (int ks = 0; ks < KD; ks += 16) {
            uint32_t a0, a1, a2, a3;
            const int j = lane >> 3, i = lane & 7;
            ldsm_x4(a0, a1, a2, a3, &sW[mrow + i + (j & 1) * 8][ks + (j >> 1) * 8]);
            // all B fragments first, then the k 0-7 halves of every n-tile before the k 8-15 halves: no back-to-back
            // MMAs into one accumulator (the per-accumulator order, and so the result, is unchanged)
            uint32_t b[BV / 16][4];
#pragma unroll
            for (int np = 0; np < BV / 16; ++np) {   // pairs of v n-tiles
                ldsm_x4(b[np][0], b[np][1], b[np][2], b[np][3], &sS[np * 16 + (j >> 1) * 8 + i][ks + (j & 1) * 8]);
            }
#pragma unroll
            for (int np = 0; np < BV / 16; ++np) {
                mma1688<T>(bv[np * 2], a0, a1, b[np][0]);
                mma1688<T>(bv[np * 2 + 1], a0, a1, b[np][2]);
            }
#pragma unroll
            for (int np = 0; np < BV / 16; ++np) {
                mma1688<T>(bv[np * 2], a2, a3, b[np][1]);
                mma1688<T>(bv[np * 2 + 1], a2, a3, b[np][3]);
            }
        }
        // v_new = u - b_v (stored), then gated -> shared (input dtype)
        const float gl = sG[nval - 1];
#pragma unroll
        for (int nt = 0; nt < BV / 8; ++nt) {
#pragma unroll
            for (int half_ = 0; half_ < 2; ++half_) {
                const int t  = mrow + gq + half_ * 8;
                const int vv = nt * 8 + tq * 2;
                float x0 = 0.0f, x1 = 0.0f;
                if (t < nval) {
                    const int64_t off = ((tok0 + t) * H + hh) * (int64_t) V + v0 + vv;
                    const float2  uu  = v2f<T>(*reinterpret_cast<const typename vec2<T>::t *>(u + off));
                    x0 = uu.x - bv[nt][half_ * 2];
                    x1 = uu.y - bv[nt][half_ * 2 + 1];
                    *reinterpret_cast<typename vec2<T>::t *>(v_new + off) = f2v<T>(x0, x1);
                    const float sc = __expf(gl - sG[t]);
                    x0 *= sc;
                    x1 *= sc;
                }
                *reinterpret_cast<typename vec2<T>::t *>(&sV[t][vv]) = f2v<T>(x0, x1);
            }
        }
        __syncthreads();
        // S = S e^gl + sum_t vg[t][v] k[t][k]: A[v][t] = sV^T (ldmatrix.trans), B[t][k] = sK (ldmatrix.trans)
        const float egl = __expf(gl);
#pragma unroll
        for (int mt = 0; mt < BV / 16; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    S[mt][nt][e] *= egl;
                }
            }
        }
#pragma unroll
        for (int ts = 0; ts < BT; ts += 16) {
            const int j = lane >> 3, i = lane & 7;
            uint32_t a[BV / 16][4];
#pragma unroll
            for (int mt = 0; mt < BV / 16; ++mt) {   // A m16k16 from sV[t][v]: matrix j -> (v-block j & 1, t-block j >> 1)
                ldsm_x4_t(a[mt][0], a[mt][1], a[mt][2], a[mt][3], &sV[ts + (j >> 1) * 8 + i][mt * 16 + (j & 1) * 8]);
            }
            uint32_t b[2][4];
#pragma unroll
            for (int np = 0; np < 2; ++np) {   // B pairs of k n-tiles from sK[t][k]: matrix j -> (t-block j & 1, k-block j >> 1)
                ldsm_x4_t(b[np][0], b[np][1], b[np][2], b[np][3], &sK[ts + (j & 1) * 8 + i][wp * 32 + np * 16 + (j >> 1) * 8]);
            }
            // t 0-7 halves of all accumulators, then the t 8-15 halves (same per-accumulator order as before)
#pragma unroll
            for (int mt = 0; mt < BV / 16; ++mt) {
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    mma1688<T>(S[mt][np * 2], a[mt][0], a[mt][1], b[np][0]);
                    mma1688<T>(S[mt][np * 2 + 1], a[mt][0], a[mt][1], b[np][2]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < BV / 16; ++mt) {
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    mma1688<T>(S[mt][np * 2], a[mt][2], a[mt][3], b[np][1]);
                    mma1688<T>(S[mt][np * 2 + 1], a[mt][2], a[mt][3], b[np][3]);
                }
            }
        }
        __syncthreads();
    }
    if (ht) {
#pragma unroll
        for (int mt = 0; mt < BV / 16; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int vv = v0 + mt * 16 + gq + (e >> 1) * 8;
                    const int kk = wp * 32 + nt * 8 + tq * 2 + (e & 1);
                    ht[((int64_t) i_nh * V + vv) * KD + kk] = S[mt][nt][e];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// chunk output: grid (V / 64, chunks, H), 4 warps, warp w takes t rows [16w, 16w + 16).
// o = (q.h^T e^g + causal(q.k^T e^(g_t - g_s)) . v_new) * scale
template <typename T>
__global__ void __launch_bounds__(128) gdn_o_kernel(const T * __restrict__ q, const T * __restrict__ k,
                                                    const T * __restrict__ v, const T * __restrict__ h,
                                                    const float * __restrict__ g, T * __restrict__ o,
                                                    const int * __restrict__ cu_seqlens, const int * __restrict__ chunk_indices,
                                                    const float scale, const int T_fixed, const int H, const int Hg, const int V) {
    constexpr int BV = 64;
    extern __shared__ __align__(16) unsigned char smem[];
    T (*sQ)[KD + PADH] = reinterpret_cast<T (*)[KD + PADH]>(smem);
    T (*sK)[KD + PADH] = sQ + BT;
    T (*sH)[KD + PADH] = sK + BT;                                   // [BV][KD]
    T (*sVn)[BV + PADH] = reinterpret_cast<T (*)[BV + PADH]>(sH + BV);
    float * sG = reinterpret_cast<float *>(sVn + BT);

    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int gq = lane >> 2, tq = lane & 3;
    const int v0 = blockIdx.x * BV, i_tg = blockIdx.y, hh = blockIdx.z, hk = hh / (H / Hg);
    int bos, Tn, it;
    if (cu_seqlens) {
        const int n = chunk_indices[2 * i_tg];
        it  = chunk_indices[2 * i_tg + 1];
        bos = cu_seqlens[n];
        Tn  = cu_seqlens[n + 1] - bos;
    } else {
        const int NT = (T_fixed + BT - 1) / BT;
        const int b  = i_tg / NT;
        it  = i_tg % NT;
        bos = b * T_fixed;
        Tn  = T_fixed;
    }
    const int t0 = it * BT, nval = min(BT, Tn - t0);
    const int64_t tok0 = (int64_t) bos + t0;
    load_tile<T, BT, 128>(&sQ[0][0], q + (tok0 * Hg + hk) * KD, (int64_t) Hg * KD, nval, tid);
    load_tile<T, BT, 128>(&sK[0][0], k + (tok0 * Hg + hk) * KD, (int64_t) Hg * KD, nval, tid);
    load_tile<T, BV, 128>(&sH[0][0], h + (((int64_t) i_tg * H + hh) * V + v0) * KD, KD, BV, tid);
#pragma unroll
    for (int i = tid; i < BT * BV / 8; i += 128) {
        const int r = i / (BV / 8), c = (i % (BV / 8)) * 8;
        int4 x = make_int4(0, 0, 0, 0);
        if (r < nval) {
            x = *reinterpret_cast<const int4 *>(v + ((tok0 + r) * H + hh) * (int64_t) V + v0 + c);
        }
        *reinterpret_cast<int4 *>(&sVn[r][c]) = x;
    }
    if (tid < BT) {
        sG[tid] = tid < nval ? g[(tok0 + tid) * H + hh] : 0.0f;
    }
    __syncthreads();

    const int mrow = wp * 16, j = lane >> 3, i = lane & 7;
    float ao[8][4], aa[8][4];   // q.h^T [t][v] (8 v n-tiles), q.k^T [t][s] (8 s n-tiles)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            ao[nt][e] = 0.0f;
            aa[nt][e] = 0.0f;
        }
    }
#pragma unroll
    for (int ks = 0; ks < KD; ks += 16) {
        uint32_t a0, a1, a2, a3;
        ldsm_x4(a0, a1, a2, a3, &sQ[mrow + i + (j & 1) * 8][ks + (j >> 1) * 8]);
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4(b0, b1, b2, b3, &sH[np * 16 + (j >> 1) * 8 + i][ks + (j & 1) * 8]);
            mma1688<T>(ao[np * 2], a0, a1, b0);
            mma1688<T>(ao[np * 2], a2, a3, b1);
            mma1688<T>(ao[np * 2 + 1], a0, a1, b2);
            mma1688<T>(ao[np * 2 + 1], a2, a3, b3);
            ldsm_x4(b0, b1, b2, b3, &sK[np * 16 + (j >> 1) * 8 + i][ks + (j & 1) * 8]);
            mma1688<T>(aa[np * 2], a0, a1, b0);
            mma1688<T>(aa[np * 2], a2, a3, b1);
            mma1688<T>(aa[np * 2 + 1], a0, a1, b2);
            mma1688<T>(aa[np * 2 + 1], a2, a3, b3);
        }
    }
    // gating + causal mask; the q.k^T accumulators become the A operand of the second product (fp16 / bf16)
    uint32_t pa[8][2];
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int half_ = 0; half_ < 2; ++half_) {
            const int t = mrow + gq + half_ * 8;
            const float gt = sG[t];
            ao[nt][half_ * 2] *= __expf(gt);
            ao[nt][half_ * 2 + 1] *= __expf(gt);
            float x[2];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int s = nt * 8 + tq * 2 + c;
                x[c] = (s <= t && t < nval && s < nval) ? aa[nt][half_ * 2 + c] * __expf(gt - sG[s]) : 0.0f;
            }
            const typename vec2<T>::t p = f2v<T>(x[0], x[1]);
            pa[nt][half_] = *reinterpret_cast<const uint32_t *>(&p);
        }
    }
    // o += P[t][s] . v_new[s][v]: B[s][v] from sVn (ldmatrix.trans)
#pragma unroll
    for (int ss = 0; ss < BT; ss += 16) {
#pragma unroll
        for (int np = 0; np < 4; ++np) {
            uint32_t b0, b1, b2, b3;
            ldsm_x4_t(b0, b1, b2, b3, &sVn[ss + (j & 1) * 8 + i][np * 16 + (j >> 1) * 8]);
            const int st = ss / 8;   // s n-tile = k-step of 8
            mma1688<T>(ao[np * 2], pa[st][0], pa[st][1], b0);
            mma1688<T>(ao[np * 2], pa[st + 1][0], pa[st + 1][1], b1);
            mma1688<T>(ao[np * 2 + 1], pa[st][0], pa[st][1], b2);
            mma1688<T>(ao[np * 2 + 1], pa[st + 1][0], pa[st + 1][1], b3);
        }
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
#pragma unroll
        for (int half_ = 0; half_ < 2; ++half_) {
            const int t = mrow + gq + half_ * 8;
            if (t < nval) {
                const int64_t off = ((tok0 + t) * H + hh) * (int64_t) V + v0 + nt * 8 + tq * 2;
                *reinterpret_cast<typename vec2<T>::t *>(o + off) =
                    f2v<T>(ao[nt][half_ * 2] * scale, ao[nt][half_ * 2 + 1] * scale);
            }
        }
    }
}

static size_t o_smem() {
    return (size_t) (2 * BT + 64) * (KD + PADH) * 2 + (size_t) BT * (64 + PADH) * 2 + BT * 4;
}

static size_t h_smem(int bv) {
    return (size_t) (2 * BT + bv) * (KD + PADH) * 2 + (size_t) BT * (bv + PADH) * 2 + BT * 4;
}

extern "C" int gdn_h_fp16(const void * k, const void * w, const void * u, const void * g, const void * h0, void * h,
                          void * v_new, void * ht, const void * cu_seqlens, const void * chunk_offsets,
                          int N, int T, int H, int Hg, int V, void * stream) {
    if (V % 64 != 0 || H % Hg != 0) {
        return -1;
    }
    constexpr int HBV = 64;
    static bool attr[64] = {};   // the shared memory limit is a per-device attribute
    int dev = 0;
    cudaGetDevice(&dev);
    if (!attr[dev & 63]) {
        cudaFuncSetAttribute(gdn_h_kernel<half, HBV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) h_smem(HBV));
        attr[dev & 63] = true;
    }
    dim3 grid(N * H, V / HBV);
    gdn_h_kernel<half, HBV><<<grid, 128, h_smem(HBV), (cudaStream_t) stream>>>(
        (const half *) k, (const half *) w, (const half *) u, (const float *) g, (const float *) h0, (half *) h,
        (half *) v_new, (float *) ht, (const int *) cu_seqlens, (const int *) chunk_offsets, T, H, Hg, V);
    return (int) cudaGetLastError();
}

extern "C" int gdn_o_fp16(const void * q, const void * k, const void * v, const void * h, const void * g, void * o,
                          const void * cu_seqlens, const void * chunk_indices, float scale, int NT_total, int T, int H,
                          int Hg, int V, void * stream) {
    if (V % 64 != 0 || H % Hg != 0) {
        return -1;
    }
    static bool attr[64] = {};   // the shared memory limit is a per-device attribute
    int dev = 0;
    cudaGetDevice(&dev);
    if (!attr[dev & 63]) {
        cudaFuncSetAttribute(gdn_o_kernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) o_smem());
        attr[dev & 63] = true;
    }
    dim3 grid(V / 64, NT_total, H);
    gdn_o_kernel<half><<<grid, 128, o_smem(), (cudaStream_t) stream>>>(
        (const half *) q, (const half *) k, (const half *) v, (const half *) h, (const float *) g, (half *) o,
        (const int *) cu_seqlens, (const int *) chunk_indices, scale, T, H, Hg, V);
    return (int) cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------------------------
// WY preparation, in place of FLA's chunk_scaled_dot_kkt_fwd + solve_tril + recompute_w_u_fwd (three Triton kernels,
// without tensor cores on sm75: 1.9 ms per 6720-token layer step). grid (H, chunks): the value heads sharing a key head
// are adjacent blocks (their K tile stays in L2); 8 warps, one chunk x value head per block.
//   A = strict_lower(beta_i k_i.k_j e^(g_i - g_j))   fp32, k.k^T on the tensor cores
//   T = (I + A)^-1                                    fp32 in shared memory: the 16 x 16 diagonal blocks by forward
//                                                     substitution (a lane per column, in registers), then the
//                                                     off-diagonal blocks by distance, T_RC = -T_RR sum_M A_RM T_MC
//   u = T16 . (v beta)16,  w = T16 . (k beta e^g)16   T and the scaled rows rounded to the input dtype as FLA does
// A is not written out (the FLA caller does not use it).
constexpr int WU_LDA = BT + 4;   // fp32 row stride (floats; rows stay 16-byte aligned for float4)

template <typename T>
__global__ void __launch_bounds__(256) gdn_wu_kernel(const T * __restrict__ k, const T * __restrict__ v,
                                                     const float * __restrict__ beta, const float * __restrict__ g,
                                                     T * __restrict__ w, T * __restrict__ u,
                                                     const int * __restrict__ cu_seqlens, const int * __restrict__ chunk_indices,
                                                     const int T_fixed, const int H, const int Hg) {
    using V2 = typename vec2<T>::t;
    extern __shared__ __align__(16) unsigned char smem[];
    T (*sK)[KD + PADH] = reinterpret_cast<T (*)[KD + PADH]>(smem);                          // 17408 B
    float (*sA)[WU_LDA] = reinterpret_cast<float (*)[WU_LDA]>(smem + BT * (KD + PADH) * sizeof(T));
    float (*sT)[WU_LDA] = sA + BT;
    T (*sTh)[BT + PADH] = reinterpret_cast<T (*)[BT + PADH]>(sA);    // after the inverse: T in the input dtype
    T (*sV)[KD + PADH] = reinterpret_cast<T (*)[KD + PADH]>(sT);     // after the conversion: v beta
    float * sG = reinterpret_cast<float *>(sT + BT);
    float * sB = sG + BT;

    const int tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
    const int gq = lane >> 2, tq = lane & 3, j = lane >> 3, i = lane & 7;
    const int hh = blockIdx.x, i_tg = blockIdx.y, hk = hh / (H / Hg);
    int bos, Tn, it;
    if (cu_seqlens) {
        const int n = chunk_indices[2 * i_tg];
        it  = chunk_indices[2 * i_tg + 1];
        bos = cu_seqlens[n];
        Tn  = cu_seqlens[n + 1] - bos;
    } else {
        const int NT = (T_fixed + BT - 1) / BT;
        it  = i_tg % NT;
        bos = (i_tg / NT) * T_fixed;
        Tn  = T_fixed;
    }
    const int t0 = it * BT, nval = min(BT, Tn - t0);
    const int64_t tok0 = (int64_t) bos + t0;
    // v is only needed at the end: its loads are issued first and land in registers meanwhile
    int4 vr[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int x = tid + e * 256, r = x / (KD / 8), c = (x % (KD / 8)) * 8;
        vr[e] = r < nval ? *reinterpret_cast<const int4 *>(v + ((tok0 + r) * H + hh) * (int64_t) KD + c)
                         : make_int4(0, 0, 0, 0);
    }
    load_tile<T, BT, 256>(&sK[0][0], k + (tok0 * Hg + hk) * KD, (int64_t) Hg * KD, nval, tid);
    if (tid < BT) {
        sG[tid] = tid < nval ? g[(tok0 + tid) * H + hh] : 0.0f;
        sB[tid] = tid < nval ? beta[(tok0 + tid) * H + hh] : 0.0f;
    }
    __syncthreads();

    // (1) A: warp (mi, nh) = rows [16 mi, +16) x cols [32 nh, +32); the all-upper warp (0, 1) only writes zeros.
    // The MMAs alternate between accumulators (no back-to-back dependent MMAs).
    {
        const int mi = wp >> 1, nh = wp & 1, mrow = mi * 16;
        float acc[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[nt][e] = 0.0f;
            }
        }
        if (nh * 32 <= mrow + 15) {
#pragma unroll
            for (int ks = 0; ks < KD; ks += 16) {
                uint32_t a0, a1, a2, a3, b[2][4];
                ldsm_x4(a0, a1, a2, a3, &sK[mrow + i + (j & 1) * 8][ks + (j >> 1) * 8]);
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    ldsm_x4(b[np][0], b[np][1], b[np][2], b[np][3],
                            &sK[nh * 32 + np * 16 + (j >> 1) * 8 + i][ks + (j & 1) * 8]);
                }
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    mma1688<T>(acc[np * 2], a0, a1, b[np][0]);
                    mma1688<T>(acc[np * 2 + 1], a0, a1, b[np][2]);
                }
#pragma unroll
                for (int np = 0; np < 2; ++np) {
                    mma1688<T>(acc[np * 2], a2, a3, b[np][1]);
                    mma1688<T>(acc[np * 2 + 1], a2, a3, b[np][3]);
                }
            }
        }
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = mrow + gq + (e >> 1) * 8, c = nh * 32 + nt * 8 + tq * 2 + (e & 1);
                sA[r][c] = (r > c && r < nval) ? acc[nt][e] * sB[r] * __expf(sG[r] - sG[c]) : 0.0f;
            }
        }
    }
    __syncthreads();

    // (2a) diagonal blocks: warps 0-1, a half warp per block, lane = column c:
    // T[r][c] = -A[r][c] - sum_{c<m<r} A[r][m] T[m][c]. Meanwhile warps 2-7 scale the k rows by beta e^g in place
    // ((k beta) e^g in fp32, rounded to the input dtype, the order of the FLA recompute_w_u kernel).
    if (wp < 2) {
        const int base = (wp * 2 + (lane >> 4)) * 16, c = lane & 15;
        float col[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            float x = r == c ? 1.0f : 0.0f;
            if (r > c) {
                x = -sA[base + r][base + c];
#pragma unroll
                for (int m = 1; m < r; ++m) {
                    if (m > c) {
                        x -= sA[base + r][base + m] * col[m];
                    }
                }
            }
            col[r] = x;
        }
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            sT[base + r][base + c] = col[r];
        }
    } else {
        for (int x = tid - 64; x < BT * KD / 8; x += 192) {
            const int r = x / (KD / 8), c = (x % (KD / 8)) * 8;
            const float b = sB[r], eg = expf(sG[r]);
            int4 raw = *reinterpret_cast<const int4 *>(&sK[r][c]);
            V2 * p = reinterpret_cast<V2 *>(&raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = v2f<T>(p[e]);
                p[e] = f2v<T>(f.x * b * eg, f.y * b * eg);
            }
            *reinterpret_cast<int4 *>(&sK[r][c]) = raw;
        }
    }
    __syncthreads();
    // (2b) off-diagonal blocks by distance d: X = sum_{M=C}^{R-1} A_RM T_MC kept in the (unused) upper block (C, R),
    // then T_RC = -T_RR X. Thread = (block, row r, 4 columns): one float4 + one broadcast load per 4 FMAs.
    {
        const int c0 = (tid & 3) * 4, r = (tid >> 2) & 15, blk = tid >> 6;
#pragma unroll
        for (int d = 1; d < 4; ++d) {
            if (blk < 4 - d) {
                const int C = blk, R = C + d;
                float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int M = C; M < R; ++M) {
#pragma unroll
                    for (int m = 0; m < 16; ++m) {
                        const float a = sA[16 * R + r][16 * M + m];
                        const float4 t = *reinterpret_cast<const float4 *>(&sT[16 * M + m][16 * C + c0]);
                        x.x += a * t.x;
                        x.y += a * t.y;
                        x.z += a * t.z;
                        x.w += a * t.w;
                    }
                }
                *reinterpret_cast<float4 *>(&sT[16 * C + r][16 * R + c0]) = x;
            }
            __syncthreads();
            if (blk < 4 - d) {
                const int C = blk, R = C + d;
                float4 y = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
                for (int m = 0; m < 16; ++m) {
                    const float a = sT[16 * R + r][16 * R + m];
                    const float4 t = *reinterpret_cast<const float4 *>(&sT[16 * C + m][16 * R + c0]);
                    y.x += a * t.x;
                    y.y += a * t.y;
                    y.z += a * t.z;
                    y.w += a * t.w;
                }
                *reinterpret_cast<float4 *>(&sT[16 * R + r][16 * C + c0]) = make_float4(-y.x, -y.y, -y.z, -y.w);
            }
            __syncthreads();
        }
    }

    // (3) T -> input dtype (upper triangle zero) into the sA space; then the prefetched v rows, scaled by beta, into
    // the sT space
    for (int x = tid; x < BT * BT / 2; x += 256) {
        const int r = x / (BT / 2), c = (x % (BT / 2)) * 2;
        *reinterpret_cast<V2 *>(&sTh[r][c]) = f2v<T>(c <= r ? sT[r][c] : 0.0f, c + 1 <= r ? sT[r][c + 1] : 0.0f);
    }
    __syncthreads();
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int x = tid + e * 256, r = x / (KD / 8), c = (x % (KD / 8)) * 8;
        const float b = sB[r];
        V2 * p = reinterpret_cast<V2 *>(&vr[e]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 f = v2f<T>(p[q]);
            p[q] = f2v<T>(f.x * b, f.y * b);
        }
        *reinterpret_cast<int4 *>(&sV[r][c]) = vr[e];
    }
    __syncthreads();

    // (4) u = T16 . vb, w = T16 . kb. T is lower triangular (row tile m needs m + 1 k-steps): warps 0-3 take row tiles
    // 0 and 3, warps 4-7 tiles 1 and 2 (equal MMA counts), each warp a 32-column quarter; the B fragments of a k-step
    // serve both tiles. The results go through shared memory (v / k are consumed) for 16-byte coalesced stores.
    {
        const int lo = wp < 4 ? 0 : 1, hi = 3 - lo, q = wp & 3;
        float au[2][4][4], aw[2][4][4];
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    au[t][nt][e] = 0.0f;
                    aw[t][nt][e] = 0.0f;
                }
            }
        }
        for (int ks = 0; ks <= hi * 16; ks += 16) {
            const bool dolo = ks <= lo * 16;
            uint32_t ah[4], al[4] = {0, 0, 0, 0};
            ldsm_x4(ah[0], ah[1], ah[2], ah[3], &sTh[hi * 16 + i + (j & 1) * 8][ks + (j >> 1) * 8]);
            if (dolo) {
                ldsm_x4(al[0], al[1], al[2], al[3], &sTh[lo * 16 + i + (j & 1) * 8][ks + (j >> 1) * 8]);
            }
#pragma unroll
            for (int np = 0; np < 2; ++np) {
                uint32_t bv[4], bk[4];
                ldsm_x4_t(bv[0], bv[1], bv[2], bv[3], &sV[ks + (j & 1) * 8 + i][q * 32 + np * 16 + (j >> 1) * 8]);
                ldsm_x4_t(bk[0], bk[1], bk[2], bk[3], &sK[ks + (j & 1) * 8 + i][q * 32 + np * 16 + (j >> 1) * 8]);
                mma1688<T>(au[1][np * 2], ah[0], ah[1], bv[0]);
                mma1688<T>(au[1][np * 2 + 1], ah[0], ah[1], bv[2]);
                mma1688<T>(aw[1][np * 2], ah[0], ah[1], bk[0]);
                mma1688<T>(aw[1][np * 2 + 1], ah[0], ah[1], bk[2]);
                mma1688<T>(au[1][np * 2], ah[2], ah[3], bv[1]);
                mma1688<T>(au[1][np * 2 + 1], ah[2], ah[3], bv[3]);
                mma1688<T>(aw[1][np * 2], ah[2], ah[3], bk[1]);
                mma1688<T>(aw[1][np * 2 + 1], ah[2], ah[3], bk[3]);
                if (dolo) {
                    mma1688<T>(au[0][np * 2], al[0], al[1], bv[0]);
                    mma1688<T>(au[0][np * 2 + 1], al[0], al[1], bv[2]);
                    mma1688<T>(aw[0][np * 2], al[0], al[1], bk[0]);
                    mma1688<T>(aw[0][np * 2 + 1], al[0], al[1], bk[2]);
                    mma1688<T>(au[0][np * 2], al[2], al[3], bv[1]);
                    mma1688<T>(au[0][np * 2 + 1], al[2], al[3], bv[3]);
                    mma1688<T>(aw[0][np * 2], al[2], al[3], bk[1]);
                    mma1688<T>(aw[0][np * 2 + 1], al[2], al[3], bk[3]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int mrow = (t ? hi : lo) * 16;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = q * 32 + nt * 8 + tq * 2;
                *reinterpret_cast<V2 *>(&sV[mrow + gq][c])     = f2v<T>(au[t][nt][0], au[t][nt][1]);
                *reinterpret_cast<V2 *>(&sV[mrow + gq + 8][c]) = f2v<T>(au[t][nt][2], au[t][nt][3]);
                *reinterpret_cast<V2 *>(&sK[mrow + gq][c])     = f2v<T>(aw[t][nt][0], aw[t][nt][1]);
                *reinterpret_cast<V2 *>(&sK[mrow + gq + 8][c]) = f2v<T>(aw[t][nt][2], aw[t][nt][3]);
            }
        }
        __syncthreads();
        for (int x = tid; x < BT * KD / 8; x += 256) {
            const int r = x / (KD / 8), c = (x % (KD / 8)) * 8;
            if (r < nval) {
                const int64_t off = ((tok0 + r) * H + hh) * (int64_t) KD + c;
                *reinterpret_cast<int4 *>(u + off) = *reinterpret_cast<const int4 *>(&sV[r][c]);
                *reinterpret_cast<int4 *>(w + off) = *reinterpret_cast<const int4 *>(&sK[r][c]);
            }
        }
    }
}

static size_t wu_smem() {
    return (size_t) BT * (KD + PADH) * 2 + 2 * (size_t) BT * WU_LDA * 4 + 2 * BT * 4;
}

// k [tokens, Hg, 128], v [tokens, H, 128] fp16; beta, g (chunk-local cumsum) [tokens, H] fp32 -> w, u [tokens, H, 128]
extern "C" int gdn_wu_fp16(const void * k, const void * v, const void * beta, const void * g, void * w, void * u,
                           const void * cu_seqlens, const void * chunk_indices, int NT_total, int T, int H, int Hg,
                           void * stream) {
    if (H % Hg != 0) {
        return -1;
    }
    static bool attr[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (!attr[dev & 63]) {
        cudaFuncSetAttribute(gdn_wu_kernel<half>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) wu_smem());
        attr[dev & 63] = true;
    }
    dim3 grid(H, NT_total);
    gdn_wu_kernel<half><<<grid, 256, wu_smem(), (cudaStream_t) stream>>>(
        (const half *) k, (const half *) v, (const float *) beta, (const float *) g, (half *) w, (half *) u,
        (const int *) cu_seqlens, (const int *) chunk_indices, T, H, Hg);
    return (int) cudaGetLastError();
}
