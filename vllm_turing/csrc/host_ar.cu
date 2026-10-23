// Two-rank all-reduce for small fp16 / bf16 tensors through pinned host memory, for GPUs without peer access
// (2x 2080 Ti on different sockets: NCCL takes ~28 us per decode-sized all-reduce there, ~129 per decode step).
// Both worker processes map the same POSIX shared-memory segment (cudaHostRegister'ed). Each rank writes its tensor
// into its own region as 8-byte LL words {4 data bytes, 32-bit sequence flag} (one store: data and flag arrive
// together, no fences), then polls the peer's region until every word carries the current sequence and adds.
// Sequence counters live in device memory (one per block): the launch is CUDA-graph capturable. Regions are double
// buffered: a rank can be at most one call ahead (finishing call s+1 needs the peer's s+1 data, written only after the
// peer's call s kernel, which read our call-s data, completed), so call s+2 may reuse call s's slot.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define HAR_THREADS 512
#define HAR_WPT 8   // words per thread per block

template <typename T> __device__ __forceinline__ uint32_t add2(uint32_t a, uint32_t b);
template <> __device__ __forceinline__ uint32_t add2<half>(uint32_t a, uint32_t b) {
    const float2 x = __half22float2(*reinterpret_cast<const half2 *>(&a));
    const float2 y = __half22float2(*reinterpret_cast<const half2 *>(&b));
    const half2  r = __floats2half2_rn(x.x + y.x, x.y + y.y);
    return *reinterpret_cast<const uint32_t *>(&r);
}
template <> __device__ __forceinline__ uint32_t add2<__nv_bfloat16>(uint32_t a, uint32_t b) {
    const float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&a));
    const float2 y = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162 *>(&b));
    const __nv_bfloat162 r = __floats2bfloat162_rn(x.x + y.x, x.y + y.y);
    return *reinterpret_cast<const uint32_t *>(&r);
}

template <typename T>
__global__ void __launch_bounds__(HAR_THREADS) har_kernel(const uint32_t * __restrict__ in, uint32_t * __restrict__ out,
                                                         const int nwords, uint64_t * my_ll, const uint64_t * peer_ll,
                                                         int * counters, const int slot_words) {
    __shared__ uint32_t s_seq;
    if (threadIdx.x == 0) {
        s_seq = (uint32_t) counters[blockIdx.x] + 1u;
    }
    __syncthreads();
    const uint32_t seq  = s_seq;
    const int      base = blockIdx.x * HAR_THREADS * HAR_WPT;
    volatile uint64_t *       mine = my_ll + (size_t) (seq & 1u) * slot_words;
    const volatile uint64_t * peer = peer_ll + (size_t) (seq & 1u) * slot_words;

    uint32_t v[HAR_WPT];
#pragma unroll
    for (int j = 0; j < HAR_WPT; ++j) {
        const int i = base + j * HAR_THREADS + threadIdx.x;
        if (i < nwords) {
            v[j]    = in[i];
            mine[i] = ((uint64_t) seq << 32) | v[j];
        }
    }
    // poll all of this thread's words at once (their PCIe round trips overlap), re-read the late ones
    uint32_t pending = 0;
#pragma unroll
    for (int j = 0; j < HAR_WPT; ++j) {
        if (base + j * HAR_THREADS + threadIdx.x < nwords) {
            pending |= 1u << j;
        }
    }
    while (pending) {
        uint64_t p[HAR_WPT];
#pragma unroll
        for (int j = 0; j < HAR_WPT; ++j) {
            if (pending & (1u << j)) {
                p[j] = peer[base + j * HAR_THREADS + threadIdx.x];
            }
        }
#pragma unroll
        for (int j = 0; j < HAR_WPT; ++j) {
            if ((pending & (1u << j)) && (uint32_t) (p[j] >> 32) == seq) {
                out[base + j * HAR_THREADS + threadIdx.x] = add2<T>(v[j], (uint32_t) p[j]);
                pending &= ~(1u << j);
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        counters[blockIdx.x] = (int) seq;
    }
}

// load the kernels now: with lazy module loading the first launch synchronizes, which can deadlock behind a peer that
// is already spinning in this kernel (device-memory mode over P2P)
extern "C" int har_preload() {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, har_kernel<half>) != cudaSuccess) {
        return 1;
    }
    return cudaFuncGetAttributes(&fa, har_kernel<__nv_bfloat16>) == cudaSuccess ? 0 : 2;
}

extern "C" int har_blocks(int nwords) {
    return (nwords + HAR_THREADS * HAR_WPT - 1) / (HAR_THREADS * HAR_WPT);
}

// register a host range (shared-memory mapping) for device access; returns the device pointer (0 on failure)
extern "C" void * har_register(void * ptr, size_t size) {
    if (cudaHostRegister(ptr, size, cudaHostRegisterMapped | cudaHostRegisterPortable) != cudaSuccess) {
        return nullptr;
    }
    void * dptr = nullptr;
    if (cudaHostGetDevicePointer(&dptr, ptr, 0) != cudaSuccess) {
        return nullptr;
    }
    return dptr;
}

extern "C" int har_unregister(void * ptr) {
    return (int) cudaHostUnregister(ptr);
}

// in / out: nwords 32-bit words (2 elements each); dtype 0 = fp16, 1 = bf16
extern "C" int har_all_reduce(const void * in, void * out, int nwords, int dtype, void * my_ll, const void * peer_ll,
                              void * counters, int slot_words, void * stream) {
    const int nb = har_blocks(nwords);
    if (nwords > slot_words) {
        return -1;
    }
    cudaStream_t s = (cudaStream_t) stream;
    if (dtype == 0) {
        har_kernel<half><<<nb, HAR_THREADS, 0, s>>>((const uint32_t *) in, (uint32_t *) out, nwords, (uint64_t *) my_ll,
                                                    (const uint64_t *) peer_ll, (int *) counters, slot_words);
    } else {
        har_kernel<__nv_bfloat16><<<nb, HAR_THREADS, 0, s>>>((const uint32_t *) in, (uint32_t *) out, nwords,
                                                             (uint64_t *) my_ll, (const uint64_t *) peer_ll,
                                                             (int *) counters, slot_words);
    }
    return (int) cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------------------------
// Compressed large all-reduce (prefill-sized operands, NCCL all-gather in between): int8 with an fp16 scale per
// 128-element group, about half the PCIe bytes of fp16. Both ranks sum the dequantized operands of both ranks
// (their own included), so the result is identical on every rank.
__global__ void ar8_quant_kernel(const half * __restrict__ x, int8_t * __restrict__ q, half * __restrict__ s,
                                 const int ngroups) {
    const int g    = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (g >= ngroups) {
        return;
    }
    const uint2 raw = *reinterpret_cast<const uint2 *>(x + (size_t) g * 128 + lane * 4);
    const float2 a = __half22float2(*reinterpret_cast<const half2 *>(&raw.x));
    const float2 b = __half22float2(*reinterpret_cast<const half2 *>(&raw.y));
    float amax = fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    const half  sh  = __float2half(amax / 127.0f);
    const float sc  = __half2float(sh);
    const float inv = sc > 0.0f ? 1.0f / sc : 0.0f;
    char4 v;
    v.x = (signed char) fmaxf(-127.0f, fminf(127.0f, rintf(a.x * inv)));
    v.y = (signed char) fmaxf(-127.0f, fminf(127.0f, rintf(a.y * inv)));
    v.z = (signed char) fmaxf(-127.0f, fminf(127.0f, rintf(b.x * inv)));
    v.w = (signed char) fmaxf(-127.0f, fminf(127.0f, rintf(b.y * inv)));
    *reinterpret_cast<char4 *>(q + (size_t) g * 128 + lane * 4) = v;
    if (lane == 0) {
        s[g] = sh;
    }
}

__global__ void ar8_dequant_sum_kernel(const int8_t * __restrict__ q0, const half * __restrict__ s0,
                                       const int8_t * __restrict__ q1, const half * __restrict__ s1,
                                       half * __restrict__ out, const int n4) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // 4 elements per thread
    if (i >= n4) {
        return;
    }
    const int   g  = i >> 5;   // 32 four-element units per 128-group
    const float a0 = __half2float(s0[g]), a1 = __half2float(s1[g]);
    const char4 u  = reinterpret_cast<const char4 *>(q0)[i];
    const char4 w  = reinterpret_cast<const char4 *>(q1)[i];
    half2 r0 = __floats2half2_rn(u.x * a0 + w.x * a1, u.y * a0 + w.y * a1);
    half2 r1 = __floats2half2_rn(u.z * a0 + w.z * a1, u.w * a0 + w.w * a1);
    uint2 o;
    o.x = *reinterpret_cast<uint32_t *>(&r0);
    o.y = *reinterpret_cast<uint32_t *>(&r1);
    reinterpret_cast<uint2 *>(out)[i] = o;
}

// x: n fp16 (n % 128 == 0); packed: n int8 then n / 128 fp16 scales
extern "C" int ar8_quant(const void * x, void * packed, int n, void * stream) {
    const int ng = n / 128;
    ar8_quant_kernel<<<(ng + 7) / 8, 256, 0, (cudaStream_t) stream>>>((const half *) x, (int8_t *) packed,
                                                                      (half *) ((int8_t *) packed + n), ng);
    return (int) cudaGetLastError();
}

// gathered: two packed blocks (rank order) of n + n / 64 bytes each
extern "C" int ar8_dequant_sum(const void * gathered, void * out, int n, void * stream) {
    const int8_t * b0 = (const int8_t *) gathered;
    const int8_t * b1 = b0 + n + n / 64;
    const int n4 = n / 4;
    ar8_dequant_sum_kernel<<<(n4 + 255) / 256, 256, 0, (cudaStream_t) stream>>>(
        b0, (const half *) (b0 + n), b1, (const half *) (b1 + n), (half *) out, n4);
    return (int) cudaGetLastError();
}
