// SiLU(gate) * up with per-row dynamic symmetric int8 quantization in one pass (the eager prefill layer tail: the MLP
// down projection's input): gate_up (M, 2I) fp16 -> q (M, I) int8, scales (M) float32. The activation row is kept in
// shared memory between the absmax and the quantization (no fp16 round trip through DRAM). Arithmetic as vLLM's
// silu_and_mul (SiLU in float rounded to fp16, the product in fp16) followed by its dynamic int8 quantization (scale
// absmax / 127, round to nearest even of x * 127 / absmax).
#include <cuda_fp16.h>
#include <stdint.h>

constexpr int THREADS = 256;

__device__ __forceinline__ half silu_h(half x) {
    float f = __half2float(x);
    return __float2half(f / (1.0f + expf(-f)));
}

__device__ __forceinline__ int8_t q8(float v) {
    float r = rintf(v);
    r = fminf(fmaxf(r, -128.0f), 127.0f);
    return (int8_t)r;
}

__global__ void __launch_bounds__(THREADS) silu_mul_quant_kernel(const half* __restrict__ gu, int8_t* __restrict__ q,
                                                               float* __restrict__ scales, int I) {
    extern __shared__ uint4 smem[];
    half* act = reinterpret_cast<half*>(smem);
    const long row = blockIdx.x;
    const uint4* g = reinterpret_cast<const uint4*>(gu + row * 2 * I);
    const uint4* u = reinterpret_cast<const uint4*>(gu + row * 2 * I + I);
    const int n8 = I / 8;
    float amax = 0.0f;
    for (int i = threadIdx.x; i < n8; i += THREADS) {
        uint4 gv = g[i], uv = u[i];
        const half* gh = reinterpret_cast<const half*>(&gv);
        const half* uh = reinterpret_cast<const half*>(&uv);
        uint4 rv;
        half* rh = reinterpret_cast<half*>(&rv);
#pragma unroll
        for (int k = 0; k < 8; k++) {
            rh[k] = __hmul(silu_h(gh[k]), uh[k]);
            amax = fmaxf(amax, fabsf(__half2float(rh[k])));
        }
        smem[i] = rv;
    }
    __shared__ float red[THREADS / 32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < THREADS / 32 ? red[threadIdx.x] : 0.0f;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
        if (threadIdx.x == 0) red[0] = v;
    }
    __syncthreads();
    amax = red[0];
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    if (threadIdx.x == 0) scales[row] = amax / 127.0f;
    uint2* qo = reinterpret_cast<uint2*>(q + row * I);
    for (int i = threadIdx.x; i < n8; i += THREADS) {
        uint4 rv = smem[i];
        const half* rh = reinterpret_cast<const half*>(&rv);
        uint2 ov;
        int8_t* ob = reinterpret_cast<int8_t*>(&ov);
#pragma unroll
        for (int k = 0; k < 8; k++) ob[k] = q8(__half2float(rh[k]) * inv);
        qo[i] = ov;
    }
}

extern "C" int silu_mul_quant(const void* gu, void* q, void* scales, int M, int I, cudaStream_t st) {
    if (I % 8 != 0 || I * 2 > 48 * 1024) return 1;
    if (M == 0) return 0;
    silu_mul_quant_kernel<<<M, THREADS, I * 2, st>>>(static_cast<const half*>(gu), static_cast<int8_t*>(q),
                                                     static_cast<float*>(scales), I);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}
