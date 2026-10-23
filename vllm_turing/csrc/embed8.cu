// INT8 vocab-parallel embedding lookup in one kernel (embed8.py): per token, the shard-local row (original vocab range
// or added-token range, as vLLM's get_masked_input_and_mask), dequantized row = q[row] * s[row] as fp16, zero rows for
// tokens owned by other TP ranks (the caller all-reduces). Replaces mask + gather + scale + masked_fill (~6 kernels).
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename I>
__global__ void __launch_bounds__(128) embed8_kernel(const I * __restrict__ ids, const int8_t * __restrict__ q,
                                                    const float * __restrict__ s, half * __restrict__ out, const int D,
                                                    const int64_t org_start, const int64_t org_end,
                                                    const int64_t added_start, const int64_t added_end,
                                                    const int64_t added_offset) {
    const int64_t t  = blockIdx.x;
    const int64_t id = (int64_t) ids[t];
    int64_t       row = -1;
    if (id >= org_start && id < org_end) {
        row = id - org_start;
    } else if (id >= added_start && id < added_end) {
        row = id - added_offset;
    }
    uint4 *       o  = reinterpret_cast<uint4 *>(out + t * D);   // 8 halves per uint4
    const int     D8 = D >> 3;
    if (row < 0) {
        for (int j = threadIdx.x; j < D8; j += blockDim.x) {
            o[j] = make_uint4(0, 0, 0, 0);
        }
        return;
    }
    const float   sc = s[row];
    const uint2 * qr = reinterpret_cast<const uint2 *>(q + row * D);   // 8 int8 per uint2
    for (int j = threadIdx.x; j < D8; j += blockDim.x) {
        const uint2 v = __ldg(qr + j);
        half2       h[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w = i < 2 ? v.x : v.y;
            const int      sh = (i & 1) * 16;
            h[i] = __floats2half2_rn((float) (int8_t) ((w >> sh) & 0xff) * sc, (float) (int8_t) ((w >> (sh + 8)) & 0xff) * sc);
        }
        o[j] = *reinterpret_cast<uint4 *>(h);
    }
}

// ids: int32 (ids64 = 0) or int64 (1) [T]; q int8 [rows, D] (D % 8 == 0); s fp32 [rows]; out fp16 [T, D]
extern "C" int embed8_lookup(const void * ids, int ids64, const void * q, const void * s, void * out, int T, int D,
                             long long org_start, long long org_end, long long added_start, long long added_end,
                             long long added_offset, void * stream) {
    if (T <= 0) {
        return 0;
    }
    if ((D & 7) != 0) {
        return -1;
    }
    cudaStream_t st = (cudaStream_t) stream;
    if (ids64) {
        embed8_kernel<int64_t><<<T, 128, 0, st>>>((const int64_t *) ids, (const int8_t *) q, (const float *) s,
                                                 (half *) out, D, org_start, org_end, added_start, added_end,
                                                 added_offset);
    } else {
        embed8_kernel<int32_t><<<T, 128, 0, st>>>((const int32_t *) ids, (const int8_t *) q, (const float *) s,
                                                 (half *) out, D, org_start, org_end, added_start, added_end,
                                                 added_offset);
    }
    return (int) cudaGetLastError();
}
