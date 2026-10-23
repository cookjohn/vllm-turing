// Two-rank fp16 all-reduce on the copy engines over P2P (2x 2080 Ti NVLink): no SM spins or waits while the data moves,
// so it can overlap the prefill GEMMs at their full SM count (NCCL's kernels hold SMs for the whole transfer).
// Rank r owns half r of the tensor: (1) copy the peer's half of this rank's partial sum into the peer's receive buffer
// R1 (peer DMA push), (2) write the call's sequence number into the peer's flag 0 (stream memory operation, after a
// memory fence), (3) wait in the stream front-end until the own flag 0 reaches it, (4) out[own half] = x[own half] + R1
// (a small kernel), (5) push out[own half] into the peer's R2, (6) flag 1, (7) wait for the own flag 1, (8) copy R2 into
// out[peer half]. Both ranks then hold the same sums (fp16 a + b is commutative). Buffer reuse needs no extra flags:
// the next call's push into the peer's R1 follows (7), which follows the peer's (4) of this call; the next push into its
// R2 follows the next (3), which follows the peer's (8) of this call. So the peer is never more than one call ahead
// and the waits test for equality (wrap-around safe); sequence numbers start at 1 (flags are zeroed).
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int dbg = -1;
#define STEP(k) do { if (dbg < 0) dbg = getenv("CE_AR_DEBUG") != nullptr; if (dbg) fprintf(stderr, "ce_ar rank %d step %d\n", rank, k); } while (0)

typedef CUresult (*write_fn)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
typedef CUresult (*wait_fn)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
static write_fn p_write = nullptr;
static wait_fn p_wait = nullptr;

static int load_driver() {
    if (p_write && p_wait) return 0;
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
    if (!h) return 1;
    p_write = (write_fn)dlsym(h, "cuStreamWriteValue32_v2");
    p_wait = (wait_fn)dlsym(h, "cuStreamWaitValue32_v2");
    if (!p_write) p_write = (write_fn)dlsym(h, "cuStreamWriteValue32");
    if (!p_wait) p_wait = (wait_fn)dlsym(h, "cuStreamWaitValue32");
    return (p_write && p_wait) ? 0 : 2;
}

__device__ __forceinline__ void add8(uint4& x, const uint4& y) {
    half2* xp = reinterpret_cast<half2*>(&x);
    const half2* yp = reinterpret_cast<const half2*>(&y);
#pragma unroll
    for (int k = 0; k < 4; k++) xp[k] = __hadd2(xp[k], yp[k]);
}

// Runs next to the prefill GEMMs, whose blocks fill the register file: each block takes a whole SM, so it should leave
// soon. 512 threads with 8 independent 16-byte loads each in flight (vs 1-2) pull several times the bandwidth per SM.
constexpr int ADD_THREADS = 512;

__global__ void __launch_bounds__(ADD_THREADS) add_halves(const uint4* __restrict__ a, const uint4* __restrict__ b,
                                                          uint4* __restrict__ o, long n8) {
    const long stride = (long)gridDim.x * ADD_THREADS;
    long i = (long)blockIdx.x * ADD_THREADS + threadIdx.x;
    for (; i + 3 * stride < n8; i += 4 * stride) {
        uint4 x[4], y[4];
#pragma unroll
        for (int u = 0; u < 4; u++) {
            x[u] = a[i + u * stride];
            y[u] = b[i + u * stride];
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            add8(x[u], y[u]);
            o[i + u * stride] = x[u];
        }
    }
    for (; i < n8; i += stride) {
        uint4 x = a[i];
        add8(x, b[i]);
        o[i] = x;
    }
}

extern "C" {

// A zeroed device buffer and its CUDA IPC handle (64 bytes into handle). Also loads the add kernel on this device: with
// lazy module loading its first launch would synchronize, which deadlocks behind the stream's wait for the peer.
int ce_alloc(size_t bytes, void** ptr, void* handle) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, add_halves) != cudaSuccess) return 5;
    if (cudaMalloc(ptr, bytes) != cudaSuccess) return 1;
    if (cudaMemset(*ptr, 0, bytes) != cudaSuccess) return 2;
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    return cudaIpcGetMemHandle(reinterpret_cast<cudaIpcMemHandle_t*>(handle), *ptr) == cudaSuccess ? 0 : 4;
}

int ce_open(const void* handle, void** ptr) {
    cudaIpcMemHandle_t h;
    memcpy(&h, handle, sizeof(h));
    return cudaIpcOpenMemHandle(ptr, h, cudaIpcMemLazyEnablePeerAccess) == cudaSuccess ? 0 : 1;
}

// Single-process tests: peer access to another device's buffers.
int ce_enable_peer(int peer) {
    cudaError_t e = cudaDeviceEnablePeerAccess(peer, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return 0; }
    return e == cudaSuccess ? 0 : 1;
}

int ce_driver_ok() { return load_driver(); }

// x, out: n fp16 values (n % 16 == 0, 16-byte aligned); my / peer: this rank's and the peer's buffer base (flags in the
// first 256 bytes, then R1, then R2, cap bytes each); seq: 1, 2, ... (mod 2^32) in the same order on both ranks.
int ce_all_reduce(const void* x, void* out, long n, int rank, void* my, void* peer, long cap, unsigned seq, int blocks,
                  cudaStream_t st) {
    if (load_driver() != 0) return 1;
    if (n % 16 != 0 || ((uintptr_t)x | (uintptr_t)out) % 16 != 0) return 2;
    const long h = n / 2;                                   // a multiple of 8: halves stay 16-byte aligned
    const long own_off = rank == 0 ? 0 : h, own_len = rank == 0 ? h : n - h;
    const long oth_off = rank == 0 ? h : 0, oth_len = rank == 0 ? n - h : h;
    if (own_len * 2 > cap || oth_len * 2 > cap) return 3;
    const half* xs = static_cast<const half*>(x);
    half* os = static_cast<half*>(out);
    char* my_b = static_cast<char*>(my);
    char* peer_b = static_cast<char*>(peer);
    CUdeviceptr my_flag = (CUdeviceptr)my_b, peer_flag = (CUdeviceptr)peer_b;
    void* my_r1 = my_b + 256;
    void* my_r2 = my_b + 256 + cap;
    void* peer_r1 = peer_b + 256;
    void* peer_r2 = peer_b + 256 + cap;
    CUstream cs = (CUstream)st;
    STEP(1);
    if (cudaMemcpyAsync(peer_r1, xs + oth_off, oth_len * 2, cudaMemcpyDeviceToDevice, st) != cudaSuccess) return 4;
    STEP(2);
    if (p_write(cs, peer_flag, seq, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS) return 5;
    STEP(3);
    if (p_wait(cs, my_flag, seq, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS) return 6;
    STEP(4);
    const long n8 = own_len / 8;
    int grid = (int)((n8 + ADD_THREADS - 1) / ADD_THREADS);
    if (grid > blocks) grid = blocks;
    add_halves<<<grid, ADD_THREADS, 0, st>>>(reinterpret_cast<const uint4*>(xs + own_off),
                                     reinterpret_cast<const uint4*>(my_r1), reinterpret_cast<uint4*>(os + own_off), n8);
    if (cudaGetLastError() != cudaSuccess) return 7;
    STEP(5);
    if (cudaMemcpyAsync(peer_r2, os + own_off, own_len * 2, cudaMemcpyDeviceToDevice, st) != cudaSuccess) return 8;
    STEP(6);
    if (p_write(cs, peer_flag + 4, seq, CU_STREAM_WRITE_VALUE_DEFAULT) != CUDA_SUCCESS) return 9;
    STEP(7);
    if (p_wait(cs, my_flag + 4, seq, CU_STREAM_WAIT_VALUE_EQ) != CUDA_SUCCESS) return 10;
    STEP(8);
    if (cudaMemcpyAsync(os + oth_off, my_r2, oth_len * 2, cudaMemcpyDeviceToDevice, st) != cudaSuccess) return 11;
    STEP(9);
    return 0;
}

}  // extern "C"
