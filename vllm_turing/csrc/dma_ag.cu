// Two-rank all-gather by copy engines through pinned host shared memory (dmaag.py), for GPUs without peer access: each
// rank copies its payload (D2H) into its own staging region (NUMA-local to its GPU: the peer reads it across the socket
// link; measured 11.3 GB/s per direction with both directions at once vs ~5.7 GB/s for NCCL's SHM all-gather) and the
// peer copies it from there (H2D), pipelined in pieces. A 1-thread kernel after each D2H piece publishes the call's
// sequence number in a host flag (stream order: after the copy; PCIe keeps the GPU's posted writes in order); the peer's
// H2D of that piece waits behind a 1-thread kernel spinning on the flag. The SMs stay free for compute (NCCL's SHM
// kernels compete with it). Staging regions and flags are double-buffered by call parity: a rank writes buffer b of call
// s only after its own call s-1 finished, which needed the peer's call s-1 D2H, which the peer issues after its call s-2
// (the last reader of buffer b) completed.
#include <cuda_runtime.h>
#include <stdint.h>

static __global__ void dag_signal(volatile int * flag, const int seq) {
    __threadfence_system();
    *flag = seq;
    __threadfence_system();
}

static __global__ void dag_wait(volatile const int * flag, const int seq) {
    while (*flag < seq) {
        __nanosleep(200);
    }
    __threadfence_system();
}

// register the shared segment (host pointer, bytes): returns the device pointer of its start (flags are accessed by
// kernels through it) or 0
extern "C" void * dag_register(void * host, size_t bytes) {
    if (cudaHostRegister(host, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    void * dev = nullptr;
    if (cudaHostGetDevicePointer(&dev, host, 0) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return dev;
}

// gathered (device, 2*m bytes) holds this rank's payload in half `rank` already (written by the caller on `main`); the
// peer's payload lands in the other half. my_stage / peer_stage: host staging (m bytes) of this call's buffer;
// my_flag_dev / peer_flag_dev: device pointers of this call's P flags (int32) of each rank. Events: ev[0] main ready,
// ev[1] D2H done, ev[2] H2D done (created by the caller, reused).
extern "C" int dag_all_gather(void * gathered, long long m, int rank, void * my_stage, const void * peer_stage,
                              int * my_flag_dev, const int * peer_flag_dev, int seq, int P, void * main_, void * d2h_,
                              void * h2d_, void * ev0, void * ev1, void * ev2) {
    cudaStream_t main = (cudaStream_t) main_, d2h = (cudaStream_t) d2h_, h2d = (cudaStream_t) h2d_;
    cudaEvent_t e_main = (cudaEvent_t) ev0, e_d2h = (cudaEvent_t) ev1, e_h2d = (cudaEvent_t) ev2;
    char * g = (char *) gathered;
    const char * mine = g + (long long) rank * m;
    char * theirs = g + (long long) (1 - rank) * m;
    cudaEventRecord(e_main, main);
    cudaStreamWaitEvent(d2h, e_main, 0);
    cudaStreamWaitEvent(h2d, e_main, 0);   // the previous call's dequant (reads `theirs`) has run
    const long long piece = ((m + P - 1) / P + 255) / 256 * 256;
    int np = 0;
    for (long long off = 0; off < m; off += piece, ++np) {
        const long long len = m - off < piece ? m - off : piece;
        cudaMemcpyAsync((char *) my_stage + off, mine + off, len, cudaMemcpyDeviceToHost, d2h);
        dag_signal<<<1, 1, 0, d2h>>>(my_flag_dev + np, seq);
    }
    np = 0;
    for (long long off = 0; off < m; off += piece, ++np) {
        const long long len = m - off < piece ? m - off : piece;
        dag_wait<<<1, 1, 0, h2d>>>(peer_flag_dev + np, seq);
        cudaMemcpyAsync(theirs + off, (const char *) peer_stage + off, len, cudaMemcpyHostToDevice, h2d);
    }
    cudaEventRecord(e_d2h, d2h);
    cudaEventRecord(e_h2d, h2d);
    cudaStreamWaitEvent(main, e_d2h, 0);   // the next call's quant overwrites `mine`
    cudaStreamWaitEvent(main, e_h2d, 0);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
