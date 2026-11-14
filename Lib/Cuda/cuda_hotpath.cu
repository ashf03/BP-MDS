/*
 * Per-bucket CUDA Prim + 2-opt. Device memory is O(bucket) / O(route) —
 * never a full-instance dense N×N cost matrix.
 */
#include "Cuda_Hotpath.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace {

std::mutex g_cuda_mu;

constexpr double kInf = std::numeric_limits<double>::infinity();

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            return static_cast<int>(_e);                                       \
        }                                                                      \
    } while (0)

__global__ void prim_seed_kernel(
    const double* __restrict__ xs,
    const double* __restrict__ ys,
    double* __restrict__ best,
    int* __restrict__ parent,
    char* __restrict__ in_tree,
    int n)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) {
        return;
    }
    if (j == 0) {
        best[0] = 0.0;
        parent[0] = -1;
        in_tree[0] = 1;
        return;
    }
    const double dx = xs[0] - xs[j];
    const double dy = ys[0] - ys[j];
    best[j] = dx * dx + dy * dy;
    parent[j] = 0;
    in_tree[j] = 0;
}

/* One candidate (weight, index) per block; invalid → +inf / -1. */
__global__ void prim_block_min_kernel(
    const double* __restrict__ best,
    const char* __restrict__ in_tree,
    int n,
    double* __restrict__ block_w,
    int* __restrict__ block_i)
{
    __shared__ double s_w[256];
    __shared__ int s_i[256];
    const int tid = threadIdx.x;
    double local_w = kInf;
    int local_i = -1;

    for (int j = blockIdx.x * blockDim.x + tid; j < n; j += blockDim.x * gridDim.x) {
        if (!in_tree[j] && best[j] < local_w) {
            local_w = best[j];
            local_i = j;
        } else if (!in_tree[j] && best[j] == local_w && (local_i < 0 || j < local_i)) {
            local_i = j;
        }
    }
    s_w[tid] = local_w;
    s_i[tid] = local_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const double ow = s_w[tid + stride];
            const int oi = s_i[tid + stride];
            if (ow < s_w[tid] || (ow == s_w[tid] && oi >= 0 && (s_i[tid] < 0 || oi < s_i[tid]))) {
                s_w[tid] = ow;
                s_i[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        block_w[blockIdx.x] = s_w[0];
        block_i[blockIdx.x] = s_i[0];
    }
}

__global__ void prim_update_kernel(
    const double* __restrict__ xs,
    const double* __restrict__ ys,
    double* __restrict__ best,
    int* __restrict__ parent,
    const char* __restrict__ in_tree,
    int v,
    int n)
{
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= n || in_tree[w]) {
        return;
    }
    const double dx = xs[v] - xs[w];
    const double dy = ys[v] - ys[w];
    const double dw = dx * dx + dy * dy;
    if (dw < best[w]) {
        best[w] = dw;
        parent[w] = v;
    }
}

__device__ __forceinline__ double d2(
    double ax, double ay, double bx, double by)
{
    const double dx = ax - bx;
    const double dy = ay - by;
    return sqrt(dx * dx + dy * dy);
}

/* Find best improving 2-opt move; write (best_delta, i, k) for the grid. */
__global__ void two_opt_search_kernel(
    const double* __restrict__ rx,
    const double* __restrict__ ry,
    int ncities,
    double depot_x,
    double depot_y,
    double* __restrict__ out_delta,
    int* __restrict__ out_i,
    int* __restrict__ out_k)
{
    __shared__ double s_d[256];
    __shared__ int s_i[256];
    __shared__ int s_k[256];

    const int tid = threadIdx.x;
    double best_d = 0.0; /* only negative improvements */
    int best_i = -1;
    int best_k = -1;

    const int n_pairs = (ncities * (ncities - 1)) / 2;
    for (int p = blockIdx.x * blockDim.x + tid; p < n_pairs; p += blockDim.x * gridDim.x) {
        /* Map linear p → (i,k) with 0 ≤ i < k < ncities */
        int i = 0;
        int rem = p;
        int row = ncities - 1;
        while (rem >= row) {
            rem -= row;
            ++i;
            --row;
        }
        const int k = i + 1 + rem;

        const double ax = rx[i];
        const double ay = ry[i];
        const double bx = rx[k];
        const double by = ry[k];
        const double lx = (i == 0) ? depot_x : rx[i - 1];
        const double ly = (i == 0) ? depot_y : ry[i - 1];
        const double rxr = (k + 1 == ncities) ? depot_x : rx[k + 1];
        const double ryr = (k + 1 == ncities) ? depot_y : ry[k + 1];

        const double delta =
            d2(lx, ly, bx, by) + d2(ax, ay, rxr, ryr)
            - d2(lx, ly, ax, ay) - d2(bx, by, rxr, ryr);

        if (delta < best_d - 1e-12) {
            best_d = delta;
            best_i = i;
            best_k = k;
        }
    }

    s_d[tid] = best_d;
    s_i[tid] = best_i;
    s_k[tid] = best_k;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_d[tid + stride] < s_d[tid] - 1e-12) {
                s_d[tid] = s_d[tid + stride];
                s_i[tid] = s_i[tid + stride];
                s_k[tid] = s_k[tid + stride];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_delta[blockIdx.x] = s_d[0];
        out_i[blockIdx.x] = s_i[0];
        out_k[blockIdx.x] = s_k[0];
    }
}

int pick_min_host(
    const std::vector<double>& bw, const std::vector<int>& bi, int nblocks)
{
    double best_w = kInf;
    int best_i = -1;
    for (int b = 0; b < nblocks; ++b) {
        if (bi[b] < 0) {
            continue;
        }
        if (bw[b] < best_w || (bw[b] == best_w && (best_i < 0 || bi[b] < best_i))) {
            best_w = bw[b];
            best_i = bi[b];
        }
    }
    return best_i;
}

} // namespace

extern "C" int bpmds_cuda_available(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        return 0;
    }
    return 1;
}

extern "C" int bpmds_cuda_prim_mst(
    const double* xs, const double* ys, int n, int* parent_out)
{
    if (n <= 1) {
        if (n == 1) {
            parent_out[0] = -1;
        }
        return 0;
    }
    if (!bpmds_cuda_available()) {
        return 1;
    }

    std::lock_guard<std::mutex> lock(g_cuda_mu);

    double *d_xs = nullptr, *d_ys = nullptr, *d_best = nullptr, *d_bw = nullptr;
    int *d_parent = nullptr, *d_bi = nullptr;
    char* d_in = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_xs);
        cudaFree(d_ys);
        cudaFree(d_best);
        cudaFree(d_parent);
        cudaFree(d_in);
        cudaFree(d_bw);
        cudaFree(d_bi);
    };

    CUDA_CHECK(cudaMalloc(&d_xs, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_ys, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_best, sizeof(double) * n));
    CUDA_CHECK(cudaMalloc(&d_parent, sizeof(int) * n));
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(char) * n));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, sizeof(double) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ys, ys, sizeof(double) * n, cudaMemcpyHostToDevice));

    constexpr int TPB = 256;
    int grid = (n + TPB - 1) / TPB;
    prim_seed_kernel<<<grid, TPB>>>(d_xs, d_ys, d_best, d_parent, d_in, n);
    CUDA_CHECK(cudaGetLastError());

    const int nblocks = grid < 128 ? grid : 128;
    CUDA_CHECK(cudaMalloc(&d_bw, sizeof(double) * nblocks));
    CUDA_CHECK(cudaMalloc(&d_bi, sizeof(int) * nblocks));
    std::vector<double> h_bw(nblocks);
    std::vector<int> h_bi(nblocks);
    std::vector<int> h_parent(n);

    for (int added = 1; added < n; ++added) {
        prim_block_min_kernel<<<nblocks, TPB>>>(d_best, d_in, n, d_bw, d_bi);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(
            h_bw.data(), d_bw, sizeof(double) * nblocks, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_bi.data(), d_bi, sizeof(int) * nblocks, cudaMemcpyDeviceToHost));

        const int v = pick_min_host(h_bw, h_bi, nblocks);
        if (v < 0) {
            cleanup();
            return 2;
        }

        char one = 1;
        CUDA_CHECK(cudaMemcpy(d_in + v, &one, 1, cudaMemcpyHostToDevice));
        prim_update_kernel<<<grid, TPB>>>(d_xs, d_ys, d_best, d_parent, d_in, v, n);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(
        h_parent.data(), d_parent, sizeof(int) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        parent_out[i] = h_parent[i];
    }
    /* Ensure root */
    parent_out[0] = -1;

    cleanup();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" int bpmds_cuda_tsp_2opt(
    int* cities,
    int ncities,
    const double* cx,
    const double* cy,
    int /*n_nodes*/,
    int depot)
{
    if (ncities <= 2) {
        return 0;
    }
    if (!bpmds_cuda_available()) {
        return 1;
    }

    std::lock_guard<std::mutex> lock(g_cuda_mu);

    std::vector<double> rx(ncities), ry(ncities);
    for (int i = 0; i < ncities; ++i) {
        rx[i] = cx[cities[i]];
        ry[i] = cy[cities[i]];
    }
    const double depot_x = cx[depot];
    const double depot_y = cy[depot];

    double *d_rx = nullptr, *d_ry = nullptr, *d_delta = nullptr;
    int *d_i = nullptr, *d_k = nullptr;

    auto cleanup = [&]() {
        cudaFree(d_rx);
        cudaFree(d_ry);
        cudaFree(d_delta);
        cudaFree(d_i);
        cudaFree(d_k);
    };

    CUDA_CHECK(cudaMalloc(&d_rx, sizeof(double) * ncities));
    CUDA_CHECK(cudaMalloc(&d_ry, sizeof(double) * ncities));

    constexpr int TPB = 256;
    const int n_pairs = (ncities * (ncities - 1)) / 2;
    int nblocks = (n_pairs + TPB - 1) / TPB;
    if (nblocks > 128) {
        nblocks = 128;
    }
    if (nblocks < 1) {
        nblocks = 1;
    }
    CUDA_CHECK(cudaMalloc(&d_delta, sizeof(double) * nblocks));
    CUDA_CHECK(cudaMalloc(&d_i, sizeof(int) * nblocks));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(int) * nblocks));

    std::vector<double> h_delta(nblocks);
    std::vector<int> h_i(nblocks), h_k(nblocks);

    constexpr int kMaxPasses = 32;
    int improve = 0;
    int passes = 0;
    while (improve < 2 && passes < kMaxPasses) {
        ++passes;
        CUDA_CHECK(cudaMemcpy(
            d_rx, rx.data(), sizeof(double) * ncities, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            d_ry, ry.data(), sizeof(double) * ncities, cudaMemcpyHostToDevice));

        two_opt_search_kernel<<<nblocks, TPB>>>(
            d_rx, d_ry, ncities, depot_x, depot_y, d_delta, d_i, d_k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(
            h_delta.data(), d_delta, sizeof(double) * nblocks, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_i.data(), d_i, sizeof(int) * nblocks, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            h_k.data(), d_k, sizeof(int) * nblocks, cudaMemcpyDeviceToHost));

        double best_d = 0.0;
        int bi = -1, bk = -1;
        for (int b = 0; b < nblocks; ++b) {
            if (h_i[b] < 0) {
                continue;
            }
            if (h_delta[b] < best_d - 1e-12) {
                best_d = h_delta[b];
                bi = h_i[b];
                bk = h_k[b];
            }
        }

        if (bi >= 0 && bk > bi) {
            std::reverse(cities + bi, cities + bk + 1);
            for (int t = bi; t <= bk; ++t) {
                rx[t] = cx[cities[t]];
                ry[t] = cy[cities[t]];
            }
            improve = 0;
        } else {
            improve++;
        }
    }

    cleanup();
    return 0;
}
