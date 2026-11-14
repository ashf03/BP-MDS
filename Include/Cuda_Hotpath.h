#ifndef BPMDS_CUDA_HOTPATH_H
#define BPMDS_CUDA_HOTPATH_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if a CUDA device is usable, else 0. */
int bpmds_cuda_available(void);

/*
 * Dense Prim on squared Euclidean distances for one bucket.
 * xs/ys length n (host). Writes parent[0..n-1] with parent[0]=-1.
 * Uses O(n) device memory only (no n×n cost matrix).
 * Returns 0 on success, non-zero on CUDA/error (caller should fall back to CPU).
 */
int bpmds_cuda_prim_mst(
    const double* xs,
    const double* ys,
    int n,
    int* parent_out);

/*
 * Best-improvement 2-opt on a single open route (customers only; depot is
 * virtual ends). cities length ncities (node ids), coords via cx/cy indexed
 * by node id (length n_nodes). Mutates cities in place.
 * Returns 0 on success, non-zero to fall back to CPU.
 */
int bpmds_cuda_tsp_2opt(
    int* cities,
    int ncities,
    const double* cx,
    const double* cy,
    int n_nodes,
    int depot);

#ifdef __cplusplus
}
#endif

#endif
