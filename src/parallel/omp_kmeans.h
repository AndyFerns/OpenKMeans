/*
 * omp_kmeans.h — OpenMP parallel K-Means header
 *
 * Exposes the parallel version of K-Means.
 * All OpenMP logic is isolated inside omp_kmeans.c.
 */

#ifndef OMP_KMEANS_H
#define OMP_KMEANS_H

#include "../core/kmeans.h"

/*
 * Run K-Means clustering using OpenMP parallelism.
 *
 *   data       — array of DataPoints
 *   n          — number of data points
 *   k          — number of clusters
 *   centroids  — output array of k centroids
 *   max_iter   — maximum iterations
 *   num_threads— number of OpenMP threads to use
 *
 * Returns the number of iterations performed.
 * Timing information is printed to stdout.
 */
int run_kmeans_parallel(DataPoint *data, int n, int k,
                        double centroids[][NUM_FEATURES],
                        int max_iter, int num_threads);

#endif /* OMP_KMEANS_H */
