/*
 * omp_kmeans.c — OpenMP-parallel K-Means clustering
 *
 * ALL OpenMP logic is confined to this file.
 * The parallelism targets the distance computation + cluster assignment
 * loop, which is the most expensive part of K-Means.
 *
 * Timing is measured with omp_get_wtime().
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>

#include "omp_kmeans.h"
#include "../core/kmeans.h"   /* euclidean_distance, init_centroids, etc. */

/* ── Parallel K-Means ───────────────────────────────────────────── */

int run_kmeans_parallel(DataPoint *data, int n, int k,
                        double centroids[][NUM_FEATURES],
                        int max_iter, int num_threads)
{
    /* Set the thread count for this run */
    omp_set_num_threads(num_threads);
    printf("  [omp] Using %d thread(s)\n", num_threads);

    /* Scratch space for centroid recomputation */
    double (*new_centroids)[NUM_FEATURES] =
        malloc(k * sizeof(double[NUM_FEATURES]));
    int *counts = (int *)malloc(k * sizeof(int));

    if (!new_centroids || !counts) {
        fprintf(stderr, "[omp] Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    double t_start = omp_get_wtime();

    int iter;
    for (iter = 0; iter < max_iter; iter++) {

        /* ──────────────────────────────────────────────────────────
         * Step 1: PARALLEL distance computation + cluster assignment
         *
         * Each thread independently processes a chunk of data points.
         * Since each point's cluster field is written only by the
         * thread that owns it, there are no race conditions.
         * ────────────────────────────────────────────────────────── */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            double min_dist = DBL_MAX;
            int    best     = 0;

            for (int c = 0; c < k; c++) {
                double d = euclidean_distance(data[i].features, centroids[c]);
                if (d < min_dist) {
                    min_dist = d;
                    best     = c;
                }
            }
            data[i].cluster = best;
        }

        /* ──────────────────────────────────────────────────────────
         * Step 2: Recompute centroids (sequential — fast for small k)
         * ────────────────────────────────────────────────────────── */
        for (int c = 0; c < k; c++) {
            counts[c] = 0;
            for (int f = 0; f < NUM_FEATURES; f++)
                new_centroids[c][f] = 0.0;
        }

        for (int i = 0; i < n; i++) {
            int c = data[i].cluster;
            counts[c]++;
            for (int f = 0; f < NUM_FEATURES; f++)
                new_centroids[c][f] += data[i].features[f];
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) {
                for (int f = 0; f < NUM_FEATURES; f++)
                    new_centroids[c][f] /= counts[c];
            }
        }

        /* ──────────────────────────────────────────────────────────
         * Step 3: Check convergence
         * ────────────────────────────────────────────────────────── */
        double max_shift = 0.0;
        for (int c = 0; c < k; c++) {
            double shift = euclidean_distance(centroids[c], new_centroids[c]);
            if (shift > max_shift)
                max_shift = shift;
        }

        /* Copy new centroids */
        for (int c = 0; c < k; c++)
            for (int f = 0; f < NUM_FEATURES; f++)
                centroids[c][f] = new_centroids[c][f];

        if (max_shift < CONVERGENCE_THRESHOLD) {
            printf("  [omp] Converged at iteration %d (shift=%.6f)\n",
                   iter + 1, max_shift);
            iter++;
            break;
        }
    }

    double t_end = omp_get_wtime();

    if (iter == max_iter)
        printf("  [omp] Reached max iterations (%d)\n", max_iter);

    printf("  [omp] Parallel time: %.6f seconds\n", t_end - t_start);

    free(new_centroids);
    free(counts);
    return iter;
}
