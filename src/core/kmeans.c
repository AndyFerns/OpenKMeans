/*
 * kmeans.c — Sequential K-Means clustering implementation
 *
 * Contains:
 *   - Euclidean distance computation
 *   - Random centroid initialisation
 *   - Iterative cluster assignment + centroid update
 *   - Convergence detection
 *
 * NO parallelism here — all OpenMP logic lives in parallel/omp_kmeans.c.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "kmeans.h"

/* ── Distance ───────────────────────────────────────────────────── */

/*
 * Euclidean distance between two feature vectors of length NUM_FEATURES.
 * Returns sqrt( Σ (a[i] - b[i])² ).
 */
double euclidean_distance(const double *a, const double *b) {
    double sum = 0.0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

/* ── Centroid Initialisation ────────────────────────────────────── */

/*
 * Pick k distinct random data points as initial centroids.
 * Uses a simple Fisher-Yates-style selection.
 */
void init_centroids(const DataPoint *data, int n, int k,
                    double centroids[][NUM_FEATURES])
{
    /* Seed the PRNG once (main.c may also seed, but this is a safety net) */
    static int seeded = 0;
    if (!seeded) {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    /* Track which indices have already been chosen */
    int *chosen = (int *)calloc(n, sizeof(int));
    if (!chosen) {
        fprintf(stderr, "[kmeans] Memory allocation failed in init_centroids\n");
        exit(EXIT_FAILURE);
    }

    for (int c = 0; c < k; c++) {
        int idx;
        do {
            idx = rand() % n;
        } while (chosen[idx]);

        chosen[idx] = 1;

        for (int f = 0; f < NUM_FEATURES; f++) {
            centroids[c][f] = data[idx].features[f];
        }
    }
    free(chosen);
}

/* ── Sequential K-Means ────────────────────────────────────────── */

/*
 * Core algorithm:
 *   1. Assign every point to its nearest centroid.
 *   2. Recompute centroids as mean of assigned points.
 *   3. Repeat until convergence or max_iter reached.
 *
 * Returns the number of iterations actually performed.
 */
int run_kmeans_sequential(DataPoint *data, int n, int k,
                          double centroids[][NUM_FEATURES], int max_iter)
{
    /* Scratch space for new centroids and per-cluster counts */
    double (*new_centroids)[NUM_FEATURES] =
        malloc(k * sizeof(double[NUM_FEATURES]));
    int *counts = (int *)malloc(k * sizeof(int));

    if (!new_centroids || !counts) {
        fprintf(stderr, "[kmeans] Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    int iter;
    for (iter = 0; iter < max_iter; iter++) {

        /* ── Step 1: Assign each point to closest centroid ─────── */
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

        /* ── Step 2: Compute new centroids ─────────────────────── */
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

        /* ── Step 3: Check convergence ─────────────────────────── */
        double max_shift = 0.0;
        for (int c = 0; c < k; c++) {
            double shift = euclidean_distance(centroids[c], new_centroids[c]);
            if (shift > max_shift)
                max_shift = shift;
        }

        /* Copy new centroids → centroids */
        for (int c = 0; c < k; c++)
            for (int f = 0; f < NUM_FEATURES; f++)
                centroids[c][f] = new_centroids[c][f];

        if (max_shift < CONVERGENCE_THRESHOLD) {
            printf("  [seq] Converged at iteration %d (shift=%.6f)\n",
                   iter + 1, max_shift);
            iter++;  /* count this iteration */
            break;
        }
    }

    if (iter == max_iter)
        printf("  [seq] Reached max iterations (%d)\n", max_iter);

    free(new_centroids);
    free(counts);
    return iter;
}
