/*
 * kmeans.h — Sequential K-Means clustering header
 *
 * Declares data structures and the sequential K-Means interface.
 * All parallel logic is kept separately in parallel/omp_kmeans.h.
 */

#ifndef KMEANS_H
#define KMEANS_H

/* Number of features per data point (Age, BP, Glucose, BMI) */
#define NUM_FEATURES 4

/* Default maximum iterations before forced convergence */
#define MAX_ITERATIONS 100

/* Convergence threshold — stop when centroid movement < this value */
#define CONVERGENCE_THRESHOLD 1e-4

/* ── Data Structures ────────────────────────────────────────────── */

/* A single data point with its assigned cluster label */
typedef struct {
    double features[NUM_FEATURES];  /* Age, BloodPressure, Glucose, BMI */
    int    cluster;                 /* Assigned cluster id (-1 = unassigned) */
} DataPoint;

/* ── Sequential K-Means API ─────────────────────────────────────── */

/*
 * Run K-Means clustering (sequential / single-threaded).
 *
 *   data       — array of DataPoints (features must be filled in)
 *   n          — number of data points
 *   k          — number of clusters
 *   centroids  — output array of k centroids (each NUM_FEATURES doubles)
 *   max_iter   — maximum number of iterations
 *
 * Returns the number of iterations performed.
 */
int run_kmeans_sequential(DataPoint *data, int n, int k,
                          double centroids[][NUM_FEATURES], int max_iter);

/*
 * Compute Euclidean distance between two feature vectors.
 */
double euclidean_distance(const double *a, const double *b);

/*
 * Initialise centroids by picking k random data points.
 */
void init_centroids(const DataPoint *data, int n, int k,
                    double centroids[][NUM_FEATURES]);

#endif /* KMEANS_H */
