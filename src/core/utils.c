/*
 * utils.c — Utility functions
 *
 * Provides data normalisation (Min-Max scaling) and
 * pretty-printing helpers for dataset info and centroids.
 */

#include <stdio.h>
#include <float.h>
#include "utils.h"

/* ── Min-Max Normalisation ──────────────────────────────────────── */

/*
 * Scale every feature to [0, 1] using:
 *   x_norm = (x - min) / (max - min)
 *
 * Modifies data in place.
 */
void normalize_data(DataPoint *data, int n) {
    double min_val[NUM_FEATURES], max_val[NUM_FEATURES];

    /* Initialise min/max with first data point */
    for (int f = 0; f < NUM_FEATURES; f++) {
        min_val[f] =  DBL_MAX;
        max_val[f] = -DBL_MAX;
    }

    /* Find min and max for each feature */
    for (int i = 0; i < n; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            if (data[i].features[f] < min_val[f])
                min_val[f] = data[i].features[f];
            if (data[i].features[f] > max_val[f])
                max_val[f] = data[i].features[f];
        }
    }

    /* Apply normalisation */
    for (int i = 0; i < n; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            double range = max_val[f] - min_val[f];
            if (range > 0.0)
                data[i].features[f] = (data[i].features[f] - min_val[f]) / range;
            else
                data[i].features[f] = 0.0;  /* constant feature */
        }
    }

    printf("[utils] Data normalised to [0, 1]\n");
}

/* ── Print Helpers ──────────────────────────────────────────────── */

/*
 * Print a summary of the loaded dataset.
 */
void print_dataset_info(const DataPoint *data, int n) {
    printf("\n══════════════════════════════════════════\n");
    printf("  Dataset Summary\n");
    printf("══════════════════════════════════════════\n");
    printf("  Points  : %d\n", n);
    printf("  Features: %d (Age, BP, Glucose, BMI)\n", NUM_FEATURES);

    /* Compute simple stats */
    double min_v[NUM_FEATURES], max_v[NUM_FEATURES], sum_v[NUM_FEATURES];
    for (int f = 0; f < NUM_FEATURES; f++) {
        min_v[f] = DBL_MAX;
        max_v[f] = -DBL_MAX;
        sum_v[f] = 0.0;
    }
    for (int i = 0; i < n; i++) {
        for (int f = 0; f < NUM_FEATURES; f++) {
            if (data[i].features[f] < min_v[f]) min_v[f] = data[i].features[f];
            if (data[i].features[f] > max_v[f]) max_v[f] = data[i].features[f];
            sum_v[f] += data[i].features[f];
        }
    }

    const char *names[] = {"Age", "BP ", "Glu", "BMI"};
    printf("  ────────────────────────────────────────\n");
    printf("  Feature   Min       Max       Mean\n");
    printf("  ────────────────────────────────────────\n");
    for (int f = 0; f < NUM_FEATURES; f++) {
        printf("  %s     %8.2f  %8.2f  %8.2f\n",
               names[f], min_v[f], max_v[f], sum_v[f] / n);
    }
    printf("══════════════════════════════════════════\n\n");
}

/*
 * Print final centroid positions.
 */
void print_centroids(const double centroids[][NUM_FEATURES], int k) {
    printf("\n── Final Centroids ───────────────────────\n");
    printf("  Cluster   Age       BP        Glucose   BMI\n");
    printf("  ───────────────────────────────────────────\n");
    for (int c = 0; c < k; c++) {
        printf("  %3d     %8.2f  %8.2f  %8.2f  %8.2f\n",
               c, centroids[c][0], centroids[c][1],
               centroids[c][2], centroids[c][3]);
    }
    printf("──────────────────────────────────────────\n\n");
}
