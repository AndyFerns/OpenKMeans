/*
 * main.c — OpenKMeans entry point
 *
 * Parses CLI arguments, loads data, runs sequential and/or parallel
 * K-Means, prints benchmarking results, and saves output.
 *
 * Usage:
 *   ./kmeans --k <clusters> --input <file> --mode <seq|omp|both>
 *            [--threads <n>] [--normalize]
 *
 * Defaults:
 *   k       = 3
 *   input   = data/patients.csv
 *   mode    = both
 *   threads = 4
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/kmeans.h"
#include "core/io.h"
#include "core/utils.h"
#include "parallel/omp_kmeans.h"

/* OpenMP header — only for timing in "both" mode */
#include <omp.h>

/* ── Helpers ────────────────────────────────────────────────────── */

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --k <int>          Number of clusters (default: 3)\n");
    printf("  --input <file>     Input CSV path (default: data/patients.csv)\n");
    printf("  --mode <mode>      seq | omp | both (default: both)\n");
    printf("  --threads <int>    OpenMP threads (default: 4)\n");
    printf("  --normalize        Normalise data before clustering\n");
    printf("  --help             Show this message\n");
}

/* ── Main ───────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {

    /* Default parameters */
    int   k          = 3;
    char *input_file = "data/patients.csv";
    char *mode       = "both";
    int   threads    = 4;
    int   normalize  = 0;

    /* ── Parse command-line arguments ──────────────────────────── */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_file = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            mode = argv[++i];
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--normalize") == 0) {
            normalize = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    /* ── Banner ─────────────────────────────────────────────────── */
    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║       OpenKMeans — HPC Clustering        ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Clusters : %-4d                         ║\n", k);
    printf("║  Input    : %-28s ║\n", input_file);
    printf("║  Mode     : %-28s ║\n", mode);
    printf("║  Threads  : %-4d                         ║\n", threads);
    printf("║  Normalize: %-4s                         ║\n", normalize ? "yes" : "no");
    printf("╚══════════════════════════════════════════╝\n\n");

    /* ── Load data ──────────────────────────────────────────────── */
    int n = 0;
    DataPoint *data = load_csv(input_file, &n);
    if (!data) {
        fprintf(stderr, "[main] Failed to load data. Exiting.\n");
        return 1;
    }

    print_dataset_info(data, n);

    if (normalize)
        normalize_data(data, n);

    /* ── Allocate centroids ─────────────────────────────────────── */
    double (*centroids)[NUM_FEATURES] = malloc(k * sizeof(double[NUM_FEATURES]));
    if (!centroids) {
        fprintf(stderr, "[main] Memory allocation failed\n");
        free(data);
        return 1;
    }

    /* ── Run Sequential ─────────────────────────────────────────── */
    double seq_time = 0.0;
    if (strcmp(mode, "seq") == 0 || strcmp(mode, "both") == 0) {
        printf("── Sequential K-Means ──────────────────\n");
        init_centroids(data, n, k, centroids);

        double t0 = omp_get_wtime();
        int iters = run_kmeans_sequential(data, n, k, centroids, MAX_ITERATIONS);
        double t1 = omp_get_wtime();
        seq_time  = t1 - t0;

        printf("  Iterations : %d\n", iters);
        printf("  Time       : %.6f seconds\n", seq_time);
        print_centroids(centroids, k);

        /* Save sequential results */
        if (strcmp(mode, "seq") == 0)
            save_results("results/clusters.csv", data, n);
    }

    /* ── Run Parallel (OpenMP) ──────────────────────────────────── */
    double par_time = 0.0;
    if (strcmp(mode, "omp") == 0 || strcmp(mode, "both") == 0) {
        printf("── Parallel K-Means (OpenMP) ───────────\n");

        /* Re-initialise: reset cluster assignments & pick fresh centroids */
        for (int i = 0; i < n; i++)
            data[i].cluster = -1;
        init_centroids(data, n, k, centroids);

        double t0 = omp_get_wtime();
        int iters = run_kmeans_parallel(data, n, k, centroids,
                                        MAX_ITERATIONS, threads);
        double t1 = omp_get_wtime();
        par_time  = t1 - t0;

        printf("  Iterations : %d\n", iters);
        printf("  Total time : %.6f seconds\n", par_time);
        print_centroids(centroids, k);

        /* Save parallel results (overwrite if "both") */
        save_results("results/clusters.csv", data, n);
    }

    /* ── Speedup Report ─────────────────────────────────────────── */
    if (strcmp(mode, "both") == 0 && par_time > 0.0) {
        printf("══════════════════════════════════════════\n");
        printf("  BENCHMARKING RESULTS\n");
        printf("══════════════════════════════════════════\n");
        printf("  Sequential Time : %.6f s\n", seq_time);
        printf("  Parallel Time   : %.6f s\n", par_time);
        printf("  Speedup         : %.2fx\n", seq_time / par_time);
        printf("══════════════════════════════════════════\n\n");
    }

    /* ── Cleanup ────────────────────────────────────────────────── */
    free(centroids);
    free(data);

    printf("[main] Done. Results saved to results/clusters.csv\n");
    return 0;
}
