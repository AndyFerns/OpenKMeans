/*
 * utils.h — Utility functions
 *
 * Data normalisation and other helper routines.
 */

#ifndef UTILS_H
#define UTILS_H

#include "kmeans.h"

/*
 * Min-Max normalise each feature to the [0, 1] range.
 *
 *   data — array of DataPoints (modified in place)
 *   n    — number of data points
 */
void normalize_data(DataPoint *data, int n);

/*
 * Print basic dataset statistics to stdout.
 */
void print_dataset_info(const DataPoint *data, int n);

/*
 * Print final centroid locations.
 */
void print_centroids(const double centroids[][NUM_FEATURES], int k);

#endif /* UTILS_H */
