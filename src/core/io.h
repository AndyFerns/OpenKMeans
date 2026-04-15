/*
 * io.h — CSV input/output helpers
 *
 * Handles reading patient data from CSV and writing cluster results.
 */

#ifndef IO_H
#define IO_H

#include "kmeans.h"

/*
 * Load a CSV file into an array of DataPoints.
 *
 *   filename — path to the CSV (first row is a header, skipped)
 *   n        — (out) number of data points loaded
 *
 * Returns a dynamically allocated array of DataPoints, or NULL on error.
 * The caller is responsible for freeing the returned array.
 */
DataPoint *load_csv(const char *filename, int *n);

/*
 * Write cluster results to a CSV file.
 *
 *   filename — output path
 *   data     — array of DataPoints with cluster assignments
 *   n        — number of data points
 *
 * Output columns: Age, BloodPressure, Glucose, BMI, Cluster
 */
void save_results(const char *filename, const DataPoint *data, int n);

#endif /* IO_H */
