/*
 * io.c — CSV input/output implementation
 *
 * Reads patient healthcare data from CSV and writes clustered results.
 * Expected CSV format: Age,BloodPressure,Glucose,BMI
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "io.h"

/* Maximum length of a single CSV line */
#define MAX_LINE_LENGTH 256

/* ── Load CSV ───────────────────────────────────────────────────── */

/*
 * Parse a CSV file into a DataPoint array.
 * Skips the first row (header).  Expects exactly 4 numeric columns.
 */
DataPoint *load_csv(const char *filename, int *n) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[io] Error: cannot open '%s'\n", filename);
        return NULL;
    }

    /* --- First pass: count data rows ----------------------------- */
    char line[MAX_LINE_LENGTH];
    int count = 0;

    /* Skip header */
    if (fgets(line, MAX_LINE_LENGTH, fp) == NULL) {
        fprintf(stderr, "[io] Error: file '%s' is empty\n", filename);
        fclose(fp);
        return NULL;
    }

    while (fgets(line, MAX_LINE_LENGTH, fp))
        count++;

    if (count == 0) {
        fprintf(stderr, "[io] Error: no data rows in '%s'\n", filename);
        fclose(fp);
        return NULL;
    }

    /* --- Allocate ------------------------------------------------ */
    DataPoint *data = (DataPoint *)malloc(count * sizeof(DataPoint));
    if (!data) {
        fprintf(stderr, "[io] Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }

    /* --- Second pass: read values -------------------------------- */
    rewind(fp);
    fgets(line, MAX_LINE_LENGTH, fp);  /* skip header again */

    int idx = 0;
    while (fgets(line, MAX_LINE_LENGTH, fp) && idx < count) {
        double age, bp, glucose, bmi;
        if (sscanf(line, "%lf,%lf,%lf,%lf", &age, &bp, &glucose, &bmi) == 4) {
            data[idx].features[0] = age;
            data[idx].features[1] = bp;
            data[idx].features[2] = glucose;
            data[idx].features[3] = bmi;
            data[idx].cluster     = -1;  /* unassigned */
            idx++;
        }
    }

    fclose(fp);
    *n = idx;
    printf("[io] Loaded %d data points from '%s'\n", idx, filename);
    return data;
}

/* ── Save Results ───────────────────────────────────────────────── */

/*
 * Write the clustered data to a CSV file.
 * Output columns: Age,BloodPressure,Glucose,BMI,Cluster
 */
void save_results(const char *filename, const DataPoint *data, int n) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "[io] Error: cannot write to '%s'\n", filename);
        return;
    }

    fprintf(fp, "Age,BloodPressure,Glucose,BMI,Cluster\n");
    for (int i = 0; i < n; i++) {
        fprintf(fp, "%.2f,%.2f,%.2f,%.2f,%d\n",
                data[i].features[0],
                data[i].features[1],
                data[i].features[2],
                data[i].features[3],
                data[i].cluster);
    }

    fclose(fp);
    printf("[io] Results saved to '%s'\n", filename);
}
