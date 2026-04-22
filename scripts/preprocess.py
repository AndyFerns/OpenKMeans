#!/usr/bin/env python3
"""
preprocess.py — Pima Indians Diabetes Dataset Preprocessing & Column Mapping

Reads the raw Pima Indians Diabetes CSV (with or without a header),
maps all 9 columns to their proper names, performs data cleaning
(zero-value imputation), selects features for clustering, and
writes a clean 4-column CSV that the OpenKMeans C engine can ingest.

Column Mapping (Pima Indians Diabetes Dataset)
──────────────────────────────────────────────────────────────────
  Index │ Name                        │ Description
  ──────┼─────────────────────────────┼──────────────────────────
    0   │ Pregnancies                 │ Number of times pregnant
    1   │ Glucose                     │ Plasma glucose concentration (2h OGTT)
    2   │ BloodPressure               │ Diastolic blood pressure (mm Hg)
    3   │ SkinThickness               │ Triceps skin fold thickness (mm)
    4   │ Insulin                     │ 2-Hour serum insulin (mu U/ml)
    5   │ BMI                         │ Body mass index (kg/m^2)
    6   │ DiabetesPedigreeFunction    │ Diabetes pedigree function
    7   │ Age                         │ Age (years)
    8   │ Outcome                     │ Class variable (0 or 1)
  ──────┴─────────────────────────────┴──────────────────────────

Usage:
    python scripts/preprocess.py [options]

    --input  <file>    Raw CSV path          (default: data/pima-Indians-diabetes-dataset.csv)
    --output <file>    Output CSV path       (default: data/pima_preprocessed.csv)
    --features <list>  Comma-separated feature names to select (exactly 4)
                       (default: Glucose,BloodPressure,BMI,Age)
    --impute           Replace biologically impossible zeros with column median
    --stats            Print detailed column statistics
    --help             Show this message
"""

import os
import sys
import csv
import math

# ── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# ── Column Definitions ──────────────────────────────────────────────

PIMA_COLUMNS = [
    {
        "index": 0,
        "name": "Pregnancies",
        "description": "Number of times pregnant",
        "dtype": "int",
        "zero_is_valid": True,   # 0 pregnancies is valid
        "range": (0, 17),
        "mean": 3.85,
        "std": 3.37,
    },
    {
        "index": 1,
        "name": "Glucose",
        "description": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test",
        "dtype": "float",
        "zero_is_valid": False,  # 0 glucose is biologically impossible
        "range": (0, 199),
        "mean": 121.0,
        "std": 32.0,
    },
    {
        "index": 2,
        "name": "BloodPressure",
        "description": "Diastolic blood pressure (mm Hg)",
        "dtype": "float",
        "zero_is_valid": False,  # 0 BP is biologically impossible
        "range": (0, 122),
        "mean": 69.1,
        "std": 19.3,
    },
    {
        "index": 3,
        "name": "SkinThickness",
        "description": "Triceps skin fold thickness (mm)",
        "dtype": "float",
        "zero_is_valid": True,   # 0 could represent missing, but is common
        "range": (0, 99),
        "mean": 20.5,
        "std": 15.9,
    },
    {
        "index": 4,
        "name": "Insulin",
        "description": "2-Hour serum insulin (mu U/ml)",
        "dtype": "float",
        "zero_is_valid": True,   # 0 could represent missing, but is common
        "range": (0, 846),
        "mean": 79.8,
        "std": 115.0,
    },
    {
        "index": 5,
        "name": "BMI",
        "description": "Body mass index (weight in kg/(height in m)^2)",
        "dtype": "float",
        "zero_is_valid": False,  # 0 BMI is biologically impossible
        "range": (0, 67.1),
        "mean": 32.0,
        "std": 7.88,
    },
    {
        "index": 6,
        "name": "DiabetesPedigreeFunction",
        "description": "Diabetes pedigree function",
        "dtype": "float",
        "zero_is_valid": False,  # minimum is 0.078, never truly 0
        "range": (0.078, 2.42),
        "mean": 0.47,
        "std": 0.33,
    },
    {
        "index": 7,
        "name": "Age",
        "description": "Age (years)",
        "dtype": "int",
        "zero_is_valid": False,  # 0 age is impossible
        "range": (21, 81),
        "mean": 33.2,
        "std": 11.8,
    },
    {
        "index": 8,
        "name": "Outcome",
        "description": "Class variable (0 or 1) — 268 of 768 are 1, the rest are 0",
        "dtype": "int",
        "zero_is_valid": True,   # 0 = no diabetes
        "range": (0, 1),
        "mean": 0.349,
        "std": 0.477,
    },
]

# Map column names for quick lookup
COLUMN_NAME_MAP = {col["name"]: col for col in PIMA_COLUMNS}
COLUMN_NAMES    = [col["name"] for col in PIMA_COLUMNS]

# The 4 features the C engine expects, mapped to Pima dataset columns.
# Default selection: Glucose, BloodPressure, BMI, Age
# These are the most clinically relevant for diabetes clustering.
DEFAULT_FEATURES = ["Glucose", "BloodPressure", "BMI", "Age"]


# ── CSV Detection ──────────────────────────────────────────────────

def detect_header(filepath):
    """
    Auto-detect whether the CSV file has a header row.

    Strategy: read the first line — if every field is parseable as a
    number, it's a data row (no header).  If any field is non-numeric,
    it's a header.
    """
    with open(filepath, "r", newline="") as f:
        first_line = f.readline().strip().replace("\r", "")

    fields = first_line.split(",")
    for field in fields:
        try:
            float(field)
        except ValueError:
            return True  # contains text → header present
    return False  # all numeric → no header


# ── Data Loading ───────────────────────────────────────────────────

def load_pima_csv(filepath):
    """
    Load the Pima Indians Diabetes CSV into a list of dicts.
    Handles both header and no-header file variants.

    Returns:
        rows (list[dict]): Each row is {column_name: float_value, ...}
        has_header (bool): Whether the file had a header row
    """
    has_header = detect_header(filepath)

    rows = []
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)

        if has_header:
            header = next(reader)
            # Validate that the header matches expected columns
            for i, col_name in enumerate(header):
                col_name = col_name.strip()
                if i < len(COLUMN_NAMES) and col_name != COLUMN_NAMES[i]:
                    print(f"  [warn] Column {i} header '{col_name}' "
                          f"differs from expected '{COLUMN_NAMES[i]}'")

        for line_num, fields in enumerate(reader, start=2 if has_header else 1):
            if len(fields) < 9:
                print(f"  [warn] Line {line_num}: expected 9 columns, "
                      f"got {len(fields)} — skipping")
                continue

            row = {}
            for i, col in enumerate(PIMA_COLUMNS):
                try:
                    row[col["name"]] = float(fields[i].strip())
                except (ValueError, IndexError):
                    print(f"  [warn] Line {line_num}, col {col['name']}: "
                          f"invalid value '{fields[i].strip()}' — using 0")
                    row[col["name"]] = 0.0
            rows.append(row)

    return rows, has_header


# ── Statistics ─────────────────────────────────────────────────────

def compute_column_stats(rows, col_name):
    """Compute min, max, mean, std, zero count, and median for a column."""
    values = [r[col_name] for r in rows]
    n = len(values)

    if n == 0:
        return {"min": 0, "max": 0, "mean": 0, "std": 0,
                "zeros": 0, "median": 0, "count": 0}

    sorted_vals = sorted(values)
    total = sum(values)
    mean  = total / n
    var   = sum((v - mean) ** 2 for v in values) / n
    std   = math.sqrt(var)
    zeros = sum(1 for v in values if v == 0.0)

    # Median
    if n % 2 == 1:
        median = sorted_vals[n // 2]
    else:
        median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    return {
        "min":    sorted_vals[0],
        "max":    sorted_vals[-1],
        "mean":   mean,
        "std":    std,
        "zeros":  zeros,
        "median": median,
        "count":  n,
    }


def print_column_stats(rows):
    """Print a detailed statistical summary for each column."""
    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║         Pima Indians Diabetes Dataset — Column Statistics          ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Total records: {len(rows):<52} ║")
    print("╚══════════════════════════════════════════════════════════════════════╝\n")

    for col in PIMA_COLUMNS:
        stats = compute_column_stats(rows, col["name"])
        zero_pct = (stats["zeros"] / stats["count"] * 100) if stats["count"] > 0 else 0

        print(f"  ── {col['name']} {'─' * (50 - len(col['name']))}")
        print(f"     {col['description']}")
        print(f"     Count  : {stats['count']}")
        print(f"     Min    : {stats['min']:.4f}")
        print(f"     Max    : {stats['max']:.4f}")
        print(f"     Mean   : {stats['mean']:.4f}")
        print(f"     Std    : {stats['std']:.4f}")
        print(f"     Median : {stats['median']:.4f}")
        print(f"     Zeros  : {stats['zeros']} ({zero_pct:.1f}%)", end="")
        if not col["zero_is_valid"] and stats["zeros"] > 0:
            print(f"  ⚠ likely missing values")
        else:
            print()
        print()


# ── Imputation ─────────────────────────────────────────────────────

def impute_zeros(rows):
    """
    Replace biologically impossible zero values with the column median.

    Only applies to columns where zero_is_valid is False:
      - Glucose, BloodPressure, BMI, DiabetesPedigreeFunction, Age
    """
    imputed_counts = {}

    for col in PIMA_COLUMNS:
        if col["zero_is_valid"]:
            continue

        col_name = col["name"]

        # Compute median from non-zero values
        non_zero = [r[col_name] for r in rows if r[col_name] != 0.0]
        if not non_zero:
            continue

        non_zero.sort()
        n = len(non_zero)
        if n % 2 == 1:
            median = non_zero[n // 2]
        else:
            median = (non_zero[n // 2 - 1] + non_zero[n // 2]) / 2.0

        # Replace zeros with median
        count = 0
        for row in rows:
            if row[col_name] == 0.0:
                row[col_name] = median
                count += 1

        if count > 0:
            imputed_counts[col_name] = (count, median)

    if imputed_counts:
        print("\n── Zero-Value Imputation (median replacement) ──────")
        for col_name, (count, median) in imputed_counts.items():
            print(f"  {col_name:30s}: {count:3d} zeros → {median:.2f}")
        print("────────────────────────────────────────────────────\n")
    else:
        print("\n[preprocess] No zero-value imputation needed.\n")


# ── Feature Selection & Output ─────────────────────────────────────

def select_features(rows, feature_names):
    """
    Extract the specified features from each row.

    Args:
        rows: list of dicts with all Pima column values
        feature_names: list of 4 column names to select

    Returns:
        selected: list of dicts with only the chosen features
    """
    # Validate feature names
    for name in feature_names:
        if name not in COLUMN_NAME_MAP:
            print(f"[error] Unknown feature '{name}'. Valid features:")
            for col in PIMA_COLUMNS:
                print(f"         - {col['name']}")
            sys.exit(1)

    if len(feature_names) != 4:
        print(f"[error] Exactly 4 features required (got {len(feature_names)}).")
        print(f"        The C engine is compiled with NUM_FEATURES=4.")
        sys.exit(1)

    return [{name: row[name] for name in feature_names} for row in rows]


def write_output_csv(filepath, selected_rows, feature_names):
    """
    Write the selected features to a CSV file compatible with
    the OpenKMeans C engine (io.c → load_csv).

    The C engine expects:
      - Line 1: header row (skipped by load_csv)
      - Lines 2+: exactly 4 comma-separated numeric values
    """
    # Map feature names to the C engine's expected header names.
    # The C engine's save_results uses: Age, BloodPressure, Glucose, BMI
    # We preserve the original Pima names in the header so the
    # visualization/analysis tools can reference them properly.
    c_header_map = {
        "Age":           "Age",
        "BloodPressure": "BloodPressure",
        "Glucose":       "Glucose",
        "BMI":           "BMI",
        # For non-standard features, keep their original names
    }

    header = [c_header_map.get(name, name) for name in feature_names]

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)  # header row — load_csv skips this
        for row in selected_rows:
            writer.writerow([row[name] for name in feature_names])

    print(f"[preprocess] Wrote {len(selected_rows)} rows → '{filepath}'")
    print(f"             Columns: {', '.join(header)}")


# ── CLI ────────────────────────────────────────────────────────────

def print_usage():
    print("Usage: python scripts/preprocess.py [options]\n")
    print("  --input  <file>    Raw CSV path")
    print(f"                     (default: data/pima-Indians-diabetes-dataset.csv)")
    print("  --output <file>    Output CSV path")
    print(f"                     (default: data/pima_preprocessed.csv)")
    print("  --features <list>  Comma-separated feature names (exactly 4)")
    print(f"                     (default: {','.join(DEFAULT_FEATURES)})")
    print("  --impute           Replace impossible zeros with median")
    print("  --stats            Print detailed column statistics")
    print("  --help             Show this message\n")
    print("Available features:")
    for col in PIMA_COLUMNS:
        marker = "✓" if col["name"] in DEFAULT_FEATURES else " "
        print(f"  [{marker}] {col['name']:30s} — {col['description']}")


def main():
    # Ensure UTF-8 output on Windows (avoids cp1252 encoding errors)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Defaults
    input_file   = os.path.join(PROJECT_ROOT, "data", "pima-Indians-diabetes-dataset.csv")
    output_file  = os.path.join(PROJECT_ROOT, "data", "pima_preprocessed.csv")
    features     = DEFAULT_FEATURES[:]
    do_impute    = False
    do_stats     = False

    # Parse CLI args
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--input" and i + 1 < len(args):
            input_file = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_file = args[i + 1]
            i += 2
        elif args[i] == "--features" and i + 1 < len(args):
            features = [f.strip() for f in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--impute":
            do_impute = True
            i += 1
        elif args[i] == "--stats":
            do_stats = True
            i += 1
        elif args[i] == "--help":
            print_usage()
            return
        else:
            print(f"[error] Unknown argument: {args[i]}")
            print_usage()
            sys.exit(1)

    # ── Banner ──────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║      OpenKMeans — Pima Diabetes Dataset Preprocessor       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Input   : {input_file:<49}║")
    print(f"║  Output  : {output_file:<49}║")
    print(f"║  Features: {', '.join(features):<49}║")
    print(f"║  Impute  : {'yes' if do_impute else 'no':<49}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Load ────────────────────────────────────────────────────────
    if not os.path.isfile(input_file):
        print(f"\n[error] Input file not found: {input_file}")
        sys.exit(1)

    print(f"\n[preprocess] Loading '{input_file}'...")
    rows, has_header = load_pima_csv(input_file)
    print(f"[preprocess] Loaded {len(rows)} records "
          f"({'with' if has_header else 'without'} header)")

    # ── Column mapping report ───────────────────────────────────────
    print("\n── Column Mapping ──────────────────────────────────────────")
    for col in PIMA_COLUMNS:
        print(f"  [{col['index']}] {col['name']:30s} → {col['description']}")
    print("────────────────────────────────────────────────────────────\n")

    # ── Stats ───────────────────────────────────────────────────────
    if do_stats:
        print_column_stats(rows)

    # ── Imputation ──────────────────────────────────────────────────
    if do_impute:
        impute_zeros(rows)

    # ── Feature selection ───────────────────────────────────────────
    print(f"[preprocess] Selecting features: {features}")
    selected = select_features(rows, features)

    # ── Write output ────────────────────────────────────────────────
    write_output_csv(output_file, selected, features)

    print(f"\n[preprocess] Done! You can now run:")
    print(f"  ./kmeans.exe --input {output_file} --k 3 --mode both --normalize")
    print()


if __name__ == "__main__":
    main()
