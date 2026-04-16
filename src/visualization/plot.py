#!/usr/bin/env python3
"""
plot.py — Cluster Visualization for OpenKMeans

Loads results/clusters.csv and produces:
  1. A matplotlib scatter plot (2D projection) saved to results/plot.png
  2. (Optional) A terminal-based plot using plotext if available

Uses the first two features (Age vs BloodPressure) as the 2D projection,
with points coloured by cluster assignment.
"""

import os
import sys
import csv

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
# RESULTS_CSV  = os.path.join(PROJECT_ROOT, "results", "clusters.csv")
# PLOT_OUTPUT  = os.path.join(PROJECT_ROOT, "results", "plot.png")


def load_results(filepath):
    """
    Load the clustered CSV data.
    Returns lists: ages, bps, glucoses, bmis, clusters.
    """
    ages, bps, glucoses, bmis, clusters = [], [], [], [], []

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ages.append(float(row["Age"]))
            bps.append(float(row["BloodPressure"]))
            glucoses.append(float(row["Glucose"]))
            bmis.append(float(row["BMI"]))
            clusters.append(int(row["Cluster"]))

    return ages, bps, glucoses, bmis, clusters


def generate_paths(input_file, k=3, mode="omp"):
    base = os.path.basename(input_file)
    name = os.path.splitext(base)[0]

    results_csv = os.path.join(PROJECT_ROOT, "results", f"{name}_k{k}_{mode}.csv")
    plot_output = os.path.join(PROJECT_ROOT, "results", f"{name}_k{k}_{mode}_plot.png")

    return results_csv, plot_output


def plot_matplotlib(ages, bps, glucoses, bmis, clusters, output_path):
    """
    Create a 2×2 subplot figure showing different feature pairs,
    coloured by cluster. Saves to results/plot.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[plot] matplotlib not found. Install it with: pip install matplotlib")
        return

    k = max(clusters) + 1
    colors = plt.get_cmap("tab10", k)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("OpenKMeans — Cluster Visualization", fontsize=16, fontweight="bold")

    # Feature pairs to plot
    pairs = [
        (ages,     bps,      "Age",           "Blood Pressure"),
        (ages,     glucoses, "Age",           "Glucose"),
        (bmis,     glucoses, "BMI",           "Glucose"),
        (bmis,     bps,      "BMI",           "Blood Pressure"),
    ]

    for ax, (x, y, xlabel, ylabel) in zip(axes.flat, pairs):
        for c in range(k):
            cx = [x[i] for i in range(len(x)) if clusters[i] == c]
            cy = [y[i] for i in range(len(y)) if clusters[i] == c]
            ax.scatter(cx, cy, c=[colors(c)], label=f"Cluster {c}",
                       alpha=0.6, edgecolors="white", linewidth=0.3, s=30)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved matplotlib plot to {output_path}")


def plot_terminal(ages, bps, clusters):
    """
    Optional: render a scatter plot in the terminal using plotext.
    Falls back gracefully if plotext is not installed.
    """
    try:
        import plotext as plt
    except ImportError:
        # plotext is optional — skip silently
        return

    plt.clear_figure()
    plt.title("OpenKMeans — Age vs Blood Pressure")

    k = max(clusters) + 1
    markers = ["dot", "cross", "star", "diamond", "triangle_up"]

    for c in range(k):
        cx = [ages[i] for i in range(len(ages)) if clusters[i] == c]
        cy = [bps[i]  for i in range(len(bps))  if clusters[i] == c]
        marker = markers[c % len(markers)]
        plt.scatter(cx, cy, label=f"Cluster {c}", marker=marker)

    plt.xlabel("Age")
    plt.ylabel("Blood Pressure")
    plt.show()


def main():
    if len(sys.argv) < 4:
        print("Usage: plot.py <input_file> <k> <mode>")
        sys.exit(1)

    input_file = sys.argv[1]
    k = int(sys.argv[2])
    mode = sys.argv[3]

    RESULTS_CSV, PLOT_OUTPUT = generate_paths(input_file, k, mode)

    if not os.path.isfile(RESULTS_CSV):
        print(f"[plot] Results file not found: {RESULTS_CSV}")
        sys.exit(1)

    print(f"[plot] Loading results from {RESULTS_CSV}")
    ages, bps, glucoses, bmis, clusters = load_results(RESULTS_CSV)
    print(f"[plot] Loaded {len(ages)} data points, {max(clusters) + 1} clusters")

    # Matplotlib plot (saved to file)
    plot_matplotlib(ages, bps, glucoses, bmis, clusters, PLOT_OUTPUT)

    # Terminal plot (if plotext is available)
    plot_terminal(ages, bps, clusters)


if __name__ == "__main__":
    main()
