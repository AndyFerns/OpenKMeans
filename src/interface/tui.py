#!/usr/bin/env python3
"""
tui.py — Terminal User Interface for OpenKMeans

Menu-driven TUI that:
  1. Asks the user for clustering parameters
  2. Calls the compiled C executable via subprocess
  3. Displays results cleanly in the terminal
"""

import subprocess
import sys
import os

# ── Paths ──────────────────────────────────────────────────────────
# Resolve paths relative to the project root (two levels up from this file)
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EXECUTABLE   = os.path.join(PROJECT_ROOT, "kmeans.exe")
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "patients.csv")
RESULTS_CSV  = os.path.join(PROJECT_ROOT, "results", "clusters.csv")


def banner():
    """Print a nice banner."""
    print()
    print("╔══════════════════════════════════════════╗")
    print("║     OpenKMeans — Terminal Interface      ║")
    print("╚══════════════════════════════════════════╝")
    print()


def get_int(prompt, default):
    """Prompt for an integer with a default value."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[ERROR] Invalid input, using default ({default})")
        return default


def get_choice(prompt, options, default):
    """Prompt for a choice from a list."""
    raw = input(f"  {prompt} ({'/'.join(options)}) [{default}]: ").strip().lower()
    if raw in options:
        return raw
    if raw == "":
        return default
    print(f"[ERROR]Invalid choice, using default ({default})")
    return default


def run_kmeans(k, threads, mode, normalize):
    """Invoke the C executable and stream its output."""
    if not os.path.isfile(EXECUTABLE):
        print(f"\n[ERROR] Executable not found: {EXECUTABLE}")
        print("     Run 'make build' first.\n")
        return

    cmd = [
        EXECUTABLE,
        "--k", str(k),
        "--input", DEFAULT_DATA,
        "--mode", mode,
        "--threads", str(threads),
    ]
    if normalize:
        cmd.append("--normalize")

    print("\n[Running]:", " ".join(cmd))
    print("  " + "─" * 44)

    try:
        result = subprocess.run(
            cmd, capture_output=True, 
            text=True,
            cwd=PROJECT_ROOT,
            encoding="utf-8"
        )
        print(result.stdout)
        if result.stderr:
            print("[STDERR] :", result.stderr)
    except Exception as e:
        print(f"[ERROR]Error running executable: {e}")


def show_results():
    """Display the first few lines of the results CSV."""
    if not os.path.isfile(RESULTS_CSV):
        print("[ERROR] No results file found.")
        return

    print("\n  ── Cluster Results (first 15 rows) ──────")
    with open(RESULTS_CSV, "r") as f:
        for i, line in enumerate(f):
            if i > 15:
                print("  ... (truncated)")
                break
            print(f"  {line.rstrip()}")
    print()


def run_visualization():
    """Call the visualization script."""
    plot_script = os.path.join(SCRIPT_DIR, "..", "visualization", "plot.py")
    if not os.path.isfile(plot_script):
        print("  [error]  plot.py not found.")
        return

    print("  ▶ Generating plot...")
    try:
        subprocess.run([sys.executable, plot_script], cwd=PROJECT_ROOT)
        print(" Plot saved to results/plot.png\n")
    except Exception as e:
        print(f"  [ERROR]: {e}")


def main():
    banner()

    while True:
        print("  ┌─────────────────────────────────┐")
        print("  │  1. Run K-Means Clustering      │")
        print("  │  2. View Results                 │")
        print("  │  3. Generate Visualization       │")
        print("  │  4. Exit                         │")
        print("  └─────────────────────────────────┘")

        choice = input("  Select option: ").strip()

        if choice == "1":
            print()
            k         = get_int("Number of clusters (k)", 3)
            threads   = get_int("Number of threads", 4)
            mode      = get_choice("Mode", ["seq", "omp", "both"], "both")
            norm_in   = get_choice("Normalise data?", ["y", "n"], "y")
            normalize = (norm_in == "y")
            run_kmeans(k, threads, mode, normalize)

        elif choice == "2":
            show_results()

        elif choice == "3":
            run_visualization()

        elif choice == "4":
            print("\n  Goodbye!\n")
            break

        else:
            print("  [ERROR]  Invalid option.\n")


if __name__ == "__main__":
    main()
