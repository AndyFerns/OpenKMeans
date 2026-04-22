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
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")


# ── Dynamic Naming ─────────────────────────────────────────────────

def generate_results_path(input_file, k, mode):
    """
    Build the results CSV path using the same naming convention
    as the C engine's generate_output_filename():
        results/<basename>_k<k>_<mode>.csv
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(RESULTS_DIR, f"{base}_k{k}_{mode}.csv")


def generate_plot_path(input_file, k, mode):
    """
    Build the plot image path matching plot.py's generate_paths():
        results/<basename>_k<k>_<mode>_plot.png
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(RESULTS_DIR, f"{base}_k{k}_{mode}_plot.png")


# ── Input Helpers ──────────────────────────────────────────────────

def banner():
    """Print a nice banner."""
    print()
    print("+" + "=" * 42 + "+")
    print("|     OpenKMeans -- Terminal Interface      |")
    print("+" + "=" * 42 + "+")
    print()


def get_int(prompt, default):
    """Prompt for an integer with a default value."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  [ERROR] Invalid input, using default ({default})")
        return default


def get_string(prompt, default):
    """Prompt for a string with a default value."""
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return default
    return raw


def get_choice(prompt, options, default):
    """Prompt for a choice from a list."""
    raw = input(f"  {prompt} ({'/'.join(options)}) [{default}]: ").strip().lower()
    if raw in options:
        return raw
    if raw == "":
        return default
    print(f"  [ERROR] Invalid choice, using default ({default})")
    return default


# ── Session State ──────────────────────────────────────────────────
# Track the last run parameters so "View Results" and "Generate Plot"
# know which file to use without re-prompting.

_last_input_file = DEFAULT_DATA
_last_k = 3
_last_mode = "both"


def run_kmeans(input_file, k, threads, mode, normalize):
    """Invoke the C executable and stream its output."""
    global _last_input_file, _last_k, _last_mode

    if not os.path.isfile(EXECUTABLE):
        print(f"\n  [ERROR] Executable not found: {EXECUTABLE}")
        print("          Run 'make build' first.\n")
        return

    if not os.path.isfile(input_file):
        print(f"\n  [ERROR] Dataset not found: {input_file}\n")
        return

    # Update session state
    _last_input_file = input_file
    _last_k = k
    _last_mode = mode

    cmd = [
        EXECUTABLE,
        "--k", str(k),
        "--input", input_file,
        "--mode", mode,
        "--threads", str(threads),
    ]
    if normalize:
        cmd.append("--normalize")

    # Show the expected output path(s)
    if mode == "both":
        print(f"\n  Output (seq): {generate_results_path(input_file, k, 'seq')}")
        print(f"  Output (omp): {generate_results_path(input_file, k, 'omp')}")
    else:
        print(f"\n  Output: {generate_results_path(input_file, k, mode)}")

    print(f"\n  [Running]: {' '.join(cmd)}")
    print("  " + "-" * 44)

    try:
        result = subprocess.run(
            cmd, capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            encoding="utf-8"
        )
        print(result.stdout)
        if result.stderr:
            print("  [STDERR]:", result.stderr)
    except Exception as e:
        print(f"  [ERROR] Error running executable: {e}")


def show_results():
    """Display the first few lines of the most recent results CSV."""
    # Determine the effective mode for lookup
    effective_mode = _last_mode if _last_mode != "both" else "omp"
    results_csv = generate_results_path(_last_input_file, _last_k, effective_mode)

    if not os.path.isfile(results_csv):
        print(f"\n  [ERROR] No results file found at: {results_csv}")
        print("          Run K-Means clustering first.\n")
        return

    print(f"\n  -- Cluster Results: {os.path.basename(results_csv)} (first 15 rows) --")
    with open(results_csv, "r") as f:
        for i, line in enumerate(f):
            if i > 15:
                print("  ... (truncated)")
                break
            print(f"  {line.rstrip()}")
    print()


def run_visualization():
    """Call the visualization script with dynamic naming parameters."""
    plot_script = os.path.join(SCRIPT_DIR, "..", "visualization", "plot.py")
    if not os.path.isfile(plot_script):
        print("  [ERROR] plot.py not found.")
        return

    effective_mode = _last_mode if _last_mode != "both" else "omp"
    results_csv = generate_results_path(_last_input_file, _last_k, effective_mode)
    plot_path = generate_plot_path(_last_input_file, _last_k, effective_mode)

    if not os.path.isfile(results_csv):
        print(f"\n  [ERROR] Results file not found: {results_csv}")
        print("          Run K-Means clustering first.\n")
        return

    print(f"  > Generating plot from: {os.path.basename(results_csv)}")
    print(f"  > Output plot:          {os.path.basename(plot_path)}")

    try:
        result = subprocess.run(
            [sys.executable, plot_script,
             _last_input_file, str(_last_k), effective_mode],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            encoding="utf-8"
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"  [STDERR]: {result.stderr}")

        if os.path.isfile(plot_path):
            print(f"  Plot saved to {plot_path}\n")
        else:
            print("  [WARN] Plot file was not created.\n")
    except Exception as e:
        print(f"  [ERROR]: {e}")


def main():
    # Ensure UTF-8 output on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    banner()

    while True:
        print("  +-----------------------------------+")
        print("  |  1. Run K-Means Clustering        |")
        print("  |  2. View Results                   |")
        print("  |  3. Generate Visualization         |")
        print("  |  4. Exit                           |")
        print("  +-----------------------------------+")

        choice = input("  Select option: ").strip()

        if choice == "1":
            print()
            input_file = get_string("Dataset path", _last_input_file)
            k          = get_int("Number of clusters (k)", _last_k)
            threads    = get_int("Number of threads", 4)
            mode       = get_choice("Mode", ["seq", "omp", "both"], "both")
            norm_in    = get_choice("Normalise data?", ["y", "n"], "y")
            normalize  = (norm_in == "y")
            run_kmeans(input_file, k, threads, mode, normalize)

        elif choice == "2":
            show_results()

        elif choice == "3":
            run_visualization()

        elif choice == "4":
            print("\n  Goodbye!\n")
            break

        else:
            print("  [ERROR] Invalid option.\n")


if __name__ == "__main__":
    main()
