# OpenKMeans

**Parallel K-Means Clustering for Healthcare Data** — An HPC project demonstrating OpenMP speedup on a from-scratch K-Means implementation.

---

## 📋 Overview

OpenKMeans clusters healthcare patient data (Age, BloodPressure, Glucose, BMI) using the K-Means algorithm, implemented in **C** with **OpenMP** parallelization. It includes Python interfaces (TUI + GUI) and matplotlib visualization.

### Key Features

- **From-scratch K-Means** — no external ML libraries
- **OpenMP parallelization** — isolated in `src/parallel/omp_kmeans.c`
- **Benchmarking** — sequential vs. parallel with speedup reporting
- **Data normalization** — Min-Max scaling to [0, 1]
- **Visualization** — 2D cluster scatter plots via matplotlib
- **Dual interface** — Terminal (TUI) and Tkinter (GUI)

---

## 🏗️ Project Structure

```
OpenKMeans/
├── data/
│   └── patients.csv            # Healthcare dataset
├── src/
│   ├── core/
│   │   ├── kmeans.c / .h       # Sequential K-Means
│   │   ├── io.c / .h           # CSV read/write
│   │   └── utils.c / .h        # Normalization, printing
│   ├── parallel/
│   │   ├── omp_kmeans.c / .h   # OpenMP parallel K-Means
│   ├── main.c                  # CLI entry point + benchmarking
│   ├── interface/
│   │   ├── tui.py              # Terminal menu interface
│   │   └── gui.py              # Tkinter GUI
│   └── visualization/
│       └── plot.py             # Matplotlib cluster plots
├── results/
│   ├── clusters.csv            # Output: clustered data
│   └── plot.png                # Output: visualization
├── Makefile
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **GCC** with OpenMP support (`gcc -fopenmp`)
- **Python 3** with `matplotlib` (`pip install matplotlib`)
- **Make** (GNU Make or compatible)

### Build & Run

```bash
# Build the project
make build

# Run with defaults (k=3, both modes, 4 threads, normalized)
make run

# Or run manually with custom options
./kmeans --k 4 --mode omp --threads 8 --normalize
```

### Python Interfaces

```bash
# Terminal UI
make tui
# or: python src/interface/tui.py

# Graphical UI
make gui
# or: python src/interface/gui.py
```

### Visualization

```bash
make plot
# or: python src/visualization/plot.py
```

---

## ⚙️ CLI Options

| Flag            | Description                     | Default              |
| --------------- | ------------------------------- | -------------------- |
| `--k <int>`     | Number of clusters              | `3`                  |
| `--input <file>`| Input CSV path                  | `data/patients.csv`  |
| `--mode <mode>` | `seq`, `omp`, or `both`         | `both`               |
| `--threads <n>` | Number of OpenMP threads        | `4`                  |
| `--normalize`   | Enable Min-Max normalization    | disabled             |
| `--help`        | Show usage information          |                      |

---

## 🧪 Benchmarking

When run in `both` mode, the program outputs:

```
══════════════════════════════════════════
  BENCHMARKING RESULTS
══════════════════════════════════════════
  Sequential Time : 0.001234 s
  Parallel Time   : 0.000456 s
  Speedup         : 2.71x
══════════════════════════════════════════
```

> **Note:** Speedup is most visible with larger datasets. The included 100-point dataset is for demonstration; for meaningful HPC benchmarks, use thousands of data points.

---

## 📊 Output

- **`results/clusters.csv`** — Original data with appended `Cluster` column
- **`results/plot.png`** — 2×2 scatter plot grid showing cluster assignments across feature pairs

---

## 🔧 Architecture & Design

### Separation of Concerns

| Module          | Responsibility                          |
| --------------- | --------------------------------------- |
| `core/`         | Pure sequential K-Means + I/O + utils   |
| `parallel/`     | OpenMP parallelization (isolated)       |
| `main.c`        | CLI parsing + orchestration             |
| `interface/`    | Python UIs (subprocess to C binary)     |
| `visualization/`| Matplotlib plotting (reads CSV output)  |

### OpenMP Strategy

- **What's parallelized:** Distance computation + cluster assignment (`O(n·k)` per iteration)
- **Pragma used:** `#pragma omp parallel for schedule(static)`
- **Thread safety:** Each thread writes only to its own data point's cluster field — no race conditions
- **Thread control:** Configurable via `--threads` flag or `OMP_NUM_THREADS` environment variable

---

## 📝 License

This project is for educational / academic purposes.
