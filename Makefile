# ──────────────────────────────────────────────────────────────────
# OpenKMeans — Makefile
#
# Targets:
#   make build   — Compile the C project with OpenMP support
#   make run     — Build and run with default settings
#   make clean   — Remove compiled binary and results
#   make plot    — Generate cluster visualization
#   make tui     — Launch the terminal interface
#   make gui     — Launch the graphical interface
# ──────────────────────────────────────────────────────────────────

CC       = gcc
CFLAGS   = -Wall -Wextra -O2 -fopenmp
LDFLAGS  = -lm -fopenmp

# Source files
SRC_CORE     = src/core/kmeans.c src/core/io.c src/core/utils.c
SRC_PARALLEL = src/parallel/omp_kmeans.c
SRC_MAIN     = src/main.c
SOURCES      = $(SRC_MAIN) $(SRC_CORE) $(SRC_PARALLEL)

# Output binary
TARGET = kmeans.exe

# Include paths
INCLUDES = -Isrc

# ── Build ────────────────────────────────────────────────────────

build: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TARGET) $(SOURCES) $(LDFLAGS)
	@echo "✓ Build complete: $(TARGET)"

# ── Run ──────────────────────────────────────────────────────────

run: build
	./$(TARGET) --k 3 --mode both --threads 4 --normalize

# ── Clean ────────────────────────────────────────────────────────

clean:
	python -c "import os, glob; [os.remove(f) for f in glob.glob('results/*.csv') + glob.glob('results/*.png') + ['kmeans.exe'] if os.path.exists(f)]"
	@echo "✓ Cleaned build artifacts"

# ── Python Interfaces ────────────────────────────────────────────

tui: build
	python src/interface/tui.py

gui: build
	python src/interface/gui.py

plot:
	python src/visualization/plot.py

# ── Preprocessing ────────────────────────────────────────────────

preprocess:
	python scripts/preprocess.py --impute --stats

preprocess-pima:
	python scripts/preprocess.py --input data/pima-Indians-diabetes-dataset.csv \
	    --output data/pima_preprocessed.csv --impute --stats

preprocess-diabetes:
	python scripts/preprocess.py --input data/diabetes.csv \
	    --output data/diabetes_preprocessed.csv --impute --stats

# ── Phony ────────────────────────────────────────────────────────

.PHONY: build run clean tui gui plot preprocess preprocess-pima preprocess-diabetes
