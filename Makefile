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
	rm -f $(TARGET)
	rm -f results/clusters.csv results/plot.png
	@echo "✓ Cleaned build artifacts"

# ── Python Interfaces ────────────────────────────────────────────

tui: build
	python src/interface/tui.py

gui: build
	python src/interface/gui.py

plot:
	python src/visualization/plot.py

# ── Phony ────────────────────────────────────────────────────────

.PHONY: build run clean tui gui plot
