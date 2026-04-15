#!/usr/bin/env python3
"""
gui.py — Tkinter GUI for OpenKMeans

Provides a simple graphical interface to:
  - Set clustering parameters (k, threads, mode)
  - Run the C executable
  - Display output in a scrollable text area
  - Generate and display the cluster plot
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import os
import sys

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EXECUTABLE   = os.path.join(PROJECT_ROOT, "kmeans.exe")
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "patients.csv")
PLOT_SCRIPT  = os.path.join(SCRIPT_DIR, "..", "visualization", "plot.py")
PLOT_IMAGE   = os.path.join(PROJECT_ROOT, "results", "plot.png")


class OpenKMeansGUI:
    """Main GUI application window."""

    def __init__(self, root):
        self.root = root
        self.root.title("OpenKMeans — HPC K-Means Clustering")
        self.root.geometry("720x600")
        self.root.resizable(True, True)

        self._build_ui()

    # ── UI Construction ─────────────────────────────────────────

    def _build_ui(self):
        """Build all widgets."""
        # Title
        title = tk.Label(self.root, text="OpenKMeans",
                         font=("Helvetica", 18, "bold"))
        title.pack(pady=(10, 0))

        subtitle = tk.Label(self.root,
                            text="Parallel K-Means Clustering for Healthcare Data",
                            font=("Helvetica", 10))
        subtitle.pack(pady=(0, 10))

        # ── Parameters Frame ────────────────────────────────────
        params = ttk.LabelFrame(self.root, text="Parameters", padding=10)
        params.pack(fill="x", padx=15, pady=5)

        # Clusters
        ttk.Label(params, text="Clusters (k):").grid(row=0, column=0,
                                                       sticky="w", padx=5)
        self.k_var = tk.StringVar(value="3")
        ttk.Entry(params, textvariable=self.k_var, width=8).grid(
            row=0, column=1, padx=5)

        # Threads
        ttk.Label(params, text="Threads:").grid(row=0, column=2,
                                                  sticky="w", padx=5)
        self.threads_var = tk.StringVar(value="4")
        ttk.Entry(params, textvariable=self.threads_var, width=8).grid(
            row=0, column=3, padx=5)

        # Mode
        ttk.Label(params, text="Mode:").grid(row=0, column=4,
                                               sticky="w", padx=5)
        self.mode_var = tk.StringVar(value="both")
        mode_combo = ttk.Combobox(params, textvariable=self.mode_var,
                                  values=["seq", "omp", "both"],
                                  width=6, state="readonly")
        mode_combo.grid(row=0, column=5, padx=5)

        # Normalise checkbox
        self.normalize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="Normalise data",
                        variable=self.normalize_var).grid(
            row=0, column=6, padx=10)

        # ── Buttons ─────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=5)

        self.run_btn = ttk.Button(btn_frame, text="▶  Run K-Means",
                                  command=self._on_run)
        self.run_btn.pack(side="left", padx=5)

        self.plot_btn = ttk.Button(btn_frame, text="📊 Generate Plot",
                                   command=self._on_plot)
        self.plot_btn.pack(side="left", padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="🗑  Clear Output",
                                    command=self._on_clear)
        self.clear_btn.pack(side="left", padx=5)

        # ── Output Area ─────────────────────────────────────────
        self.output = scrolledtext.ScrolledText(
            self.root, wrap="word", font=("Consolas", 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white"
        )
        self.output.pack(fill="both", expand=True, padx=15, pady=10)

        # ── Status Bar ──────────────────────────────────────────
        self.status = tk.Label(self.root, text="Ready", anchor="w",
                               relief="sunken", font=("Helvetica", 9))
        self.status.pack(fill="x", side="bottom")

    # ── Actions ─────────────────────────────────────────────────

    def _on_run(self):
        """Run the C executable in a background thread."""
        if not os.path.isfile(EXECUTABLE):
            messagebox.showerror("Error",
                                 f"Executable not found:\n{EXECUTABLE}\n\n"
                                 "Run 'make build' first.")
            return

        self.run_btn.config(state="disabled")
        self.status.config(text="Running...")

        # Build command
        cmd = [
            EXECUTABLE,
            "--k", self.k_var.get(),
            "--input", DEFAULT_DATA,
            "--mode", self.mode_var.get(),
            "--threads", self.threads_var.get(),
        ]
        if self.normalize_var.get():
            cmd.append("--normalize")

        self._append(f"▶ Running: {' '.join(cmd)}\n{'─' * 50}\n")

        # Run in background so UI doesn't freeze
        threading.Thread(target=self._run_process, args=(cmd,),
                         daemon=True).start()

    def _run_process(self, cmd):
        """Execute the command and display output."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    cwd=PROJECT_ROOT)
            self.root.after(0, self._append, result.stdout)
            if result.stderr:
                self.root.after(0, self._append,
                                f"\n⚠ STDERR:\n{result.stderr}")
            self.root.after(0, self.status.config, {"text": "Done"})
        except Exception as e:
            self.root.after(0, self._append, f"\n❌ Error: {e}\n")
            self.root.after(0, self.status.config, {"text": "Error"})
        finally:
            self.root.after(0, self.run_btn.config, {"state": "normal"})

    def _on_plot(self):
        """Generate the cluster visualization."""
        if not os.path.isfile(PLOT_SCRIPT):
            messagebox.showerror("Error", "plot.py not found.")
            return

        self.status.config(text="Generating plot...")
        self._append("▶ Generating cluster plot...\n")

        try:
            result = subprocess.run([sys.executable, PLOT_SCRIPT],
                                    capture_output=True, text=True,
                                    cwd=PROJECT_ROOT)
            self._append(result.stdout)
            if result.stderr:
                self._append(f"⚠ {result.stderr}")
            self.status.config(text="Plot saved to results/plot.png")

            # Show the plot if it exists
            if os.path.isfile(PLOT_IMAGE):
                self._append("✓ Plot saved to results/plot.png\n")
        except Exception as e:
            self._append(f"❌ Error: {e}\n")
            self.status.config(text="Error")

    def _on_clear(self):
        """Clear the output area."""
        self.output.delete("1.0", tk.END)
        self.status.config(text="Ready")

    def _append(self, text):
        """Append text to the output area and auto-scroll."""
        self.output.insert(tk.END, text)
        self.output.see(tk.END)


# ── Entry Point ─────────────────────────────────────────────────

def main():
    root = tk.Tk()
    OpenKMeansGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
