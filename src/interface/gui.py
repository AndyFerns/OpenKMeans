#!/usr/bin/env python3
"""
gui.py — Tkinter GUI for OpenKMeans

Provides a simple graphical interface to:
  - Set clustering parameters (k, threads, mode)
  - Select an input dataset via file browser
  - Run the C executable
  - Display output in a scrollable text area
  - Generate and display the cluster plot
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import subprocess
import threading
import re
import os
import sys
from PIL import Image, ImageTk

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
EXECUTABLE   = os.path.join(PROJECT_ROOT, "kmeans.exe")
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "patients.csv")
PLOT_SCRIPT  = os.path.join(SCRIPT_DIR, "..", "visualization", "plot.py")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")


# ── Dynamic Naming ─────────────────────────────────────────────────

def generate_results_path(input_file: str, k: int, mode: str) -> str:
    """
    Build the results CSV path matching the C engine's naming:
        results/<basename>_k<k>_<mode>.csv
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(RESULTS_DIR, f"{base}_k{k}_{mode}.csv")


def generate_plot_path(input_file: str, k: int, mode: str) -> str:
    """
    Build the plot image path matching plot.py's generate_paths():
        results/<basename>_k<k>_<mode>_plot.png
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(RESULTS_DIR, f"{base}_k{k}_{mode}_plot.png")


class OpenKMeansGUI:
    """Main GUI application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("OpenKMeans -- HPC K-Means Clustering")
        self.root.geometry("720x640")
        self.root.resizable(True, True)

        # ── State ──────────────────────────────────────────────
        # Mutable dataset path (changed by the file picker)
        self.data_path = DEFAULT_DATA

        # Image zoom state
        self._original_image: Image.Image | None = None
        self._zoom_level: float = 1.0
        self._tk_img: ImageTk.PhotoImage | None = None  # keep reference alive

        # Track which panel is currently shown ("output" or "plot")
        self._current_panel = "output"

        # Track the last plot path so View Plot uses the correct file
        self._last_plot_path: str | None = None

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

        # ── Dataset selector ────────────────────────────────────
        dataset_frame = ttk.LabelFrame(self.root, text="Dataset", padding=8)
        dataset_frame.pack(fill="x", padx=15, pady=5)

        self.dataset_label = tk.Label(
            dataset_frame,
            text=self._short_path(self.data_path),
            anchor="w",
            font=("Consolas", 9),
            fg="#555555",
            width=70,
        )
        self.dataset_label.pack(side="left", padx=(0, 8))

        ttk.Button(
            dataset_frame,
            text="  Browse...",
            command=self._on_browse_dataset,
        ).pack(side="left")

        # ── Buttons ─────────────────────────────────────────────
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=15, pady=5)

        self.run_btn = ttk.Button(btn_frame, text="Run K-Means",
                                  command=self._on_run)
        self.run_btn.pack(side="left", padx=5)

        self.plot_btn = ttk.Button(btn_frame, text="Generate Plot",
                                   command=self._on_plot)
        self.plot_btn.pack(side="left", padx=5)

        self.view_plot_btn = ttk.Button(btn_frame, text="View Plot",
                                        command=self._show_plot_panel)
        self.view_plot_btn.pack(side="left", padx=5)

        self.metrics_btn = ttk.Button(btn_frame, text="Metrics",
                                      command=self._display_metrics)
        self.metrics_btn.pack(side="left", padx=5)

        self.clear_btn = ttk.Button(btn_frame, text="Clear Output",
                                    command=self._on_clear)
        self.clear_btn.pack(side="left", padx=5)

        # ── Content area (output text OR plot image) ─────────────
        # Both panels live inside a single container so only one is
        # ever packed at a time and the status bar never moves.
        self._content_frame = tk.Frame(self.root)
        self._content_frame.pack(fill="both", expand=True, padx=15, pady=(0, 5))

        # -- Output panel --
        self.output = scrolledtext.ScrolledText(
            self._content_frame, wrap="word", font=("Consolas", 10),
            bg="#1e1e1e", fg="#d4d4d4", insertbackground="white"
        )
        self.output.pack(fill="both", expand=True)   # visible by default

        # -- Plot panel --
        self.plot_frame = tk.Frame(self._content_frame)
        # Not packed yet — shown on demand by _show_plot_panel()

        self.canvas = tk.Canvas(self.plot_frame, bg="#111111")
        self.scroll_y = tk.Scrollbar(self.plot_frame, orient="vertical",
                                     command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.plot_frame, orient="horizontal",
                                     command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set,
                              xscrollcommand=self.scroll_x.set)

        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)

        # Zoom via mouse wheel
        self.canvas.bind("<MouseWheel>", self._on_zoom)       # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_zoom)         # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_zoom)         # Linux scroll down

        # ── Status Bar ──────────────────────────────────────────
        self.status = tk.Label(self.root, text="Ready", anchor="w",
                               relief="sunken", font=("Helvetica", 9))
        self.status.pack(fill="x", side="bottom")

    # ── Dynamic naming helpers ──────────────────────────────────

    def _effective_mode(self) -> str:
        """Return the mode to use for file lookups (both → omp)."""
        mode = self.mode_var.get()
        return mode if mode != "both" else "omp"

    def _current_k(self) -> int:
        """Return the current k value, defaulting to 3 on parse error."""
        try:
            return int(self.k_var.get())
        except ValueError:
            return 3

    def _current_results_path(self) -> str:
        """Build the results CSV path for the current settings."""
        return generate_results_path(
            self.data_path, self._current_k(), self._effective_mode()
        )

    def _current_plot_path(self) -> str:
        """Build the plot image path for the current settings."""
        return generate_plot_path(
            self.data_path, self._current_k(), self._effective_mode()
        )

    # ── Dataset helper ──────────────────────────────────────────

    @staticmethod
    def _short_path(path: str, max_len: int = 70) -> str:
        """Truncate a long path with ellipsis for display."""
        return path if len(path) <= max_len else f"...{path[-(max_len - 3):]}"

    def _on_browse_dataset(self):
        """Open a file-picker dialog and update the active dataset path."""
        initial_dir = (os.path.dirname(self.data_path)
                       if os.path.isfile(self.data_path)
                       else PROJECT_ROOT)
        chosen = filedialog.askopenfilename(
            title="Select dataset",
            initialdir=initial_dir,
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("All files", "*.*"),
            ],
        )
        if chosen:                          # user didn't cancel
            self.data_path = chosen
            self.dataset_label.config(text=self._short_path(chosen))
            self._append(f"Dataset set to: {chosen}\n")
            self.status.config(text=f"Dataset: {os.path.basename(chosen)}")

    # ── Panel switching ─────────────────────────────────────────

    def _switch_to_output(self):
        """Show the scrolled-text output panel."""
        if self._current_panel != "output":
            self.plot_frame.pack_forget()
            self.output.pack(fill="both", expand=True)
            self._current_panel = "output"

    def _switch_to_plot(self):
        """Show the canvas / plot panel."""
        if self._current_panel != "plot":
            self.output.pack_forget()
            self.plot_frame.pack(fill="both", expand=True)
            self._current_panel = "plot"

    # ── Actions ─────────────────────────────────────────────────

    def _on_run(self):
        """Run the C executable in a background thread."""
        if not os.path.isfile(EXECUTABLE):
            messagebox.showerror(
                "Error",
                f"Executable not found:\n{EXECUTABLE}\n\nRun 'make build' first.",
            )
            return

        if not os.path.isfile(self.data_path):
            messagebox.showerror(
                "Error",
                f"Dataset not found:\n{self.data_path}\n\nChoose a valid CSV file.",
            )
            return

        self.run_btn.config(state="disabled")
        self.status.config(text="Running...")
        self._switch_to_output()

        cmd = [
            EXECUTABLE,
            "--k", self.k_var.get(),
            "--input", self.data_path,       # uses the selected dataset
            "--mode", self.mode_var.get(),
            "--threads", self.threads_var.get(),
        ]
        if self.normalize_var.get():
            cmd.append("--normalize")

        self._append(f"Running: {' '.join(cmd)}\n{'=' * 50}\n")

        threading.Thread(target=self._run_process, args=(cmd,),
                         daemon=True).start()

    def _run_process(self, cmd):
        """Execute the command and display output (background thread)."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                encoding="utf-8",
            )
            self.root.after(0, self._append, result.stdout)
            if result.stderr:
                self.root.after(0, self._append,
                                f"\nSTDERR:\n{result.stderr}")

            # Report the output file path
            results_path = self._current_results_path()
            msg = f"Results: {os.path.basename(results_path)}"
            self.root.after(0, self.status.config, {"text": msg})
        except Exception as e:
            self.root.after(0, self._append, f"\nError: {e}\n")
            self.root.after(0, self.status.config, {"text": "Error"})
        finally:
            self.root.after(0, self.run_btn.config, {"state": "normal"})

    def _on_plot(self):
        """Generate the cluster visualization with dynamic naming."""
        if not os.path.isfile(PLOT_SCRIPT):
            messagebox.showerror("Error", "plot.py not found.")
            return

        effective_mode = self._effective_mode()
        k = self._current_k()
        results_path = self._current_results_path()
        plot_path = self._current_plot_path()

        if not os.path.isfile(results_path):
            messagebox.showwarning(
                "No Results",
                f"Results file not found:\n{results_path}\n\n"
                "Run K-Means first to generate results.",
            )
            return

        self.status.config(text="Generating plot...")
        self._switch_to_output()
        self._append(f"Generating plot from: {os.path.basename(results_path)}\n")

        try:
            result = subprocess.run(
                [sys.executable, PLOT_SCRIPT,
                 self.data_path, str(k), effective_mode],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                encoding="utf-8",
            )
            if result.stdout:
                self._append(result.stdout)
            if result.stderr:
                self._append(f"STDERR: {result.stderr}")

            if os.path.isfile(plot_path):
                self._last_plot_path = plot_path
                self._append(
                    f"Plot saved: {os.path.basename(plot_path)}\n"
                )
                self.status.config(
                    text=f"Plot saved: {os.path.basename(plot_path)} "
                         "-- click 'View Plot' to display"
                )
            else:
                self.status.config(text="Plot generation finished (no output?)")

        except Exception as e:
            self._append(f"Error: {e}\n")
            self.status.config(text="Error")

    # ── Plot display & zoom ─────────────────────────────────────

    def _show_plot_panel(self):
        """Load the plot image and switch to the canvas panel."""
        # Use the last generated plot, or try the dynamically computed path
        plot_path = self._last_plot_path or self._current_plot_path()

        if not os.path.isfile(plot_path):
            self._switch_to_output()
            self._append(
                f"Plot image not found: {plot_path}\n"
                "Run 'Generate Plot' first.\n"
            )
            return

        # Load original at full resolution and reset zoom
        self._original_image = Image.open(plot_path)
        self._zoom_level = 1.0
        self._switch_to_plot()
        self._render_image()
        self.status.config(text=f"Viewing: {os.path.basename(plot_path)}")

    def _render_image(self):
        """Re-render the image at the current zoom level onto the canvas."""
        if self._original_image is None:
            return

        w = max(1, int(self._original_image.width  * self._zoom_level))
        h = max(1, int(self._original_image.height * self._zoom_level))

        resized = self._original_image.resize((w, h), Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)   # keep reference!

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        self.canvas.config(scrollregion=(0, 0, w, h))

    def _on_zoom(self, event):
        """Zoom in / out by re-rendering the image at a new scale."""
        # Normalise delta across platforms
        if event.num == 4:          # Linux scroll up
            delta = 1
        elif event.num == 5:        # Linux scroll down
            delta = -1
        else:                       # Windows / macOS
            delta = event.delta

        factor = 1.1 if delta > 0 else 0.9
        self._zoom_level = max(0.1, min(10.0, self._zoom_level * factor))
        self._render_image()

    # ── Metrics ─────────────────────────────────────────────────

    def _display_metrics(self):
        """Display performance metrics in a clean formatted block."""
        self._switch_to_output()
        text = self.output.get("1.0", tk.END)

        seq_match   = re.search(r"Sequential Time\s*:\s*([\d.]+)", text)
        par_match   = re.search(r"Parallel Time\s*:\s*([\d.]+)", text)
        speed_match = re.search(r"Speedup\s*:\s*([\d.]+)", text)

        if not seq_match or not par_match:
            self._append("Run K-Means in 'both' mode first to generate metrics.\n")
            return

        seq_time = float(seq_match.group(1))
        par_time = float(par_match.group(1))
        speedup  = (float(speed_match.group(1)) if speed_match
                    else (seq_time / par_time if par_time > 0 else 0.0))

        try:
            thread_count = max(1, int(self.threads_var.get()))
        except ValueError:
            thread_count = 1
        efficiency = speedup / thread_count

        report = f"""
==============================================
         PERFORMANCE ANALYSIS REPORT
==============================================

 Execution Summary:
  - Sequential Time : {seq_time:.6f} s
  - Parallel Time   : {par_time:.6f} s
  - Speedup         : {speedup:.2f}x
  - Efficiency      : {efficiency:.2f}

  Configuration:
  - Threads         : {self.threads_var.get()}
  - Clusters (k)    : {self.k_var.get()}
  - Mode            : {self.mode_var.get()}
  - Dataset         : {os.path.basename(self.data_path)}

 Interpretation:
"""
        if speedup > 1:
            report += "   Parallelization is effective (speedup achieved)\n"
        elif speedup == 1:
            report += "   No gain from parallelization\n"
        else:
            report += "   Overhead dominates (dataset too small)\n"

        report += """
 Recommendation:
  - Use larger datasets for better parallel performance
  - Tune thread count vs data size
  - Avoid excessive threads for small workloads

==============================================
"""
        self._append(report)

    # ── Clear ───────────────────────────────────────────────────

    def _on_clear(self):
        """Clear output and return to the text panel."""
        self._switch_to_output()
        self.output.delete("1.0", tk.END)
        self._original_image = None
        self._zoom_level = 1.0
        self.status.config(text="Ready")

    # ── Utility ─────────────────────────────────────────────────

    def _append(self, text: str):
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