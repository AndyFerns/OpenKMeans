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
from PIL import Image, ImageTk

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
        
        self.view_plot_btn = ttk.Button(btn_frame, text="🗺️ View Plot",
                                    command=self._display_plot_image)
        self.view_plot_btn.pack(side="left", padx=5)
        
        self.metrics_btn = ttk.Button(btn_frame, text="📝 Metrics",
                                    command=self._display_metrics)
        self.metrics_btn.pack(side="left", padx=5)

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
        
        # --- Image display frame ---
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(self.plot_frame, bg="black")
        self.scroll_y = tk.Scrollbar(self.plot_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.plot_frame, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.scroll_y.set,
                            xscrollcommand=self.scroll_x.set)

        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(fill="both", expand=True)
        
        # Bind mousewheel to zoom
        self.canvas.bind("<MouseWheel>", self._zoom)

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
            result = subprocess.run(
                cmd, 
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                encoding="utf-8"
            )
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
            result = subprocess.run(
                [sys.executable, PLOT_SCRIPT],
                capture_output=True, 
                text=True,
                cwd=PROJECT_ROOT, 
                encoding="utf-8"
            )
            self._append(result.stdout)
            if result.stderr:
                self._append(f"⚠ {result.stderr}")
            self.status.config(text="Plot saved to results/plot.png")

            # Show the plot if it exists
            if os.path.isfile(PLOT_IMAGE):
                self._append("✓ Plot saved to results/plot.png\n")
                # self._display_plot_image()

        except Exception as e:
            self._append(f"❌ Error: {e}\n")
            self.status.config(text="Error")

    def _zoom(self, event):
        """Standard mousewheel binded zoom"""
        scale = 1.1 if event.delta > 0 else 0.9
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        self.canvas.scale("all", x, y, scale, scale)
        
    def _display_plot_image(self):
        if not os.path.isfile(PLOT_IMAGE):
            self._append("⚠ Plot image not found\n")
            return
        
            # Hide text
        self.output.pack_forget()

        # Show plot area
        self.plot_frame.pack(fill="both", expand=True)

        # Clear previous image
        self.canvas.delete("all")

        img = Image.open(PLOT_IMAGE)

        # Optional: resize if too big
        img = img.resize((800, 600))

        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def _on_clear(self):
        """Clear the output area."""
        self.plot_frame.pack_forget()   # hide plot
        self.output.pack(fill="both", expand=True, padx=15, pady=10)
    
        self.output.delete("1.0", tk.END)
        self.status.config(text="Ready")
        
    def _display_metrics(self):
        """Display performance metrics for the given dataset + clusters and a final overview"""
        pass

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
