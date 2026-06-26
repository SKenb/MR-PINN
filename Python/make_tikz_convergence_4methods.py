#!/usr/bin/env python3
"""
Generate standalone TikZ convergence figures for four methods:
PINN, rPINN nearest neighbor, rPINN linear interpolation, and rPINN spline interpolation.

This writes epoch-based CSVs plus two standalone .tex files:
- convergence.tex (MATLAB colors)
- convergence_bw.tex (black/white)

The x-axis is epoch count, not normalized progress.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

METHODS = ["pinn", "nearest", "linear", "spline"]
METHOD_LABELS = {
    "pinn": "PINN",
    "nearest": "rPINN: nearest neighbor",
    "linear": "rPINN: linear interpolation",
    "spline": "rPINN: spline interpolation",
}

MATLAB = {
    "pinn": ("#0072BD", "#D9EAF7"),
    "nearest": ("#D95319", "#F8DCCF"),
    "linear": ("#EDB120", "#FBEDC7"),
    "spline": ("#7E2F8E", "#E9D7EE"),
}


def read_loss_series(csv_path: Path, x_name: str, metric_name: str, std_name: str | None = None):
    epochs = []
    mean_vals = []
    std_vals = []
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(float(row[x_name]))
            mean_vals.append(float(row[metric_name]))
            if std_name is not None:
                std_vals.append(float(row[std_name]))
    return np.asarray(epochs, dtype=np.float64), np.asarray(mean_vals, dtype=np.float64), np.asarray(std_vals, dtype=np.float64)


def load_raw_histories(method_dir: Path):
    histories = []
    for path in sorted(method_dir.glob("loss_history_run*.npy")):
        histories.append(np.asarray(np.load(path), dtype=np.float64))
    if not histories:
        raise FileNotFoundError(f"No loss_history_run*.npy files found in {method_dir}")
    max_len = max(len(hist) for hist in histories)
    padded = np.full((len(histories), max_len), np.nan, dtype=np.float64)
    for idx, hist in enumerate(histories):
        padded[idx, : len(hist)] = hist
    epochs = np.arange(max_len, dtype=np.float64)
    mean_vals = np.nanmean(padded, axis=0)
    std_vals = np.nanstd(padded, axis=0)
    return epochs, mean_vals, std_vals


def write_epoch_csv(out_path: Path, epochs: np.ndarray, mean_vals: np.ndarray, std_vals: np.ndarray, prefix: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", f"{prefix}_mean_loss", f"{prefix}_std_loss"])
        for epoch, mean_val, std_val in zip(epochs, mean_vals, std_vals):
            writer.writerow([f"{epoch:.12g}", f"{mean_val:.12g}", f"{std_val:.12g}"])


def write_standalone_tex(out_path: Path, palette: str, xmax: float, tables: dict[str, str]):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if palette == "matlab":
        line_specs = {
            "pinn": ("pinnColor", "pinnFill", "solid"),
            "nearest": ("nearestColor", "nearestFill", "solid"),
            "linear": ("linearColor", "linearFill", "solid"),
            "spline": ("splineColor", "splineFill", "solid"),
        }
    else:
        line_specs = {
            "pinn": ("black", "black!12", "solid"),
            "nearest": ("black", "black!25", "dashed"),
            "linear": ("black", "black!40", "densely dashed"),
            "spline": ("black", "black!55", "dotted"),
        }

    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Standalone TikZ/PDF figure for four-method convergence\n")
        f.write("\\documentclass[tikz,border=3pt]{standalone}\n")
        f.write("\\usepackage{xcolor}\n")
        f.write("\\usepackage{pgfplots}\n")
        f.write("\\usepackage{pgfplotstable}\n")
        f.write("\\usepgfplotslibrary{fillbetween}\n")
        f.write("\\pgfplotsset{compat=1.18, every axis plot/.append style={line width=1.2pt}}\n")
        if palette == "matlab":
            f.write("\\definecolor{pinnColor}{HTML}{0072BD}\n")
            f.write("\\definecolor{nearestColor}{HTML}{D95319}\n")
            f.write("\\definecolor{linearColor}{HTML}{EDB120}\n")
            f.write("\\definecolor{splineColor}{HTML}{7E2F8E}\n")
            f.write("\\definecolor{pinnFill}{HTML}{D9EAF7}\n")
            f.write("\\definecolor{nearestFill}{HTML}{F8DCCF}\n")
            f.write("\\definecolor{linearFill}{HTML}{FBEDC7}\n")
            f.write("\\definecolor{splineFill}{HTML}{E9D7EE}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{tikzpicture}\n")
        f.write("\\begin{axis}[\n")
        f.write("    width=15.5cm,\n")
        f.write("    height=8.2cm,\n")
        f.write("    title={Training convergence},\n")
        f.write("    xlabel={Training epochs},\n")
        f.write("    ylabel={Training loss},\n")
        f.write("    ymode=log,\n")
        f.write(f"    xmin=0, xmax={xmax:.12g},\n")
        f.write(f"    xtick={{0,500,1000,1500,2000,2500,{int(xmax)}}},\n")
        f.write("    scaled x ticks=false,\n")
        f.write("    grid=both,\n")
        f.write("    legend pos=south west,\n")
        f.write("    legend cell align=left,\n")
        f.write("]\n")

        for method in METHODS:
            table_name = tables[method]
            line_color, fill_color, line_style = line_specs[method]
            f.write(f"\\pgfplotstableread[col sep=comma]{{{table_name}}}\\{method}convergence\n")

            f.write(f"% {METHOD_LABELS[method]}\n")
            f.write(f"\\addplot[name path={method}lower, draw=none, forget plot] table [x=epoch, y expr={{max(1e-12, \\thisrow{{{method}_mean_loss}}-\\thisrow{{{method}_std_loss}})}}] {{\\{method}convergence}};\n")
            f.write(f"\\addplot[name path={method}upper, draw=none, forget plot] table [x=epoch, y expr={{\\thisrow{{{method}_mean_loss}}+\\thisrow{{{method}_std_loss}}}}] {{\\{method}convergence}};\n")
            f.write(f"\\addplot[{fill_color}, forget plot] fill between[of={method}lower and {method}upper];\n")
            f.write(f"\\addplot[{line_color}, line width=1.2pt, {line_style}] table [x=epoch, y={method}_mean_loss] {{\\{method}convergence}};\n")
            f.write(f"\\addlegendentry{{{METHOD_LABELS[method]} ($\\mu \\pm \\sigma$)}}\n")

        f.write("\\end{axis}\n")
        f.write("\\end{tikzpicture}\n")
        f.write("\\end{document}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinn-csv", type=Path, required=True, help="Path to the normal benchmark convergence_tikz_data.csv")
    parser.add_argument("--history-root", type=Path, required=True, help="Folder containing per-method loss_history_run*.npy files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder for tikz files and CSVs")
    parser.add_argument("--pinn-epochs", type=int, default=2900, help="Epoch budget used for the PINN axis scale")
    parser.add_argument("--xmax", type=int, default=2900, help="Maximum epoch on the x-axis")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # PINN data comes from the normal benchmark's convergence CSV.
    pinn_x_norm, pinn_mean, pinn_std = read_loss_series(args.pinn_csv, "normalized_progress", "pinn_mean_loss", "pinn_std_loss")
    pinn_epochs = pinn_x_norm * float(args.pinn_epochs)
    write_epoch_csv(args.output_dir / "pinn_convergence.csv", pinn_epochs, pinn_mean, pinn_std, "pinn")

    method_csvs: dict[str, str] = {"pinn": str((args.output_dir / "pinn_convergence.csv").relative_to(args.output_dir))}

    for method in ["nearest", "linear", "spline"]:
        epochs, mean_vals, std_vals = load_raw_histories(args.history_root / method)
        write_epoch_csv(args.output_dir / f"{method}_convergence.csv", epochs, mean_vals, std_vals, method)
        method_csvs[method] = f"{method}_convergence.csv"

    write_standalone_tex(args.output_dir / "convergence.tex", "matlab", float(args.xmax), method_csvs)
    write_standalone_tex(args.output_dir / "convergence_bw.tex", "bw", float(args.xmax), method_csvs)

    print("Wrote:")
    for path in [
        args.output_dir / "pinn_convergence.csv",
        args.output_dir / "nearest_convergence.csv",
        args.output_dir / "linear_convergence.csv",
        args.output_dir / "spline_convergence.csv",
        args.output_dir / "convergence.tex",
        args.output_dir / "convergence_bw.tex",
    ]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()