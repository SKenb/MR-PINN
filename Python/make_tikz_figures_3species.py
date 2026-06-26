from __future__ import annotations

import json
import subprocess
from pathlib import Path
import argparse
import os
import csv

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH_DIR = ROOT / "Data" / "rpinn_3species-longrun_30runs"
DEFAULT_OUT_DIR = ROOT / "tikz_3species_longrun_30runs"


def resolve_cli_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path

    candidate = (Path.cwd() / path).resolve(strict=False)
    if candidate.exists():
        return candidate

    if path.parts and path.parts[0] == "..":
        stripped = Path(*path.parts[1:])
        if stripped.parts:
            return (Path.cwd() / stripped).resolve(strict=False)

    return candidate


def fmt_num(x: float) -> str:
    ax = abs(x)
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.6e}"
    if ax < 1:
        return f"{x:.6f}".rstrip("0").rstrip(".")
    return f"{x:.4f}".rstrip("0").rstrip(".")


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=tex_path.parent,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def write_main_scores_tex(summary: dict[str, object], out_dir: Path) -> None:
    pinn = summary["PINN"]
    rpinn = summary["rPINN"]
    metrics = [
        ("train_time_s", "Train time [s]", True),
        ("outlet_mse", "Outlet MSE", True),
        ("spatial_mse", "Spatial MSE", True),
        ("rate_mse", "Reaction-rate MSE", True),
        ("best_loss", "Best loss", True),
    ]

    lines = [
        "\\documentclass[tikz,border=3pt]{standalone}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  width=15.5cm,",
        "  height=8.2cm,",
        "  ylabel={Value},",
        "  ymode=log,",
        "  ymin=1e-6,",
        "  grid=both,",
        "  legend style={at={(0.5,-0.18)},anchor=north,legend columns=2},",
        "  xtick={1,2,3,4,5},",
        "  xticklabels={Train time,Outlet MSE,Spatial MSE,Rate MSE,Best loss},",
        "  xticklabel style={rotate=15,anchor=east},",
        "  title={3-species main scores (mean $\\pm$ std over 30 runs)},",
        "]",
    ]

    for method, color, shift in [("PINN", "blue", -0.18), ("rPINN", "red", 0.18)]:
        lines.append(f"\\addplot+[only marks, mark=*, mark size=2.8pt, color={color}, error bars/.cd, y dir=both, y explicit] coordinates {{")
        for idx, (key, _title, _log) in enumerate(metrics, start=1):
            stat = pinn[key] if method == "PINN" else rpinn[key]
            mean = float(stat["mean"])
            std = float(stat["std"])
            lines.append(f"({idx + shift},{fmt_num(mean)}) +- (0,{fmt_num(std)})")
        lines.append("};")
        display = method if method == "PINN" else "MR-PINN"
        lines.append(f"\\addlegendentry{{{display}}}")

    lines.extend(["\\end{axis}", "\\end{tikzpicture}", "\\end{document}"])
    (out_dir / "main_scores.tex").write_text("\n".join(lines), encoding="utf-8")


def write_convergence_tex_from_csv(
    convergence_csv: Path,
    out_dir: Path,
    title: str | None = None,
    pinn_epochs: int = 1800,
    rpinn_epochs: int = 2250,
) -> None:
    """Write a convergence.tex into out_dir using the provided CSV file.

    The CSV path is written relative to the output directory so the generated .tex
    can be compiled from the output folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_csv = os.path.relpath(convergence_csv, out_dir)
    if title is None:
        title = "Training convergence"

    x_max = max(pinn_epochs, rpinn_epochs)
    xticks = ",".join(str(tick) for tick in range(0, x_max + 1, 250))

    tex_template = r"""
\documentclass[tikz,border=3pt]{standalone}

\usepackage{pgfplots}
\usepackage{pgfplotstable}
\usepgfplotslibrary{fillbetween}

\pgfplotsset{compat=1.18}

\begin{document}

\begin{tikzpicture}

\begin{axis}[
    width=15.5cm,
    height=8.2cm,
    title={__TITLE__},
    xlabel={Training epochs},
    ylabel={Training loss},
    ymode=log,
    xmin=0,
    xmax=__XMAX__,
    xtick={__XTICKS__},
    scaled x ticks=false,
    grid=both,
    legend pos=south west,
    legend cell align=left,
]

\pgfplotstableread[col sep=comma]
{__CSV__}
\convergence

% =========================================================
% PINN
% =========================================================

\addplot[
    name path=pinnlower,
    draw=none,
    forget plot
]
table[
    x expr=\thisrow{normalized_progress}*__PINN_EPOCHS__,
    y expr=\thisrow{pinn_mean_loss}-\thisrow{pinn_std_loss}
] {\convergence};

\addplot[
    name path=pinnupper,
    draw=none,
    forget plot
]
table[
    x expr=\thisrow{normalized_progress}*__PINN_EPOCHS__,
    y expr=\thisrow{pinn_mean_loss}+\thisrow{pinn_std_loss}
] {\convergence};

\addplot[
    blue!20,
    forget plot
]
fill between[of=pinnlower and pinnupper];

\addplot[
    blue,
    thick
]
table[
    x expr=\thisrow{normalized_progress}*__PINN_EPOCHS__,
    y=pinn_mean_loss
] {\convergence};

\addlegendentry{PINN ($\mu \pm \sigma$)}

% =========================================================
% rPINN
% =========================================================

\addplot[
    name path=rpinnlower,
    draw=none,
    forget plot
]
table[
    x expr=\thisrow{normalized_progress}*__RPINN_EPOCHS__,
    y expr=\thisrow{rpinn_mean_loss}-\thisrow{rpinn_std_loss}
] {\convergence};

\addplot[
    name path=rpinnupper,
    draw=none,
    forget plot
]
table[
    x expr=\thisrow{normalized_progress}*__RPINN_EPOCHS__,
    y expr=\thisrow{rpinn_mean_loss}+\thisrow{rpinn_std_loss}
] {\convergence};

\addplot[
    red!20,
    forget plot
]
fill between[of=rpinnlower and rpinnupper];

\addplot[
    red,
    thick
]
table[
    x expr=\thisrow{normalized_progress}*__RPINN_EPOCHS__,
    y=rpinn_mean_loss
] {\convergence};

\addlegendentry{MR-PINN ($\mu \pm \sigma$)}

\end{axis}

\end{tikzpicture}

\end{document}
"""
    tex = (
        tex_template.replace("__CSV__", rel_csv)
        .replace("__TITLE__", title)
        .replace("__PINN_EPOCHS__", str(pinn_epochs))
        .replace("__RPINN_EPOCHS__", str(rpinn_epochs))
        .replace("__XMAX__", str(x_max))
        .replace("__XTICKS__", xticks)
    )
    (out_dir / "convergence.tex").write_text(tex, encoding="utf-8")


def load_records(bench_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    csv_path = bench_dir / "benchmark_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        records.append(
            {
                "run_idx": int(row["run_idx"]),
                "seed": int(row["seed"]),
                "method": row["method"],
                "train_time_s": float(row["train_time_s"]),
                "epochs_used": int(float(row["epochs_used"])),
                "outlet_mse": float(row["outlet_mse"]),
                "spatial_mse": float(row["spatial_mse"]),
                "best_loss": float(row["best_loss"]),
                "stage_epoch_counts": [int(token) for token in row["stage_epoch_counts"].split(",") if token.strip()],
                "stage_durations": [float(token) for token in row["stage_durations"].split(",") if token.strip()],
            }
        )
    return records


def write_metric_distributions_tex(records: list[dict[str, object]], out_dir: Path) -> None:
    metric_specs = [
        ("train_time_s", "Training Time [s]", True),
        ("outlet_mse", "Outlet MSE", True),
        ("spatial_mse", "Spatial MSE", True),
    ]

    lines = [
        "\\documentclass[tikz,border=3pt]{standalone}",
        "\\usepackage{pgfplots}",
        "\\usepgfplotslibrary{groupplots}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[group style={group size=3 by 1, horizontal sep=1.35cm}, width=5.7cm, height=5.8cm]",
    ]

    for panel_idx, (key, title, log_scale) in enumerate(metric_specs, start=1):
        ylabel = "ylabel={Value}" if panel_idx == 1 else ""
        lines.append(
            f"\\nextgroupplot[title={{{title}}}, xtick={{1,2}}, xticklabels={{PINN,MR-PINN}}, grid=both, "
            + ("ymode=log, " if log_scale else "")
            + ylabel
            + "]"
        )
        for idx, (method, color, mark) in enumerate([("PINN", "blue", "*"), ("rPINN", "red", "square*")], start=1):
            values = np.asarray([float(r[key]) for r in records if r["method"] == method], dtype=float)
            if values.size == 0:
                points = ""
            else:
                x_coords = np.full_like(values, idx, dtype=float)
                jitter = (np.linspace(-0.08, 0.08, num=values.size) if values.size > 1 else np.array([0.0]))
                points = " ".join(f"({fmt_num(x)},{fmt_num(y)})" for x, y in zip(x_coords + jitter, values))
            mean = float(values.mean()) if values.size > 0 else 0.0
            std = float(values.std(ddof=1)) if values.size > 1 else 0.0
            lines.append(f"\\addplot+[only marks, mark={mark}, mark options={{fill={color}}}, color={color}, opacity=0.55] coordinates {{{points}}};")
            coord = f"({idx},{fmt_num(mean)}) +- (0,{fmt_num(std)})"
            lines.append("\\addplot+[only marks, mark=x, draw=" + color + ", mark options={scale=1.2, thick}, error bars/.cd, y dir=both, y explicit] coordinates {" + coord + "};")

    lines.extend(["\\end{groupplot}", "\\end{tikzpicture}", "\\end{document}"])
    (out_dir / "metric_distributions.tex").write_text("\n".join(lines), encoding="utf-8")


def write_speed_accuracy_tex(records: list[dict[str, object]], out_dir: Path) -> None:
    lines = [
        "\\documentclass[tikz,border=3pt]{standalone}",
        "\\usepackage{pgfplots}",
        "\\pgfplotsset{compat=1.18}",
        "\\begin{document}",
        "\\begin{tikzpicture}",
        "\\begin{axis}[width=12cm,height=8cm,xmode=log,ymode=log,xlabel={Training time [s]},ylabel={Outlet MSE},title={Speed-Accuracy Tradeoff},grid=both,legend pos=north west]",
    ]

    for method, color, mark in [("PINN", "blue", "*"), ("rPINN", "red", "square*")]:
        subset = [r for r in records if r["method"] == method]
        xs = np.asarray([float(r["train_time_s"]) for r in subset], dtype=float)
        ys = np.asarray([float(r["outlet_mse"]) for r in subset], dtype=float)
        coords = " ".join(f"({fmt_num(x)},{fmt_num(y)})" for x, y in zip(xs, ys))
        mx = float(xs.mean()) if xs.size > 0 else 0.0
        my = float(ys.mean()) if ys.size > 0 else 0.0
        sx = float(xs.std(ddof=1)) if xs.size > 1 else 0.0
        sy = float(ys.std(ddof=1)) if ys.size > 1 else 0.0
        lines.append(f"\\addplot+[only marks, mark={mark}, mark options={{fill={color}}}, color={color}] coordinates {{{coords}}};")
        display = method if method == "PINN" else "MR-PINN"
        lines.append(f"\\addlegendentry{{{display}}}")
        coord = f"({fmt_num(mx)},{fmt_num(my)}) +- ({fmt_num(sx)},{fmt_num(sy)})"
        lines.append("\\addplot+[only marks, mark=x, draw=" + color + ", mark options={scale=1.2, thick}, error bars/.cd, x dir=both, x explicit, y dir=both, y explicit] coordinates {" + coord + "};")
        lines.append(f"\\addlegendentry{{{display} mean ± std}}")

    lines.extend(["\\end{axis}", "\\end{tikzpicture}", "\\end{document}"])
    (out_dir / "speed_vs_accuracy.tex").write_text("\n".join(lines), encoding="utf-8")


def write_readme(out_dir: Path, bench_dir: Path) -> None:
    try:
        bench_dir_text = str(bench_dir.relative_to(ROOT.parent))
    except ValueError:
        bench_dir_text = os.path.relpath(bench_dir, out_dir)
    text = """# 3-species TikZ exports

Standalone TikZ/PDF figures generated from `__BENCH_DIR__`.

Files:
- `main_scores.tex`
- `convergence.tex`
- `metric_distributions.tex`
- `speed_vs_accuracy.tex`

Compile from this folder with:

```bash
pdflatex main_scores.tex
pdflatex convergence.tex
```
"""
    (out_dir / "README.md").write_text(text.replace("__BENCH_DIR__", bench_dir_text), encoding="utf-8")


def make_bw_tex(tex_text: str, kind: str) -> str:
    replacements: list[tuple[str, str]] = []
    if kind == "convergence":
        replacements = [
            ("legend pos=south west", "legend pos=north east"),
            ("blue!20", "black!15"),
            ("blue,\n    thick", "black,\n    very thick"),
            ("red!20", "black!30"),
            ("red,\n    thick", "black,\n    very thick, dashed"),
        ]
    elif kind == "metric_distributions":
        replacements = [
            ("mark=*, mark options={fill=blue}, color=blue, opacity=0.55", "mark=*, mark options={fill=black}, color=black, opacity=0.55"),
            ("mark=square*, mark options={fill=red}, color=red, opacity=0.55", "mark=square*, mark options={fill=black!45}, color=black, opacity=0.55"),
            ("draw=blue", "draw=black"),
            ("draw=red", "draw=black"),
        ]
    elif kind == "speed_vs_accuracy":
        replacements = [
            ("mark=*, mark options={fill=blue}, color=blue", "mark=*, mark options={fill=black}, color=black"),
            ("mark=square*, mark options={fill=red}, color=red", "mark=square*, mark options={fill=black!45}, color=black"),
            ("draw=blue", "draw=black"),
            ("draw=red", "draw=black"),
        ]
    elif kind == "main_scores":
        replacements = [
            ("color=blue", "color=black"),
            ("color=red", "color=black"),
        ]
    for old, new in replacements:
        tex_text = tex_text.replace(old, new)
    return tex_text


def write_bw_variants(out_dir: Path) -> None:
    for name in ["main_scores", "convergence", "metric_distributions", "speed_vs_accuracy"]:
        tex_path = out_dir / f"{name}.tex"
        tex_path.write_text(make_bw_tex(tex_path.read_text(encoding="utf-8"), name), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TikZ figures from benchmark folders.")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_BENCH_DIR), help="Benchmark folder containing benchmark_summary.json and convergence_tikz_data.csv")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder for normal TikZ files")
    parser.add_argument("--bw-output-dir", type=str, default=None, help="Optional output folder for black-and-white TikZ files")
    parser.add_argument("--pinn-epochs", type=int, default=1800, help="Epoch count used to scale the PINN convergence curve")
    parser.add_argument("--rpinn-epochs", type=int, default=2250, help="Epoch count used to scale the rPINN convergence curve")
    args = parser.parse_args()

    in_dir = resolve_cli_path(args.input_dir)
    out_dir = resolve_cli_path(args.output_dir)
    bw_out_dir = resolve_cli_path(args.bw_output_dir) if args.bw_output_dir else None

    summary_path = in_dir / "benchmark_summary.json"
    csv_path = in_dir / "convergence_tikz_data.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary JSON not found: {summary_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Convergence CSV not found: {csv_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_records(in_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_main_scores_tex(summary, out_dir)
    write_convergence_tex_from_csv(csv_path, out_dir, pinn_epochs=args.pinn_epochs, rpinn_epochs=args.rpinn_epochs)
    write_metric_distributions_tex(records, out_dir)
    write_speed_accuracy_tex(records, out_dir)
    write_readme(out_dir, in_dir)
    for tex_name in ["main_scores.tex", "convergence.tex", "metric_distributions.tex", "speed_vs_accuracy.tex"]:
        compile_tex(out_dir / tex_name)
    print(f"Wrote normal TikZ files to {out_dir}")

    if bw_out_dir is not None:
        bw_out_dir.mkdir(parents=True, exist_ok=True)
        write_main_scores_tex(summary, bw_out_dir)
        write_convergence_tex_from_csv(csv_path, bw_out_dir, pinn_epochs=args.pinn_epochs, rpinn_epochs=args.rpinn_epochs)
        write_metric_distributions_tex(records, bw_out_dir)
        write_speed_accuracy_tex(records, bw_out_dir)
        write_readme(bw_out_dir, in_dir)
        write_bw_variants(bw_out_dir)
        for tex_name in ["main_scores.tex", "convergence.tex", "metric_distributions.tex", "speed_vs_accuracy.tex"]:
            compile_tex(bw_out_dir / tex_name)
        print(f"Wrote black-and-white TikZ files to {bw_out_dir}")


if __name__ == "__main__":
    main()
