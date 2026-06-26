from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.interpolate import CubicSpline

import A3_PINN_VS_rPINN as base

INTERPOLATION_METHODS = ("nearest", "linear", "spline")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "Data" / "rpinn_benchmark_1species_interpolation_compare"


def interpolate_1d(values: np.ndarray, new_size: int, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if new_size <= 0:
        raise ValueError("new_size must be positive")
    if values.size == 0:
        return np.zeros((new_size,), dtype=np.float64)
    if new_size == 1:
        return np.array([float(values[0])], dtype=np.float64)
    if values.size == 1:
        return np.full((new_size,), float(values[0]), dtype=np.float64)

    x_old = np.linspace(0.0, 1.0, values.size, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, new_size, dtype=np.float64)

    if method == "nearest":
        nearest_idx = np.rint(x_new * (values.size - 1)).astype(int)
        nearest_idx = np.clip(nearest_idx, 0, values.size - 1)
        return values[nearest_idx]

    if method == "linear":
        return np.interp(x_new, x_old, values)

    if method == "spline":
        if values.size < 4:
            return np.interp(x_new, x_old, values)
        return CubicSpline(x_old, values, bc_type="natural")(x_new)

    raise ValueError(f"Unknown interpolation method: {method}")


def expand_output_with_method(model: base.RPinN, new_output_dim: int, method: str) -> None:
    old_weight = model.fc3.weight.data.clone()
    old_bias = model.fc3.bias.data.clone()
    old_output_dim, in_features = old_weight.shape

    new_fc3 = nn.Linear(in_features, new_output_dim)
    new_fc3 = new_fc3.to(device=old_weight.device, dtype=old_weight.dtype)

    old_weight_np = old_weight.detach().cpu().numpy()
    old_bias_np = old_bias.detach().cpu().numpy()
    new_weight_np = np.stack(
        [interpolate_1d(old_weight_np[:, feature_idx], new_output_dim, method) for feature_idx in range(in_features)],
        axis=1,
    )
    new_bias_np = interpolate_1d(old_bias_np, new_output_dim, method)

    with torch.no_grad():
        new_fc3.weight.copy_(torch.tensor(new_weight_np, dtype=old_weight.dtype, device=old_weight.device))
        new_fc3.bias.copy_(torch.tensor(new_bias_np, dtype=old_bias.dtype, device=old_bias.device))

    model.fc3 = new_fc3


def train_rpinn_once_with_method(config, tensors, device, seed, interpolation_method, capture_reference=False):
    base.set_global_seed(seed)

    cin_history, cin_current, target_profile, cout_target, dt = tensors
    total_points = target_profile.shape[1]

    schedule = config["rpinn_schedule"]
    stage_epochs = config["rpinn_epochs"]
    stage_thresholds = config["rpinn_thresholds"]

    model = base.RPinN(input_dim=cin_history.shape[1], output_dim=schedule[0]).to(device)

    if config["use_trainable_params"]:
        init_D = 1e-3
        raw_init_D = np.log(np.expm1(init_D))
        model.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32, device=device))
        model.k_param = nn.Parameter(torch.tensor(0.25, dtype=torch.float32, device=device))

    all_loss_history = []
    stage_epoch_counts = []
    stage_durations = []
    best_loss_global = float("inf")
    stage_reference = None

    base.synchronize_if_needed(device)
    start_time = time.perf_counter()

    for stage_idx, out_dim in enumerate(schedule):
        if stage_idx > 0:
            expand_output_with_method(model, out_dim, interpolation_method)
            model = model.to(device)

        anchor_indices = base.select_anchor_indices(total_points, out_dim)
        anchor_target = target_profile[:, anchor_indices]

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        best_loss = float("inf")
        stale = 0

        base.synchronize_if_needed(device)
        stage_start = time.perf_counter()

        stage_epoch_used = 0
        for epoch in range(stage_epochs[stage_idx]):
            model.train()
            optimizer.zero_grad()

            predicted_profile = model(cin_history)
            predicted_prev = model(config["cin_history_prev"])

            supervised_loss = base.compute_supervised_loss(
                loss_mode=config["loss_mode"],
                predicted_profile=predicted_profile,
                target_profile=anchor_target,
                cin_current=cin_current,
                cout_target=cout_target,
            )

            if config["use_trainable_params"]:
                D = F.softplus(model.D_param)
                k = F.softplus(model.k_param)
            else:
                D = torch.tensor(config["fixed_dispersion"], dtype=torch.float32, device=device)
                k = torch.tensor(config["fixed_reaction_rate"], dtype=torch.float32, device=device)

            residual = base.compute_residual(
                predicted_profile=predicted_profile,
                predicted_prev=predicted_prev,
                dt=dt,
                reactor_length=config["reactor_length"],
                D=D,
                k=k,
                q=1.0,
            )
            phys_loss = torch.mean(residual ** 2)

            loss = supervised_loss + config["phys_weight"] * phys_loss
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            all_loss_history.append(loss_value)

            if loss_value + 1e-12 < best_loss:
                best_loss = loss_value
                stale = 0
            else:
                stale += 1

            stage_epoch_used = epoch + 1

            if not config.get("force_fixed_epochs", False):
                if best_loss < stage_thresholds[stage_idx] and epoch >= config["min_epochs_before_stop"]:
                    break
                if stale >= config["rpinn_patience"]:
                    break

        base.synchronize_if_needed(device)
        stage_durations.append(time.perf_counter() - stage_start)
        stage_epoch_counts.append(stage_epoch_used)
        best_loss_global = min(best_loss_global, best_loss)

        if capture_reference and stage_idx == 0 and stage_reference is None:
            with torch.no_grad():
                stage_reference = {
                    "interpolation_method": interpolation_method,
                    "seed": int(seed),
                    "stage1_out_dim": int(out_dim),
                    "stage2_out_dim": int(schedule[1]) if len(schedule) > 1 else int(out_dim),
                    "anchor_indices_stage1": np.asarray(anchor_indices, dtype=np.int64),
                    "stage1_anchor_pred": model(cin_history).detach().cpu().numpy().astype(np.float32),
                    "target_profile": target_profile.detach().cpu().numpy().astype(np.float32),
                    "time_axis": (np.arange(target_profile.shape[0], dtype=np.float32) * float(dt)),
                    "reactor_length": float(config["reactor_length"]),
                    "total_points": int(total_points),
                }

    base.synchronize_if_needed(device)
    if config.get("use_trainable_params", False) and config.get("k_finetune_epochs", 0) > 0:
        k_epochs = int(config.get("k_finetune_epochs", 0))
        if hasattr(model, "k_param"):
            finetune_optimizer = optim.Adam([model.k_param], lr=config["learning_rate"])
            for _ in range(k_epochs):
                model.train()
                finetune_optimizer.zero_grad()

                predicted_profile = model(cin_history)
                predicted_prev = model(config["cin_history_prev"])
                anchor_indices_full = base.select_anchor_indices(total_points, model.fc3.out_features)
                anchor_target = target_profile[:, anchor_indices_full]

                supervised_loss = base.compute_supervised_loss(
                    loss_mode=config["loss_mode"],
                    predicted_profile=predicted_profile,
                    target_profile=anchor_target,
                    cin_current=cin_current,
                    cout_target=cout_target,
                )

                D = F.softplus(model.D_param) if config.get("use_trainable_params", False) else torch.tensor(
                    config["fixed_dispersion"], dtype=torch.float32, device=device
                )
                k = F.softplus(model.k_param)

                residual = base.compute_residual(
                    predicted_profile=predicted_profile,
                    predicted_prev=predicted_prev,
                    dt=dt,
                    reactor_length=config["reactor_length"],
                    D=D,
                    k=k,
                    q=1.0,
                )
                phys_loss = torch.mean(residual ** 2)

                loss = supervised_loss + config["phys_weight"] * phys_loss
                loss.backward()
                finetune_optimizer.step()
                all_loss_history.append(float(loss.item()))
                stage_epoch_counts[-1] = int(stage_epoch_counts[-1]) + 1

    base.synchronize_if_needed(device)
    train_time_s = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        anchor_pred = model(cin_history).detach().cpu().numpy()

    if config["use_trainable_params"]:
        reaction_rate = float(F.softplus(model.k_param).detach().cpu().item())
    else:
        reaction_rate = float(config["fixed_reaction_rate"])

    anchor_indices = base.select_anchor_indices(total_points, model.fc3.out_features)
    full_pred_profile = base.reconstruct_full_profile(anchor_pred, anchor_indices, total_points, interpolation_method=interpolation_method)

    target_np = target_profile.detach().cpu().numpy()
    cout_np = cout_target.detach().cpu().numpy()[:, 0]

    spatial_mse = float(np.mean((full_pred_profile - target_np) ** 2))
    outlet_mse = float(np.mean((full_pred_profile[:, -1] - cout_np) ** 2))
    reaction_rate_abs_error = float(abs(reaction_rate - float(config["fixed_reaction_rate"])))

    result = {
        "method": "rPINN",
        "train_time_s": train_time_s,
        "epochs_used": int(sum(stage_epoch_counts)),
        "outlet_mse": outlet_mse,
        "spatial_mse": spatial_mse,
        "best_loss": best_loss_global,
        "reaction_rate": reaction_rate,
        "reaction_rate_abs_error": reaction_rate_abs_error,
        "loss_history": np.asarray(all_loss_history, dtype=np.float64),
        "stage_epoch_counts": stage_epoch_counts,
        "stage_durations": stage_durations,
    }
    return result, stage_reference


def save_reference_npz(reference_path: Path, reference: dict[str, object]) -> None:
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(reference_path, **reference)


def save_method_report(summary: dict[str, object], out_path: Path, method: str) -> None:
    method_stats = summary["rPINN"]
    lines = [
        f"Interpolation method: {method}",
        f"Runs: {method_stats['runs']}",
        f"Train time [s]: mean={method_stats['train_time_s']['mean']:.3f}, std={method_stats['train_time_s']['std']:.3f}, min={method_stats['train_time_s']['min']:.3f}, max={method_stats['train_time_s']['max']:.3f}",
        f"Epochs used: mean={method_stats['epochs_used']['mean']:.3f}, std={method_stats['epochs_used']['std']:.3f}, min={method_stats['epochs_used']['min']:.3f}, max={method_stats['epochs_used']['max']:.3f}",
        f"Outlet MSE: mean={method_stats['outlet_mse']['mean']:.6e}, std={method_stats['outlet_mse']['std']:.6e}, min={method_stats['outlet_mse']['min']:.6e}, max={method_stats['outlet_mse']['max']:.6e}",
        f"Spatial MSE: mean={method_stats['spatial_mse']['mean']:.6e}, std={method_stats['spatial_mse']['std']:.6e}, min={method_stats['spatial_mse']['min']:.6e}, max={method_stats['spatial_mse']['max']:.6e}",
        f"Best loss: mean={method_stats['best_loss']['mean']:.6e}, std={method_stats['best_loss']['std']:.6e}, min={method_stats['best_loss']['min']:.6e}, max={method_stats['best_loss']['max']:.6e}",
        f"Reaction rate: mean={method_stats['reaction_rate']['mean']:.6g}, std={method_stats['reaction_rate']['std']:.6g}, min={method_stats['reaction_rate']['min']:.6g}, max={method_stats['reaction_rate']['max']:.6g}",
        f"Reaction-rate abs. error: mean={method_stats['reaction_rate_abs_error']['mean']:.6e}, std={method_stats['reaction_rate_abs_error']['std']:.6e}, min={method_stats['reaction_rate_abs_error']['min']:.6e}, max={method_stats['reaction_rate_abs_error']['max']:.6e}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method_benchmark(method: str, args, device, tensors, method_root: Path, reference_path: Path | None) -> dict[str, object]:
    method_root.mkdir(parents=True, exist_ok=True)

    rpinn_config = dict(args.base_config)
    rpinn_config.update(
        {
            "rpinn_schedule": args.rpinn_schedule,
            "rpinn_epochs": args.rpinn_epochs,
            "rpinn_thresholds": args.rpinn_thresholds,
            "rpinn_patience": args.rpinn_patience,
            "force_fixed_epochs": getattr(args, "force_fixed_epochs", False),
        }
    )

    run_records = []
    reference_written = False

    print(f"Running interpolation benchmark: {method} | runs={args.runs}")
    for run_idx in range(args.runs):
        seed = args.base_seed + run_idx
        print(f"Run {run_idx + 1}/{args.runs} | seed={seed}")

        rpinn_result, stage_reference = train_rpinn_once_with_method(
            rpinn_config,
            tensors,
            device,
            seed,
            interpolation_method=method,
            capture_reference=(reference_path is not None and not reference_written),
        )

        run_records.append(
            {
                "run_idx": run_idx,
                "seed": seed,
                "method": rpinn_result["method"],
                "train_time_s": rpinn_result["train_time_s"],
                "epochs_used": rpinn_result["epochs_used"],
                "outlet_mse": rpinn_result["outlet_mse"],
                "spatial_mse": rpinn_result["spatial_mse"],
                "best_loss": rpinn_result["best_loss"],
                "reaction_rate": rpinn_result["reaction_rate"],
                "reaction_rate_abs_error": rpinn_result["reaction_rate_abs_error"],
                "loss_history": rpinn_result["loss_history"],
                "stage_epoch_counts": rpinn_result.get("stage_epoch_counts", []),
                "stage_durations": rpinn_result.get("stage_durations", []),
            }
        )
        np.save(method_root / f"loss_history_run{run_idx}.npy", np.asarray(rpinn_result["loss_history"], dtype=np.float64))

        if reference_path is not None and not reference_written and stage_reference is not None:
            save_reference_npz(reference_path, stage_reference)
            reference_written = True

        print(
            "  rPINN: "
            f"time={rpinn_result['train_time_s']:.2f}s, "
            f"outlet_mse={rpinn_result['outlet_mse']:.3e}, "
            f"spatial_mse={rpinn_result['spatial_mse']:.3e}, "
            f"k={rpinn_result['reaction_rate']:.6g}"
        )

    summary = {
        "rPINN": base.summarize_method(run_records, "rPINN"),
        "config": {
            "runs": args.runs,
            "base_seed": args.base_seed,
            "device": str(device),
            "num_delays": args.num_delays,
            "reactor_length": args.reactor_length,
            "learning_rate": args.learning_rate,
            "phys_weight": args.phys_weight,
            "loss_mode": args.loss_mode,
            "pinn_epochs": args.pinn_epochs,
            "pinn_threshold": args.pinn_threshold,
            "pinn_patience": args.pinn_patience,
            "rpinn_schedule": args.rpinn_schedule,
            "rpinn_epochs": args.rpinn_epochs,
            "rpinn_thresholds": args.rpinn_thresholds,
            "rpinn_patience": args.rpinn_patience,
            "min_epochs_before_stop": args.min_epochs_before_stop,
            "use_trainable_params": not args.no_trainable_params,
            "fixed_dispersion": args.fixed_dispersion,
            "fixed_reaction_rate": args.fixed_reaction_rate,
            "data_path": args.data_path,
            "interpolation_method": method,
        },
    }

    csv_path = method_root / "benchmark_results.csv"
    base.save_csv(run_records, str(csv_path))

    summary_path = method_root / "benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = method_root / "method_report.txt"
    save_method_report(summary, report_path, method)

    return {
        "method": method,
        "method_dir": str(method_root),
        "summary_path": str(summary_path),
        "summary": summary,
        "reference_path": str(reference_path) if reference_path is not None else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark rPINN with alternate stage interpolation methods.")
    parser.add_argument("--data-path", type=str, default="./Data/flowReactorData.pkl")
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-delays", type=int, default=30)
    parser.add_argument("--reactor-length", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phys-weight", type=float, default=0.1)
    parser.add_argument(
        "--loss-mode",
        type=str,
        default="strict_boundary_phys",
        choices=["strict_boundary_phys", "profile_phys"],
        help="Shared supervised term for both methods.",
    )
    parser.add_argument("--pinn-epochs", type=int, default=2900)
    parser.add_argument("--pinn-threshold", type=float, default=5e-5)
    parser.add_argument("--pinn-patience", type=int, default=120)
    parser.add_argument("--rpinn-schedule", type=str, default="4,20,70,100")
    parser.add_argument("--rpinn-epochs", type=str, default="300,500,900,800")
    parser.add_argument("--rpinn-thresholds", type=str, default="5e-4,2e-4,1e-4,2e-5")
    parser.add_argument("--rpinn-patience", type=int, default=60)
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--force-fixed-epochs", action="store_true", help="Force each method to run its full configured epoch budget")
    parser.add_argument("--fixed-dispersion", type=float, default=1e-3)
    parser.add_argument("--fixed-reaction-rate", type=float, default=0.25)
    parser.add_argument("--no-trainable-params", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--k-finetune-epochs", type=int, default=0)
    parser.add_argument("--methods", type=str, default=",".join(INTERPOLATION_METHODS), help="Comma-separated interpolation methods to compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    method_names = [token.strip().lower() for token in args.methods.split(",") if token.strip()]
    if not method_names:
        raise ValueError("At least one interpolation method must be specified")
    for method_name in method_names:
        if method_name not in INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported interpolation method: {method_name}")

    rpinn_schedule = base.parse_int_list(args.rpinn_schedule)
    rpinn_epochs = base.parse_int_list(args.rpinn_epochs)
    rpinn_thresholds = base.parse_float_list(args.rpinn_thresholds)

    if not (len(rpinn_schedule) == len(rpinn_epochs) == len(rpinn_thresholds)):
        raise ValueError("rpinn-schedule, rpinn-epochs, and rpinn-thresholds must have equal lengths")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(args.device)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.data_path, "rb") as handle:
        data = pickle.load(handle)

    cin_history, cin_current, target_profile, cout_target, dt, cin_history_prev = base.build_tensors_for_training(
        data=data,
        num_delays=args.num_delays,
        device=device,
    )
    tensors = (cin_history, cin_current, target_profile, cout_target, dt)

    common_base_config = {
        "reactor_length": args.reactor_length,
        "learning_rate": args.learning_rate,
        "phys_weight": args.phys_weight,
        "loss_mode": args.loss_mode,
        "use_trainable_params": not args.no_trainable_params,
        "k_finetune_epochs": args.k_finetune_epochs,
        "fixed_dispersion": args.fixed_dispersion,
        "fixed_reaction_rate": args.fixed_reaction_rate,
        "min_epochs_before_stop": args.min_epochs_before_stop,
        "cin_history_prev": cin_history_prev,
    }
    args.base_config = common_base_config
    args.rpinn_schedule = rpinn_schedule
    args.rpinn_epochs = rpinn_epochs
    args.rpinn_thresholds = rpinn_thresholds

    comparison_summary: dict[str, object] = {
        "methods": {},
        "config": {
            "runs": args.runs,
            "base_seed": args.base_seed,
            "device": str(device),
            "num_delays": args.num_delays,
            "reactor_length": args.reactor_length,
            "learning_rate": args.learning_rate,
            "phys_weight": args.phys_weight,
            "loss_mode": args.loss_mode,
            "pinn_epochs": args.pinn_epochs,
            "pinn_threshold": args.pinn_threshold,
            "pinn_patience": args.pinn_patience,
            "rpinn_schedule": rpinn_schedule,
            "rpinn_epochs": rpinn_epochs,
            "rpinn_thresholds": rpinn_thresholds,
            "rpinn_patience": args.rpinn_patience,
            "min_epochs_before_stop": args.min_epochs_before_stop,
            "use_trainable_params": not args.no_trainable_params,
            "fixed_dispersion": args.fixed_dispersion,
            "fixed_reaction_rate": args.fixed_reaction_rate,
            "data_path": args.data_path,
            "methods": method_names,
        },
    }

    reference_path = output_root / "stage_transition_reference.npz"
    for index, method in enumerate(method_names):
        method_dir = output_root / method
        method_result = run_method_benchmark(
            method=method,
            args=args,
            device=device,
            tensors=tensors,
            method_root=method_dir,
            reference_path=reference_path if index == 0 else None,
        )
        comparison_summary["methods"][method] = {
            "method_dir": method_result["method_dir"],
            "summary_path": method_result["summary_path"],
            "reference_path": method_result["reference_path"],
            "summary": method_result["summary"],
        }

    comparison_csv = output_root / "comparison_summary.csv"
    with comparison_csv.open("w", encoding="utf-8") as handle:
        handle.write("method,train_time_mean,train_time_std,outlet_mse_mean,spatial_mse_mean,best_loss_mean,reaction_rate_abs_error_mean\n")
        for method in method_names:
            method_summary = comparison_summary["methods"][method]["summary"]["rPINN"]
            handle.write(
                f"{method},{method_summary['train_time_s']['mean']:.6g},{method_summary['train_time_s']['std']:.6g},"
                f"{method_summary['outlet_mse']['mean']:.6g},{method_summary['spatial_mse']['mean']:.6g},"
                f"{method_summary['best_loss']['mean']:.6g},{method_summary['reaction_rate_abs_error']['mean']:.6g}\n"
            )

    comparison_path = output_root / "comparison_summary.json"
    comparison_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")

    print("\nComparison summary saved to:")
    print(f"  - {comparison_path}")
    print(f"  - {comparison_csv}")
    if reference_path.exists():
        print(f"  - {reference_path}")
    for method in method_names:
        print(f"  - {output_root / method}")


if __name__ == "__main__":
    main()
