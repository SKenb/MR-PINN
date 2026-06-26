from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import B3_PINN_VS_rPINN_3Species as benchmark


def save_reaction_rate_plot(result, time_axis, out_path):
    true_rate = result["true_rate"].mean(axis=2)
    pred_rate = result["pred_rate"].mean(axis=2)

    species_labels = ["A", "B", "C"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(time_axis, true_rate[:, idx], label="Real", color="#2ca02c")
        ax.plot(time_axis, pred_rate[:, idx], label="PINN predicted", color="#1f77b4", linestyle="--")
        ax.set_ylabel(f"r_{species_labels[idx]}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("PINN Predicted vs Real Reaction Rates (spatial mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PINN on the 3-species flow reactor data.")
    parser.add_argument("--data-path", type=str, default="./Data/flowReactorData_3species.pkl")
    parser.add_argument("--output-dir", type=str, default="./Data/pinn_3species")
    parser.add_argument("--num-delays", type=int, default=30)
    parser.add_argument("--reactor-length", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phys-weight", type=float, default=0.1)
    parser.add_argument("--loss-mode", type=str, default="strict_boundary_phys", choices=["strict_boundary_phys", "profile_phys"])
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--threshold", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--fixed-dispersion", type=float, default=1e-3)
    parser.add_argument("--fixed-reaction-rate", type=float, default=0.05)
    parser.add_argument("--no-trainable-params", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    true_k = float(benchmark.reaction_rate_constant(float(data.get("temperature", 150.0)), float(data.get("kRef", 0.08)), float(data.get("eaKJPerMol", 35.0)) * 1000.0))

    cin_history, cin_current, target_profile, target_profile_3d, cout_target, dt, cin_history_prev = benchmark.build_tensors_for_training(data=data, num_delays=args.num_delays, device=device)

    config = {
        "reactor_length": args.reactor_length,
        "learning_rate": args.learning_rate,
        "phys_weight": args.phys_weight,
        "loss_mode": args.loss_mode,
        "use_trainable_params": not args.no_trainable_params,
        "fixed_dispersion": args.fixed_dispersion,
        "fixed_reaction_rate": args.fixed_reaction_rate,
        "min_epochs_before_stop": args.min_epochs_before_stop,
        "pinn_epochs": args.epochs,
        "pinn_threshold": args.threshold,
        "pinn_patience": args.patience,
        "cin_history_prev": cin_history_prev,
        "true_k": true_k,
    }
    tensors = (cin_history, cin_current, target_profile, target_profile_3d, cout_target, dt)

    result = benchmark.train_pinn_once(config, tensors, device, seed=0)

    summary = {
        "method": result["method"],
        "train_time_s": result["train_time_s"],
        "epochs_used": result["epochs_used"],
        "outlet_mse": result["outlet_mse"],
        "spatial_mse": result["spatial_mse"],
        "rate_mse": result["rate_mse"],
        "best_loss": result["best_loss"],
        "config": {
            "data_path": args.data_path,
            "num_delays": args.num_delays,
            "reactor_length": args.reactor_length,
            "learning_rate": args.learning_rate,
            "phys_weight": args.phys_weight,
            "loss_mode": args.loss_mode,
            "epochs": args.epochs,
            "threshold": args.threshold,
            "patience": args.patience,
            "min_epochs_before_stop": args.min_epochs_before_stop,
            "fixed_dispersion": args.fixed_dispersion,
            "fixed_reaction_rate": args.fixed_reaction_rate,
            "use_trainable_params": not args.no_trainable_params,
            "device": str(device),
            "true_k": true_k,
        },
    }

    summary_path = os.path.join(args.output_dir, "pinn_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    result_path = os.path.join(args.output_dir, "pinn_result.pkl")
    with open(result_path, "wb") as f:
        pickle.dump(result, f)

    rate_plot_path = os.path.join(args.output_dir, "reaction_rate_comparison.png")
    save_reaction_rate_plot(result, np.asarray(data["timeOutVec"], dtype=np.float64), rate_plot_path)

    rate_csv_path = os.path.join(args.output_dir, "reaction_rate_comparison.csv")
    true_rate = result["true_rate"].mean(axis=2)
    pred_rate = result["pred_rate"].mean(axis=2)
    with open(rate_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow(["time_s", "true_rate_A", "true_rate_B", "true_rate_C", "pred_rate_A", "pred_rate_B", "pred_rate_C"])
        for i, t in enumerate(data["timeOutVec"]):
            writer.writerow([float(t), float(true_rate[i, 0]), float(true_rate[i, 1]), float(true_rate[i, 2]), float(pred_rate[i, 0]), float(pred_rate[i, 1]), float(pred_rate[i, 2])])

    print(f"Saved summary to {summary_path}")
    print(f"Saved result pickle to {result_path}")
    print(f"Saved reaction-rate plot to {rate_plot_path}")
    print(f"Saved reaction-rate CSV to {rate_csv_path}")


if __name__ == "__main__":
    main()