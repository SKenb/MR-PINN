from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


R_GAS = 8.31446261815324


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FullPINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        return self.fc3(out)


class RPinN(nn.Module):
    def __init__(self, input_dim: int = 11, output_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        return self.fc3(out)

    def expand_output(self, new_output_dim: int) -> None:
        old_weight = self.fc3.weight.data.clone()
        old_bias = self.fc3.bias.data.clone()
        old_output_dim, in_features = old_weight.shape

        new_fc3 = nn.Linear(in_features, new_output_dim)
        with torch.no_grad():
            for new_idx in range(new_output_dim):
                if new_output_dim == 1:
                    pos = 0.0
                else:
                    pos = new_idx * (old_output_dim - 1) / (new_output_dim - 1)
                left_idx = int(np.floor(pos))
                right_idx = int(np.ceil(pos))
                if left_idx == right_idx:
                    new_fc3.weight[new_idx].copy_(old_weight[left_idx])
                    new_fc3.bias[new_idx].copy_(old_bias[left_idx])
                else:
                    alpha = float(pos - left_idx)
                    new_fc3.weight[new_idx].copy_((1.0 - alpha) * old_weight[left_idx] + alpha * old_weight[right_idx])
                    new_fc3.bias[new_idx].copy_((1.0 - alpha) * old_bias[left_idx] + alpha * old_bias[right_idx])

        self.fc3 = new_fc3


def build_lagged_cin_features(cin: np.ndarray, num_delays: int = 10) -> torch.Tensor:
    cin = np.asarray(cin, dtype=np.float32)
    n_species, n_samples = cin.shape
    features = np.zeros((n_samples, n_species * (num_delays + 1)), dtype=np.float32)

    for sample_idx in range(n_samples):
        for species_idx in range(n_species):
            for delay_idx in range(num_delays + 1):
                source_idx = sample_idx - delay_idx
                if source_idx >= 0:
                    features[sample_idx, species_idx * (num_delays + 1) + delay_idx] = cin[species_idx, source_idx]

    return torch.tensor(features, dtype=torch.float32)


def build_training_tensors(data, num_delays: int = 10):
    conc_vec_in = data["Cin"]
    c_spatial = data["Cspatial"]

    n_species, _, n_time = c_spatial.shape
    n_space = c_spatial.shape[1] - 1

    cin_current = torch.tensor(conc_vec_in.T, dtype=torch.float32)  # [time, species]
    cin_history = build_lagged_cin_features(conc_vec_in, num_delays=num_delays)
    target_profile_3d = torch.tensor(c_spatial[:, 1:, :].transpose(2, 0, 1), dtype=torch.float32)  # [time, species, space]
    target_profile = target_profile_3d.reshape(n_time, n_species * n_space)

    return cin_history, cin_current, target_profile, target_profile_3d


def select_anchor_indices(total_points: int, out_dim: int) -> np.ndarray:
    if out_dim <= 1:
        return np.array([total_points - 1], dtype=int)
    return np.unique(np.round(np.linspace(0, total_points - 1, out_dim)).astype(int))


def reconstruct_full_profile(
    anchor_profile: np.ndarray, anchor_indices: np.ndarray, total_points: int, n_species: int, interpolation_method: str = "linear"
) -> np.ndarray:
    anchor_profile = np.asarray(anchor_profile, dtype=np.float32)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    n_samples = anchor_profile.shape[0]
    out_dim = anchor_indices.size
    reconstructed = np.zeros((n_samples, n_species, total_points), dtype=np.float32)
    full_x = np.arange(total_points, dtype=np.float32)
    for species_idx in range(n_species):
        start = species_idx * out_dim
        end = (species_idx + 1) * out_dim
        for row_idx in range(n_samples):
            row_vals = anchor_profile[row_idx, start:end]
            if interpolation_method == "nearest":
                idx = np.argmin(np.abs(anchor_indices.reshape(-1, 1) - full_x.reshape(1, -1)), axis=0)
                reconstructed[row_idx, species_idx, :] = row_vals[idx]
            elif interpolation_method == "linear":
                reconstructed[row_idx, species_idx, :] = np.interp(full_x, anchor_indices.astype(np.float32), row_vals)
            elif interpolation_method == "spline":
                try:
                    from scipy.interpolate import CubicSpline

                    if anchor_indices.size < 4 or row_vals.size < 4:
                        reconstructed[row_idx, species_idx, :] = np.interp(full_x, anchor_indices.astype(np.float32), row_vals)
                    else:
                        cs = CubicSpline(anchor_indices.astype(np.float64), row_vals.astype(np.float64), bc_type="natural")
                        reconstructed[row_idx, species_idx, :] = cs(full_x).astype(np.float32)
                except Exception:
                    reconstructed[row_idx, species_idx, :] = np.interp(full_x, anchor_indices.astype(np.float32), row_vals)
            else:
                raise ValueError(f"Unknown interpolation method: {interpolation_method}")
    return reconstructed.reshape(n_samples, n_species * total_points)


def flatten_profile(profile_3d: torch.Tensor) -> torch.Tensor:
    return profile_3d.reshape(profile_3d.shape[0], -1)


def reaction_rate_constant(temperature_c: float, k_ref: float, ea_j_per_mol: float, t_ref_c: float = 150.0) -> float:
    temperature_k = temperature_c + 273.15
    t_ref_k = t_ref_c + 273.15
    return k_ref * math.exp((ea_j_per_mol / R_GAS) * ((1.0 / t_ref_k) - (1.0 / temperature_k)))


def split_profile(profile_flat: torch.Tensor, n_species: int, n_space: int) -> torch.Tensor:
    return profile_flat.reshape(profile_flat.shape[0], n_species, n_space)


def compute_reaction_rate(profile_3d: torch.Tensor, k_value: torch.Tensor | float) -> torch.Tensor:
    a = torch.clamp(profile_3d[:, 0, :], min=0.0)
    b = torch.clamp(profile_3d[:, 1, :], min=0.0)
    rate = torch.as_tensor(k_value, dtype=profile_3d.dtype, device=profile_3d.device) * a * b
    return torch.stack((-rate, -rate, rate), dim=1)


def compute_residual(predicted_profile: torch.Tensor, predicted_prev: torch.Tensor, dt: float, reactor_length: float, D, k, q: float = 1.0) -> torch.Tensor:
    dCdt = (predicted_profile - predicted_prev) / dt
    n_species = predicted_profile.shape[1]
    nx = predicted_profile.shape[2]
    dx = reactor_length / (nx - 1) if nx > 1 else reactor_length

    dCdx = torch.zeros_like(predicted_profile)
    d2Cdx2 = torch.zeros_like(predicted_profile)
    if nx > 2:
        dCdx[:, :, 1:-1] = (predicted_profile[:, :, 2:] - predicted_profile[:, :, :-2]) / (2.0 * dx)
        d2Cdx2[:, :, 1:-1] = (predicted_profile[:, :, 2:] - 2.0 * predicted_profile[:, :, 1:-1] + predicted_profile[:, :, :-2]) / (dx ** 2)
    if nx > 1:
        dCdx[:, :, 0] = (predicted_profile[:, :, 1] - predicted_profile[:, :, 0]) / dx
        dCdx[:, :, -1] = (predicted_profile[:, :, -1] - predicted_profile[:, :, -2]) / dx
    if nx > 2:
        d2Cdx2[:, :, 0] = (predicted_profile[:, :, 2] - 2.0 * predicted_profile[:, :, 1] + predicted_profile[:, :, 0]) / (dx ** 2)
        d2Cdx2[:, :, -1] = (predicted_profile[:, :, -1] - 2.0 * predicted_profile[:, :, -2] + predicted_profile[:, :, -3]) / (dx ** 2)

    reaction = compute_reaction_rate(predicted_profile, k)
    if n_species != 3:
        raise ValueError("This benchmark expects exactly 3 species.")

    residual = dCdt - (D * d2Cdx2 - q * dCdx + reaction)
    return residual


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def compute_supervised_loss(loss_mode: str, predicted_profile: torch.Tensor, target_profile: torch.Tensor, cin_current: torch.Tensor, cout_target: torch.Tensor) -> torch.Tensor:
    if loss_mode == "strict_boundary_phys":
        return torch.mean((predicted_profile[:, :, 0] - cin_current) ** 2) + torch.mean((predicted_profile[:, :, -1] - cout_target) ** 2)
    if loss_mode == "profile_phys":
        return torch.mean((predicted_profile - target_profile) ** 2)
    raise ValueError(f"Unknown loss_mode: {loss_mode}")


def train_pinn_once(config, tensors, device: torch.device, seed: int):
    set_global_seed(seed)
    cin_history, cin_current, target_profile, target_profile_3d, cout_target, dt = tensors
    n_species = target_profile_3d.shape[1]
    n_space = target_profile_3d.shape[2]

    model = FullPINN(input_dim=cin_history.shape[1], output_dim=n_species * n_space).to(device)

    if config["use_trainable_params"]:
        init_D = 1e-3
        raw_init_D = np.log(np.expm1(init_D))
        model.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32, device=device))
        model.k_param = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, device=device))

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    loss_history = []
    best_loss = float("inf")
    stale = 0

    synchronize_if_needed(device)
    start_time = time.perf_counter()

    epoch_used = 0
    for epoch in range(config["pinn_epochs"]):
        model.train()
        optimizer.zero_grad()

        predicted_profile = split_profile(model(cin_history), n_species, n_space)
        predicted_prev = split_profile(model(config["cin_history_prev"]), n_species, n_space)

        supervised_loss = compute_supervised_loss(
            loss_mode=config["loss_mode"],
            predicted_profile=predicted_profile,
            target_profile=target_profile_3d,
            cin_current=cin_current,
            cout_target=cout_target,
        )

        if config["use_trainable_params"]:
            D = F.softplus(model.D_param)
            k = F.softplus(model.k_param)
        else:
            D = torch.tensor(config["fixed_dispersion"], dtype=torch.float32, device=device)
            k = torch.tensor(config["fixed_reaction_rate"], dtype=torch.float32, device=device)

        residual = compute_residual(predicted_profile, predicted_prev, dt, config["reactor_length"], D, k, q=1.0)
        phys_loss = torch.mean(residual ** 2)
        loss = supervised_loss + config["phys_weight"] * phys_loss

        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        if loss_value + 1e-12 < best_loss:
            best_loss = loss_value
            stale = 0
        else:
            stale += 1

        epoch_used = epoch + 1
        if not config.get("force_fixed_epochs", False):
            if best_loss < config["pinn_threshold"] and epoch >= config["min_epochs_before_stop"]:
                break
            if stale >= config["pinn_patience"]:
                break

    synchronize_if_needed(device)
    train_time_s = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        final_profile = split_profile(model(cin_history), n_species, n_space)

    spatial_mse = float(torch.mean((final_profile - target_profile_3d) ** 2).item())
    outlet_mse = float(torch.mean((final_profile[:, :, -1] - cout_target) ** 2).item())

    if config["use_trainable_params"]:
        D_learn = float(F.softplus(model.D_param).detach().cpu().numpy())
        k_learn = float(F.softplus(model.k_param).detach().cpu().numpy())
    else:
        D_learn = float(config["fixed_dispersion"])
        k_learn = float(config["fixed_reaction_rate"])

    pred_rate = compute_reaction_rate(final_profile, k_learn)
    true_rate = compute_reaction_rate(target_profile_3d, config["true_k"])
    rate_mse = float(torch.mean((pred_rate - true_rate) ** 2).item())

    return {
        "method": "PINN",
        "train_time_s": train_time_s,
        "epochs_used": epoch_used,
        "outlet_mse": outlet_mse,
        "spatial_mse": spatial_mse,
        "rate_mse": rate_mse,
        "best_loss": best_loss,
        "loss_history": np.asarray(loss_history, dtype=np.float64),
        "final_profile": final_profile.detach().cpu().numpy(),
        "pred_rate": pred_rate.detach().cpu().numpy(),
        "true_rate": true_rate.detach().cpu().numpy(),
        "k_learn": k_learn,
        "D_learn": D_learn,
    }


def train_rpinn_once(config, tensors, device: torch.device, seed: int):
    set_global_seed(seed)
    cin_history, cin_current, target_profile, target_profile_3d, cout_target, dt = tensors
    n_species = target_profile_3d.shape[1]
    n_space = target_profile_3d.shape[2]

    schedule = config["rpinn_schedule"]
    stage_epochs = config["rpinn_epochs"]
    stage_thresholds = config["rpinn_thresholds"]

    model = RPinN(input_dim=cin_history.shape[1], output_dim=n_species * schedule[0]).to(device)

    if config["use_trainable_params"]:
        init_D = 1e-3
        raw_init_D = np.log(np.expm1(init_D))
        model.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32, device=device))
        model.k_param = nn.Parameter(torch.tensor(0.05, dtype=torch.float32, device=device))

    all_loss_history = []
    stage_epoch_counts = []
    stage_durations = []
    best_loss_global = float("inf")

    synchronize_if_needed(device)
    start_time = time.perf_counter()

    for stage_idx, out_dim in enumerate(schedule):
        if stage_idx > 0:
            model.expand_output(n_species * out_dim)
            model = model.to(device)

        anchor_indices = select_anchor_indices(n_space, out_dim)
        anchor_target = target_profile_3d[:, :, anchor_indices].reshape(target_profile_3d.shape[0], -1)

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        best_loss = float("inf")
        stale = 0

        synchronize_if_needed(device)
        stage_start = time.perf_counter()

        stage_epoch_used = 0
        for epoch in range(stage_epochs[stage_idx]):
            model.train()
            optimizer.zero_grad()

            predicted_profile = split_profile(model(cin_history), n_species, out_dim)
            predicted_prev = split_profile(model(config["cin_history_prev"]), n_species, out_dim)

            supervised_loss = compute_supervised_loss(
                loss_mode=config["loss_mode"],
                predicted_profile=predicted_profile,
                target_profile=target_profile_3d[:, :, anchor_indices],
                cin_current=cin_current,
                cout_target=cout_target[:, :,],
            )

            if config["use_trainable_params"]:
                D = F.softplus(model.D_param)
                k = F.softplus(model.k_param)
            else:
                D = torch.tensor(config["fixed_dispersion"], dtype=torch.float32, device=device)
                k = torch.tensor(config["fixed_reaction_rate"], dtype=torch.float32, device=device)

            residual = compute_residual(predicted_profile, predicted_prev, dt, config["reactor_length"], D, k, q=1.0)
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

        synchronize_if_needed(device)
        stage_durations.append(time.perf_counter() - stage_start)
        stage_epoch_counts.append(stage_epoch_used)
        best_loss_global = min(best_loss_global, best_loss)

    synchronize_if_needed(device)
    train_time_s = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        anchor_pred = split_profile(model(cin_history), n_species, schedule[-1]).detach().cpu().numpy()

    full_pred_profile = reconstruct_full_profile(anchor_pred.reshape(anchor_pred.shape[0], -1), select_anchor_indices(n_space, schedule[-1]), n_space, n_species)
    full_pred_profile_3d = full_pred_profile.reshape(target_profile_3d.shape[0], n_species, n_space)
    target_np = target_profile_3d.detach().cpu().numpy()
    cout_np = cout_target.detach().cpu().numpy()

    spatial_mse = float(np.mean((full_pred_profile_3d - target_np) ** 2))
    outlet_mse = float(np.mean((full_pred_profile_3d[:, :, -1] - cout_np) ** 2))

    if config["use_trainable_params"]:
        k_learn = float(F.softplus(model.k_param).detach().cpu().numpy())
        D_learn = float(F.softplus(model.D_param).detach().cpu().numpy())
    else:
        k_learn = float(config["fixed_reaction_rate"])
        D_learn = float(config["fixed_dispersion"])

    pred_rate = compute_reaction_rate(torch.tensor(full_pred_profile_3d, dtype=torch.float32, device=device), k_learn).detach().cpu().numpy()
    true_rate = compute_reaction_rate(target_profile_3d, config["true_k"]).detach().cpu().numpy()
    rate_mse = float(np.mean((pred_rate - true_rate) ** 2))

    return {
        "method": "rPINN",
        "train_time_s": train_time_s,
        "epochs_used": int(sum(stage_epoch_counts)),
        "outlet_mse": outlet_mse,
        "spatial_mse": spatial_mse,
        "rate_mse": rate_mse,
        "best_loss": best_loss_global,
        "loss_history": np.asarray(all_loss_history, dtype=np.float64),
        "stage_epoch_counts": stage_epoch_counts,
        "stage_durations": stage_durations,
        "final_profile": full_pred_profile_3d,
        "pred_rate": pred_rate,
        "true_rate": true_rate,
        "k_learn": k_learn,
        "D_learn": D_learn,
    }


def resample_curve(y, n_points: int = 300):
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.full((n_points,), np.nan, dtype=np.float64)
    if y.size == 1:
        return np.full((n_points,), y[0], dtype=np.float64)
    old_x = np.linspace(0.0, 1.0, y.size)
    new_x = np.linspace(0.0, 1.0, n_points)
    return np.interp(new_x, old_x, y)


def summarize_method(records, method: str):
    subset = [r for r in records if r["method"] == method]

    def agg(key):
        vals = np.asarray([r[key] for r in subset], dtype=np.float64)
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return {
        "runs": len(subset),
        "train_time_s": agg("train_time_s"),
        "epochs_used": agg("epochs_used"),
        "outlet_mse": agg("outlet_mse"),
        "spatial_mse": agg("spatial_mse"),
        "rate_mse": agg("rate_mse"),
        "best_loss": agg("best_loss"),
    }


def _normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _welch_p_value_two_sided(t_stat, df):
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(2.0 * scipy_stats.t.sf(abs(t_stat), df)), "scipy_t"
    except Exception:
        return float(2.0 * _normal_sf(abs(t_stat))), "normal_approx"


def welch_ttest_and_effect_size(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_x = x.size
    n_y = y.size
    if n_x < 2 or n_y < 2:
        return {"n_pinn": int(n_x), "n_rpinn": int(n_y), "mean_pinn": float(np.mean(x)) if n_x else float("nan"), "mean_rpinn": float(np.mean(y)) if n_y else float("nan"), "mean_diff_pinn_minus_rpinn": float(np.mean(x) - np.mean(y)) if n_x and n_y else float("nan"), "relative_change_rpinn_vs_pinn_percent": float("nan"), "t_stat": float("nan"), "df": float("nan"), "p_value": float("nan"), "p_method": "insufficient_samples", "cohens_d": float("nan"), "hedges_g": float("nan"), "ci95_diff": [float("nan"), float("nan")],}

    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    a = var_x / n_x
    b = var_y / n_y
    se = math.sqrt(a + b)
    if se == 0.0:
        t_stat = 0.0
        df = float("inf")
    else:
        t_stat = (mean_x - mean_y) / se
        denom = (a * a) / (n_x - 1) + (b * b) / (n_y - 1)
        df = ((a + b) ** 2) / denom if denom > 0 else float("inf")
    p_value, p_method = _welch_p_value_two_sided(t_stat, df)
    pooled_denom = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
    pooled_std = math.sqrt(pooled_denom) if pooled_denom > 0 else 0.0
    cohens_d = (mean_x - mean_y) / pooled_std if pooled_std > 0 else 0.0
    correction = 1.0 - (3.0 / (4.0 * (n_x + n_y) - 9.0)) if (n_x + n_y) > 2 else 1.0
    hedges_g = cohens_d * correction
    try:
        from scipy import stats as scipy_stats  # type: ignore

        t_crit = float(scipy_stats.t.ppf(0.975, df)) if np.isfinite(df) else 1.96
    except Exception:
        t_crit = 1.96
    diff = mean_x - mean_y
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se
    rel = ((mean_y - mean_x) / mean_x * 100.0) if mean_x != 0 else float("nan")
    return {"n_pinn": int(n_x), "n_rpinn": int(n_y), "mean_pinn": mean_x, "mean_rpinn": mean_y, "mean_diff_pinn_minus_rpinn": diff, "relative_change_rpinn_vs_pinn_percent": rel, "t_stat": float(t_stat), "df": float(df), "p_value": float(p_value), "p_method": p_method, "cohens_d": float(cohens_d), "hedges_g": float(hedges_g), "ci95_diff": [float(ci_low), float(ci_high)]}


def build_stat_tests(records):
    metrics = ["train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "rate_mse", "best_loss"]
    results = {}
    for metric in metrics:
        x = [r[metric] for r in records if r["method"] == "PINN"]
        y = [r[metric] for r in records if r["method"] == "rPINN"]
        results[metric] = welch_ttest_and_effect_size(x, y)
    return results


def save_stats_text_report(stat_tests, out_path):
    lines = ["PINN vs rPINN Statistical Report", "Columns: mean(PINN), mean(rPINN), diff=PINN-rPINN, p-value, Hedges g", ""]
    for metric in ["train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "rate_mse", "best_loss"]:
        s = stat_tests[metric]
        lines.append(
            f"{metric}: mean_pinn={s['mean_pinn']:.6g}, mean_rpinn={s['mean_rpinn']:.6g}, diff={s['mean_diff_pinn_minus_rpinn']:.6g}, p={s['p_value']:.6g}, g={s['hedges_g']:.6g}, ci95=[{s['ci95_diff'][0]:.6g}, {s['ci95_diff'][1]:.6g}] ({s['p_method']})"
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_statistical_report(stat_tests, out_path):
    metrics = ["train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "rate_mse", "best_loss"]
    pretty = {"train_time_s": "Train time", "epochs_used": "Epochs used", "outlet_mse": "Outlet MSE", "spatial_mse": "Spatial MSE", "rate_mse": "Rate MSE", "best_loss": "Best loss"}
    g_vals = np.array([stat_tests[m]["hedges_g"] for m in metrics], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.1, 1.4]})
    ax0 = axes[0]
    y_pos = np.arange(len(metrics))
    colors = ["#2ca02c" if g < 0 else "#d62728" for g in g_vals]
    ax0.barh(y_pos, g_vals, color=colors, alpha=0.85)
    ax0.axvline(0.0, color="black", linewidth=1)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([pretty[m] for m in metrics])
    ax0.invert_yaxis()
    ax0.set_xlabel("Hedges g (PINN - rPINN)")
    ax0.set_title("Effect Size")
    ax0.grid(axis="x", alpha=0.3)
    ax1 = axes[1]
    ax1.axis("off")
    lines = ["Welch t-test (two-sided)", ""]
    for m in metrics:
        s = stat_tests[m]
        sig = "*" if np.isfinite(s["p_value"]) and s["p_value"] < 0.05 else "ns"
        lines.append(f"{pretty[m]}: p={s['p_value']:.3g} ({sig}), t={s['t_stat']:.3g}, df={s['df']:.1f}, diff={s['mean_diff_pinn_minus_rpinn']:.3g}")
    lines.append("")
    lines.append("Interpretation: negative g favors rPINN")
    lines.append("(lower is better for all reported metrics)")
    ax1.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10)
    fig.suptitle("PINN vs rPINN Significance Summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_csv(records, out_csv_path):
    fieldnames = ["run_idx", "seed", "method", "train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "rate_mse", "best_loss", "stage_epoch_counts", "stage_durations"]
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            row_out = dict(row)
            row_out.pop("loss_history", None)
            row_out.pop("final_profile", None)
            row_out.pop("pred_rate", None)
            row_out.pop("true_rate", None)
            row_out.pop("k_learn", None)
            row_out.pop("D_learn", None)
            row_out["stage_epoch_counts"] = ",".join(str(v) for v in row.get("stage_epoch_counts", []))
            row_out["stage_durations"] = ",".join(f"{v:.6f}" for v in row.get("stage_durations", []))
            writer.writerow(row_out)


def plot_metric_distributions(records, out_path):
    methods = ["PINN", "rPINN"]
    colors = {"PINN": "#1f77b4", "rPINN": "#d62728"}
    markers = {"PINN": "o", "rPINN": "s"}
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metric_specs = [("train_time_s", "Training Time [s]", True), ("outlet_mse", "Outlet MSE", True), ("spatial_mse", "Spatial MSE", True), ("rate_mse", "Reaction-rate MSE", True)]
    for ax, (key, title, log_scale) in zip(axes, metric_specs):
        data = [np.asarray([r[key] for r in records if r["method"] == method], dtype=np.float64) for method in methods]
        bp = ax.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.25)
        for idx, method in enumerate(methods, start=1):
            if len(data[idx - 1]) > 0:
                jitter = (np.random.rand(len(data[idx - 1])) - 0.5) * 0.15
                ax.scatter(np.full(len(data[idx - 1]), idx) + jitter, data[idx - 1], alpha=0.8, s=32, c=colors[method], marker=markers[method])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(methods)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if log_scale:
            ax.set_yscale("log")
    fig.suptitle("PINN vs rPINN: Statistical Distribution Across Runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(records, out_path, n_points: int = 300):
    methods = ["PINN", "rPINN"]
    colors = {"PINN": "#1f77b4", "rPINN": "#d62728"}
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in methods:
        curves = [resample_curve(r["loss_history"], n_points=n_points) for r in records if r["method"] == method]
        if not curves:
            continue
        Y = np.vstack(curves)
        mean_y = np.nanmean(Y, axis=0)
        std_y = np.nanstd(Y, axis=0)
        x = np.linspace(0.0, 1.0, n_points)
        ax.plot(x, mean_y, color=colors[method], label=f"{method} mean")
        ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=colors[method], alpha=0.2, linewidth=0)
    ax.set_title("Convergence (loss over normalized training progress)")
    ax.set_xlabel("Normalized progress")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_convergence_data(records, out_path, n_points: int = 300):
    methods = ["PINN", "rPINN"]
    method_curves = {}
    for method in methods:
        curves = [resample_curve(r["loss_history"], n_points=n_points) for r in records if r["method"] == method]
        if not curves:
            method_curves[method] = {"mean": np.full((n_points,), np.nan, dtype=np.float64), "std": np.full((n_points,), np.nan, dtype=np.float64)}
            continue
        stacked = np.vstack(curves)
        method_curves[method] = {"mean": np.nanmean(stacked, axis=0), "std": np.nanstd(stacked, axis=0)}
    normalized_progress = np.linspace(0.0, 1.0, n_points)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["normalized_progress", "pinn_mean_loss", "pinn_std_loss", "rpinn_mean_loss", "rpinn_std_loss"])
        for idx in range(n_points):
            writer.writerow([float(normalized_progress[idx]), float(method_curves["PINN"]["mean"][idx]), float(method_curves["PINN"]["std"][idx]), float(method_curves["rPINN"]["mean"][idx]), float(method_curves["rPINN"]["std"][idx])])


def plot_speed_accuracy_scatter(records, out_path):
    methods = ["PINN", "rPINN"]
    colors = {"PINN": "#1f77b4", "rPINN": "#d62728"}
    markers = {"PINN": "o", "rPINN": "s"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in methods:
        subset = [r for r in records if r["method"] == method]
        xs = np.asarray([r["train_time_s"] for r in subset], dtype=np.float64)
        ys = np.asarray([r["rate_mse"] for r in subset], dtype=np.float64)
        ax.scatter(xs, ys, c=colors[method], marker=markers[method], s=50, alpha=0.8, label=method)
        if xs.size > 0:
            mx = float(np.mean(xs))
            my = float(np.mean(ys))
            sx = float(np.std(xs, ddof=1)) if xs.size > 1 else 0.0
            sy = float(np.std(ys, ddof=1)) if ys.size > 1 else 0.0
            ax.errorbar(mx, my, xerr=sx, yerr=sy, fmt="x", color=colors[method], capsize=4, markersize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training time [s]")
    ax.set_ylabel("Reaction-rate MSE")
    ax.set_title("Speed-Accuracy Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rpinn_stage_stats(records, rpinn_schedule, out_path):
    subset = [r for r in records if r["method"] == "rPINN"]
    if not subset:
        return
    stage_count = len(rpinn_schedule)
    durations = np.array([r.get("stage_durations", [np.nan] * stage_count) for r in subset], dtype=np.float64)
    epochs = np.array([r.get("stage_epoch_counts", [np.nan] * stage_count) for r in subset], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"S{i + 1}\n({d})" for i, d in enumerate(rpinn_schedule)]
    axes[0].boxplot([durations[:, i] for i in range(stage_count)], tick_labels=labels)
    axes[0].set_title("rPINN Stage Duration [s]")
    axes[0].set_ylabel("Seconds")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)
    axes[1].boxplot([epochs[:, i] for i in range(stage_count)], tick_labels=labels)
    axes[1].set_title("rPINN Stage Epochs Used")
    axes[1].set_ylabel("Epochs")
    axes[1].grid(alpha=0.3)
    fig.suptitle("rPINN Internal Stage Statistics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_reaction_rate_comparison(run_record_pinn, run_record_rpinn, time_axis, out_path):
    true_rate = run_record_pinn["true_rate"]
    pinn_rate = run_record_pinn["pred_rate"]
    rpinn_rate = run_record_rpinn["pred_rate"]

    true_mean = true_rate.mean(axis=2)
    pinn_mean = pinn_rate.mean(axis=2)
    rpinn_mean = rpinn_rate.mean(axis=2)

    species_labels = ["A", "B", "C"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(time_axis, true_mean[:, idx], label="Real", color="#2ca02c")
        ax.plot(time_axis, pinn_mean[:, idx], label="PINN predicted", color="#1f77b4", linestyle="--")
        ax.plot(time_axis, rpinn_mean[:, idx], label="rPINN predicted", color="#d62728", linestyle=":")
        ax.set_ylabel(f"r_{species_labels[idx]}")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Predicted vs Real Reaction Rates (spatial mean)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_reaction_rate_data(run_record_pinn, run_record_rpinn, time_axis, out_csv_path):
    true_rate = run_record_pinn["true_rate"].mean(axis=2)
    pinn_rate = run_record_pinn["pred_rate"].mean(axis=2)
    rpinn_rate = run_record_rpinn["pred_rate"].mean(axis=2)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "true_rate_A", "true_rate_B", "true_rate_C", "pinn_rate_A", "pinn_rate_B", "pinn_rate_C", "rpinn_rate_A", "rpinn_rate_B", "rpinn_rate_C"])
        for i, t in enumerate(time_axis):
            writer.writerow([float(t), float(true_rate[i, 0]), float(true_rate[i, 1]), float(true_rate[i, 2]), float(pinn_rate[i, 0]), float(pinn_rate[i, 1]), float(pinn_rate[i, 2]), float(rpinn_rate[i, 0]), float(rpinn_rate[i, 1]), float(rpinn_rate[i, 2])])


def print_summary(summary):
    for method_name in ["PINN", "rPINN"]:
        m = summary[method_name]
        print(f"\n{method_name} ({m['runs']} runs)")
        print(f"  train_time_s: mean={m['train_time_s']['mean']:.3f}, std={m['train_time_s']['std']:.3f}, min={m['train_time_s']['min']:.3f}, max={m['train_time_s']['max']:.3f}")
        print(f"  outlet_mse:   mean={m['outlet_mse']['mean']:.3e}, std={m['outlet_mse']['std']:.3e}, min={m['outlet_mse']['min']:.3e}, max={m['outlet_mse']['max']:.3e}")
        print(f"  spatial_mse:  mean={m['spatial_mse']['mean']:.3e}, std={m['spatial_mse']['std']:.3e}, min={m['spatial_mse']['min']:.3e}, max={m['spatial_mse']['max']:.3e}")
        print(f"  rate_mse:     mean={m['rate_mse']['mean']:.3e}, std={m['rate_mse']['std']:.3e}, min={m['rate_mse']['min']:.3e}, max={m['rate_mse']['max']:.3e}")


def parse_int_list(csv_text: str):
    return [int(token.strip()) for token in csv_text.split(",") if token.strip()]


def parse_float_list(csv_text: str):
    return [float(token.strip()) for token in csv_text.split(",") if token.strip()]


def build_tensors_for_training(data, num_delays: int, device: torch.device):
    cin_history, cin_current, target_profile, target_profile_3d = build_training_tensors(data, num_delays=num_delays)
    cout_target = torch.tensor(data["Cout"].T, dtype=torch.float32)
    time_vec_out = data["timeOutVec"]
    dt = float(time_vec_out[1] - time_vec_out[0]) if len(time_vec_out) > 1 else 1.0
    zeros_row = torch.zeros(1, cin_history.shape[1], dtype=cin_history.dtype)
    cin_history_prev = torch.cat((zeros_row, cin_history[:-1, :]), dim=0)
    return cin_history.to(device), cin_current.to(device), target_profile.to(device), target_profile_3d.to(device), cout_target.to(device), dt, cin_history_prev.to(device)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PINN vs rPINN on 3-species flow reactor data.")
    parser.add_argument("--data-path", type=str, default="./Data/flowReactorData_3species.pkl")
    parser.add_argument("--output-dir", type=str, default="./Data/rpinn_benchmark_3species")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-delays", type=int, default=30)
    parser.add_argument("--reactor-length", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--phys-weight", type=float, default=0.1)
    parser.add_argument("--loss-mode", type=str, default="strict_boundary_phys", choices=["strict_boundary_phys", "profile_phys"])
    parser.add_argument("--pinn-epochs", type=int, default=1800)
    parser.add_argument("--pinn-threshold", type=float, default=5e-5)
    parser.add_argument("--pinn-patience", type=int, default=120)
    parser.add_argument("--rpinn-schedule", type=str, default="4,20,70,100")
    parser.add_argument("--rpinn-epochs", type=str, default="250,400,700,900")
    parser.add_argument("--rpinn-thresholds", type=str, default="5e-4,2e-4,1e-4,5e-5")
    parser.add_argument("--rpinn-patience", type=int, default=60)
    parser.add_argument("--force-fixed-epochs", action="store_true", help="Force both methods to run fixed identical epoch counts (disable early stopping)")
    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--fixed-dispersion", type=float, default=1e-3)
    parser.add_argument("--fixed-reaction-rate", type=float, default=0.05)
    parser.add_argument("--no-trainable-params", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    rpinn_schedule = parse_int_list(args.rpinn_schedule)
    rpinn_epochs = parse_int_list(args.rpinn_epochs)
    rpinn_thresholds = parse_float_list(args.rpinn_thresholds)
    if not (len(rpinn_schedule) == len(rpinn_epochs) == len(rpinn_thresholds)):
        raise ValueError("rpinn-schedule, rpinn-epochs, and rpinn-thresholds must have equal lengths")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    # directory to save per-run loss histories
    loss_hist_dir = os.path.join(args.output_dir, "loss_histories")
    os.makedirs(loss_hist_dir, exist_ok=True)

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    true_k = float(reaction_rate_constant(float(data.get("temperature", 150.0)), float(data.get("kRef", 0.08)), float(data.get("eaKJPerMol", 35.0)) * 1000.0))

    cin_history, cin_current, target_profile, target_profile_3d, cout_target, dt, cin_history_prev = build_tensors_for_training(data=data, num_delays=args.num_delays, device=device)

    base_config = {
        "reactor_length": args.reactor_length,
        "learning_rate": args.learning_rate,
        "phys_weight": args.phys_weight,
        "loss_mode": args.loss_mode,
        "use_trainable_params": not args.no_trainable_params,
        "fixed_dispersion": args.fixed_dispersion,
        "fixed_reaction_rate": args.fixed_reaction_rate,
        "min_epochs_before_stop": args.min_epochs_before_stop,
        "cin_history_prev": cin_history_prev,
        "true_k": true_k,
    }

    pinn_config = dict(base_config)
    rpinn_config = dict(base_config)
    # if forcing fixed identical epochs, set pinn epochs to total rpinn epochs and disable early stopping
    if args.force_fixed_epochs:
        total_rpinn_epochs = sum(rpinn_epochs)
        pinn_config.update({"pinn_epochs": int(total_rpinn_epochs), "pinn_threshold": args.pinn_threshold, "pinn_patience": args.pinn_patience, "force_fixed_epochs": True})
        rpinn_config.update({"rpinn_schedule": rpinn_schedule, "rpinn_epochs": rpinn_epochs, "rpinn_thresholds": rpinn_thresholds, "rpinn_patience": args.rpinn_patience, "force_fixed_epochs": True})
    else:
        pinn_config.update({"pinn_epochs": args.pinn_epochs, "pinn_threshold": args.pinn_threshold, "pinn_patience": args.pinn_patience, "force_fixed_epochs": False})
        rpinn_config.update({"rpinn_schedule": rpinn_schedule, "rpinn_epochs": rpinn_epochs, "rpinn_thresholds": rpinn_thresholds, "rpinn_patience": args.rpinn_patience, "force_fixed_epochs": False})

    tensors = (cin_history, cin_current, target_profile, target_profile_3d, cout_target, dt)

    run_records = []
    representative = {}
    print(f"Running 3-species benchmark on {device} | runs={args.runs}")
    for run_idx in range(args.runs):
        seed = args.base_seed + run_idx
        print(f"Run {run_idx + 1}/{args.runs} | seed={seed}")
        pinn_result = train_pinn_once(pinn_config, tensors, device, seed)
        rpinn_result = train_rpinn_once(rpinn_config, tensors, device, seed)
        # save per-run loss histories for later unnormalized plotting
        try:
            np.savez_compressed(os.path.join(loss_hist_dir, f"pinn_run_{run_idx}.npz"), loss_history=pinn_result.get("loss_history", np.asarray([])))
            np.savez_compressed(os.path.join(loss_hist_dir, f"rpinn_run_{run_idx}.npz"), loss_history=rpinn_result.get("loss_history", np.asarray([])))
        except Exception:
            pass
        if run_idx == 0:
            representative["PINN"] = pinn_result
            representative["rPINN"] = rpinn_result
        run_records.append({"run_idx": run_idx, "seed": seed, "method": pinn_result["method"], "train_time_s": pinn_result["train_time_s"], "epochs_used": pinn_result["epochs_used"], "outlet_mse": pinn_result["outlet_mse"], "spatial_mse": pinn_result["spatial_mse"], "rate_mse": pinn_result["rate_mse"], "best_loss": pinn_result["best_loss"], "loss_history": pinn_result["loss_history"]})
        run_records.append({"run_idx": run_idx, "seed": seed, "method": rpinn_result["method"], "train_time_s": rpinn_result["train_time_s"], "epochs_used": rpinn_result["epochs_used"], "outlet_mse": rpinn_result["outlet_mse"], "spatial_mse": rpinn_result["spatial_mse"], "rate_mse": rpinn_result["rate_mse"], "best_loss": rpinn_result["best_loss"], "loss_history": rpinn_result["loss_history"], "stage_epoch_counts": rpinn_result.get("stage_epoch_counts", []), "stage_durations": rpinn_result.get("stage_durations", [])})
        print(f"  PINN: time={pinn_result['train_time_s']:.2f}s, outlet_mse={pinn_result['outlet_mse']:.3e}, spatial_mse={pinn_result['spatial_mse']:.3e}, rate_mse={pinn_result['rate_mse']:.3e}")
        print(f"  rPINN: time={rpinn_result['train_time_s']:.2f}s, outlet_mse={rpinn_result['outlet_mse']:.3e}, spatial_mse={rpinn_result['spatial_mse']:.3e}, rate_mse={rpinn_result['rate_mse']:.3e}")

    summary = {
        "PINN": summarize_method(run_records, "PINN"),
        "rPINN": summarize_method(run_records, "rPINN"),
        "stat_tests": build_stat_tests(run_records),
        "config": {"runs": args.runs, "base_seed": args.base_seed, "device": str(device), "num_delays": args.num_delays, "reactor_length": args.reactor_length, "learning_rate": args.learning_rate, "phys_weight": args.phys_weight, "loss_mode": args.loss_mode, "pinn_epochs": args.pinn_epochs, "pinn_threshold": args.pinn_threshold, "pinn_patience": args.pinn_patience, "rpinn_schedule": rpinn_schedule, "rpinn_epochs": rpinn_epochs, "rpinn_thresholds": rpinn_thresholds, "rpinn_patience": args.rpinn_patience, "min_epochs_before_stop": args.min_epochs_before_stop, "use_trainable_params": not args.no_trainable_params, "fixed_dispersion": args.fixed_dispersion, "fixed_reaction_rate": args.fixed_reaction_rate, "data_path": args.data_path, "true_k": true_k},
    }

    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    save_csv(run_records, csv_path)

    summary_path = os.path.join(args.output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    stat_json_path = os.path.join(args.output_dir, "statistical_tests.json")
    with open(stat_json_path, "w", encoding="utf-8") as f:
        json.dump(summary["stat_tests"], f, indent=2)

    stat_txt_path = os.path.join(args.output_dir, "statistical_report.txt")
    save_stats_text_report(summary["stat_tests"], stat_txt_path)

    plot_metric_distributions(run_records, out_path=os.path.join(args.output_dir, "metric_distributions.png"))
    plot_convergence(run_records, out_path=os.path.join(args.output_dir, "convergence_mean_std.png"), n_points=300)
    convergence_data_path = os.path.join(args.output_dir, "convergence_tikz_data.csv")
    save_convergence_data(run_records, out_path=convergence_data_path, n_points=300)
    plot_speed_accuracy_scatter(run_records, out_path=os.path.join(args.output_dir, "speed_vs_accuracy.png"))
    plot_rpinn_stage_stats(run_records, rpinn_schedule=rpinn_schedule, out_path=os.path.join(args.output_dir, "rpinn_stage_stats.png"))
    plot_statistical_report(summary["stat_tests"], out_path=os.path.join(args.output_dir, "statistical_report.png"))

    if representative:
        reaction_rate_plot = os.path.join(args.output_dir, "reaction_rate_comparison.png")
        plot_reaction_rate_comparison(representative["PINN"], representative["rPINN"], np.asarray(data["timeOutVec"], dtype=np.float64), reaction_rate_plot)
        reaction_rate_data = os.path.join(args.output_dir, "reaction_rate_comparison.csv")
        save_reaction_rate_data(representative["PINN"], representative["rPINN"], np.asarray(data["timeOutVec"], dtype=np.float64), reaction_rate_data)

    print_summary(summary)
    print("\nSaved outputs:")
    for p in [csv_path, summary_path, stat_json_path, stat_txt_path, os.path.join(args.output_dir, "metric_distributions.png"), os.path.join(args.output_dir, "convergence_mean_std.png"), convergence_data_path, os.path.join(args.output_dir, "speed_vs_accuracy.png"), os.path.join(args.output_dir, "rpinn_stage_stats.png"), os.path.join(args.output_dir, "statistical_report.png"), os.path.join(args.output_dir, "reaction_rate_comparison.png"), os.path.join(args.output_dir, "reaction_rate_comparison.csv")]:
        if os.path.exists(p):
            print(f"  - {p}")


if __name__ == "__main__":
    main()