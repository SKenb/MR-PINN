import argparse
import csv
import json
import math
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FullPINN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        return self.fc3(out)


class RPinN(nn.Module):
    def __init__(self, input_dim=11, output_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        return self.fc3(out)

    def expand_output(self, new_output_dim):
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


def build_lagged_cin_features(cin, num_delays=10):
    cin = np.asarray(cin, dtype=np.float32)
    n_samples = cin.shape[0]
    features = np.zeros((n_samples, num_delays + 1), dtype=np.float32)

    for sample_idx in range(n_samples):
        for delay_idx in range(num_delays + 1):
            source_idx = sample_idx - delay_idx
            if source_idx >= 0:
                features[sample_idx, delay_idx] = cin[source_idx]

    return torch.tensor(features, dtype=torch.float32)


def build_training_tensors(data, num_delays=10):
    conc_vec_in = data["Cin"]
    c_spatial = data["Cspatial"]

    n_samples = conc_vec_in.shape[1]
    cin_current = torch.tensor(conc_vec_in[0, :], dtype=torch.float32).view(n_samples, 1)
    cin_history = build_lagged_cin_features(conc_vec_in[0, :], num_delays=num_delays)
    target_profile = torch.tensor(c_spatial[0, 1:, :].T, dtype=torch.float32)

    return cin_history, cin_current, target_profile


def select_anchor_indices(total_points, out_dim):
    if out_dim <= 1:
        return np.array([total_points - 1], dtype=int)
    return np.unique(np.round(np.linspace(0, total_points - 1, out_dim)).astype(int))


def reconstruct_full_profile(anchor_profile, anchor_indices, total_points, interpolation_method: str = "linear"):
    """Reconstruct full spatial profile from anchor values.

    interpolation_method: one of 'linear', 'nearest', 'spline'
    """
    anchor_profile = np.asarray(anchor_profile, dtype=np.float32)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    reconstructed = np.zeros((anchor_profile.shape[0], total_points), dtype=np.float32)
    full_x = np.arange(total_points, dtype=np.float32)

    for row_idx in range(anchor_profile.shape[0]):
        row_vals = anchor_profile[row_idx, :]
        if interpolation_method == "nearest":
            # for each full_x pick nearest anchor index
            # vectorized: compute distances and argmin
            idx = np.argmin(np.abs(anchor_indices.reshape(-1, 1) - full_x.reshape(1, -1)), axis=0)
            reconstructed[row_idx, :] = row_vals[idx]
        elif interpolation_method == "linear":
            reconstructed[row_idx, :] = np.interp(full_x, anchor_indices.astype(np.float32), row_vals)
        elif interpolation_method == "spline":
            try:
                from scipy.interpolate import CubicSpline

                if anchor_indices.size < 4 or row_vals.size < 4:
                    reconstructed[row_idx, :] = np.interp(full_x, anchor_indices.astype(np.float32), row_vals)
                else:
                    cs = CubicSpline(anchor_indices.astype(np.float64), row_vals.astype(np.float64), bc_type="natural")
                    reconstructed[row_idx, :] = cs(full_x).astype(np.float32)
            except Exception:
                reconstructed[row_idx, :] = np.interp(full_x, anchor_indices.astype(np.float32), row_vals)
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")

    return reconstructed


def compute_residual(predicted_profile, predicted_prev, dt, reactor_length, D, k, q=1.0):
    dCdt = (predicted_profile - predicted_prev) / dt

    Nx = predicted_profile.shape[1]
    dx = reactor_length / (Nx - 1) if Nx > 1 else reactor_length

    dCdx = torch.zeros_like(predicted_profile)
    d2Cdx2 = torch.zeros_like(predicted_profile)

    if Nx > 2:
        dCdx[:, 1:-1] = (predicted_profile[:, 2:] - predicted_profile[:, :-2]) / (2.0 * dx)
        d2Cdx2[:, 1:-1] = (
            predicted_profile[:, 2:] - 2.0 * predicted_profile[:, 1:-1] + predicted_profile[:, :-2]
        ) / (dx ** 2)

    if Nx > 1:
        dCdx[:, 0] = (predicted_profile[:, 1] - predicted_profile[:, 0]) / dx
        dCdx[:, -1] = (predicted_profile[:, -1] - predicted_profile[:, -2]) / dx

    if Nx > 2:
        d2Cdx2[:, 0] = (
            predicted_profile[:, 2] - 2.0 * predicted_profile[:, 1] + predicted_profile[:, 0]
        ) / (dx ** 2)
        d2Cdx2[:, -1] = (
            predicted_profile[:, -1] - 2.0 * predicted_profile[:, -2] + predicted_profile[:, -3]
        ) / (dx ** 2)

    residual = dCdt - (D * d2Cdx2 - q * dCdx - k * predicted_profile)
    return residual


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def compute_supervised_loss(loss_mode, predicted_profile, target_profile, cin_current, cout_target):
    if loss_mode == "strict_boundary_phys":
        return (
            torch.mean((predicted_profile[:, 0] - cin_current[:, 0]) ** 2)
            + torch.mean((predicted_profile[:, -1] - cout_target[:, 0]) ** 2)
        )
    if loss_mode == "profile_phys":
        return torch.mean((predicted_profile - target_profile) ** 2)
    raise ValueError(f"Unknown loss_mode: {loss_mode}")


def train_pinn_once(config, tensors, device, seed):
    set_global_seed(seed)

    cin_history, cin_current, target_profile, cout_target, dt = tensors

    model = FullPINN(input_dim=cin_history.shape[1], output_dim=target_profile.shape[1]).to(device)

    if config["use_trainable_params"]:
        init_D = 1e-3
        raw_init_D = np.log(np.expm1(init_D))
        model.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32, device=device))
        model.k_param = nn.Parameter(torch.tensor(0.25, dtype=torch.float32, device=device))

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

        predicted_profile = model(cin_history)
        predicted_prev = model(config["cin_history_prev"])

        supervised_loss = compute_supervised_loss(
            loss_mode=config["loss_mode"],
            predicted_profile=predicted_profile,
            target_profile=target_profile,
            cin_current=cin_current,
            cout_target=cout_target,
        )

        if config["use_trainable_params"]:
            D = F.softplus(model.D_param)
            k = F.softplus(model.k_param)
        else:
            D = torch.tensor(config["fixed_dispersion"], dtype=torch.float32, device=device)
            k = torch.tensor(config["fixed_reaction_rate"], dtype=torch.float32, device=device)

        residual = compute_residual(
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

    # Optional finetune phase: continue training only the reaction-rate param (k_param)
    if config.get("use_trainable_params", False) and config.get("k_finetune_epochs", 0) > 0:
        k_epochs = int(config.get("k_finetune_epochs", 0))
        # optimize only k_param to refine reaction rate without early stopping
        if hasattr(model, "k_param"):
            finetune_optimizer = optim.Adam([model.k_param], lr=config["learning_rate"])
            for ke in range(k_epochs):
                model.train()
                finetune_optimizer.zero_grad()

                predicted_profile = model(cin_history)
                predicted_prev = model(config["cin_history_prev"])

                supervised_loss = compute_supervised_loss(
                    loss_mode=config["loss_mode"],
                    predicted_profile=predicted_profile,
                    target_profile=target_profile,
                    cin_current=cin_current,
                    cout_target=cout_target,
                )

                D = F.softplus(model.D_param) if config.get("use_trainable_params", False) else torch.tensor(
                    config["fixed_dispersion"], dtype=torch.float32, device=device
                )
                k = F.softplus(model.k_param)

                residual = compute_residual(
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
                loss_value = float(loss.item())
                loss_history.append(loss_value)
                epoch_used += 1

    synchronize_if_needed(device)
    train_time_s = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        final_profile = model(cin_history)

    if config["use_trainable_params"]:
        reaction_rate = float(F.softplus(model.k_param).detach().cpu().item())
    else:
        reaction_rate = float(config["fixed_reaction_rate"])

    spatial_mse = float(torch.mean((final_profile - target_profile) ** 2).item())
    outlet_mse = float(torch.mean((final_profile[:, -1] - cout_target[:, 0]) ** 2).item())
    reaction_rate_abs_error = float(abs(reaction_rate - float(config["fixed_reaction_rate"])))

    return {
        "method": "PINN",
        "train_time_s": train_time_s,
        "epochs_used": epoch_used,
        "outlet_mse": outlet_mse,
        "spatial_mse": spatial_mse,
        "best_loss": best_loss,
        "reaction_rate": reaction_rate,
        "reaction_rate_abs_error": reaction_rate_abs_error,
        "loss_history": np.asarray(loss_history, dtype=np.float64),
    }


def train_rpinn_once(config, tensors, device, seed):
    set_global_seed(seed)

    cin_history, cin_current, target_profile, cout_target, dt = tensors
    total_points = target_profile.shape[1]

    schedule = config["rpinn_schedule"]
    stage_epochs = config["rpinn_epochs"]
    stage_thresholds = config["rpinn_thresholds"]

    model = RPinN(input_dim=cin_history.shape[1], output_dim=schedule[0]).to(device)

    if config["use_trainable_params"]:
        init_D = 1e-3
        raw_init_D = np.log(np.expm1(init_D))
        model.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32, device=device))
        model.k_param = nn.Parameter(torch.tensor(0.25, dtype=torch.float32, device=device))

    all_loss_history = []
    stage_epoch_counts = []
    stage_durations = []
    best_loss_global = float("inf")

    synchronize_if_needed(device)
    start_time = time.perf_counter()

    for stage_idx, out_dim in enumerate(schedule):
        if stage_idx > 0:
            model.expand_output(out_dim)
            model = model.to(device)

        anchor_indices = select_anchor_indices(total_points, out_dim)
        anchor_target = target_profile[:, anchor_indices]

        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        best_loss = float("inf")
        stale = 0

        synchronize_if_needed(device)
        stage_start = time.perf_counter()

        stage_epoch_used = 0
        for epoch in range(stage_epochs[stage_idx]):
            model.train()
            optimizer.zero_grad()

            predicted_profile = model(cin_history)
            predicted_prev = model(config["cin_history_prev"])

            supervised_loss = compute_supervised_loss(
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

            residual = compute_residual(
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

        synchronize_if_needed(device)
        stage_durations.append(time.perf_counter() - stage_start)
        stage_epoch_counts.append(stage_epoch_used)
        best_loss_global = min(best_loss_global, best_loss)

    synchronize_if_needed(device)
    # Optional finetune phase: continue training only the reaction-rate param (k_param)
    if config.get("use_trainable_params", False) and config.get("k_finetune_epochs", 0) > 0:
        k_epochs = int(config.get("k_finetune_epochs", 0))
        if hasattr(model, "k_param"):
            finetune_optimizer = optim.Adam([model.k_param], lr=config["learning_rate"])
            for ke in range(k_epochs):
                model.train()
                finetune_optimizer.zero_grad()

                # use full-anchor prediction for finetune
                predicted_profile = model(cin_history)
                predicted_prev = model(config["cin_history_prev"])

                anchor_indices_full = select_anchor_indices(total_points, model.fc3.out_features)
                anchor_target = target_profile[:, anchor_indices_full]

                supervised_loss = compute_supervised_loss(
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

                residual = compute_residual(
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
                # count finetune epochs in the reported epoch counts
                stage_epoch_counts[-1] = int(stage_epoch_counts[-1]) + 1

    train_time_s = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        anchor_pred = model(cin_history).detach().cpu().numpy()

    if config["use_trainable_params"]:
        reaction_rate = float(F.softplus(model.k_param).detach().cpu().item())
    else:
        reaction_rate = float(config["fixed_reaction_rate"])

    anchor_indices = select_anchor_indices(total_points, model.fc3.out_features)
    full_pred_profile = reconstruct_full_profile(anchor_pred, anchor_indices, total_points)

    target_np = target_profile.detach().cpu().numpy()
    cout_np = cout_target.detach().cpu().numpy()[:, 0]

    spatial_mse = float(np.mean((full_pred_profile - target_np) ** 2))
    outlet_mse = float(np.mean((full_pred_profile[:, -1] - cout_np) ** 2))
    reaction_rate_abs_error = float(abs(reaction_rate - float(config["fixed_reaction_rate"])))

    return {
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


def resample_curve(y, n_points=300):
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return np.full((n_points,), np.nan, dtype=np.float64)
    if y.size == 1:
        return np.full((n_points,), y[0], dtype=np.float64)

    old_x = np.linspace(0.0, 1.0, y.size)
    new_x = np.linspace(0.0, 1.0, n_points)
    return np.interp(new_x, old_x, y)


def summarize_method(records, method):
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
        "best_loss": agg("best_loss"),
        "reaction_rate": agg("reaction_rate"),
        "reaction_rate_abs_error": agg("reaction_rate_abs_error"),
    }


def _normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _welch_p_value_two_sided(t_stat, df):
    try:
        from scipy import stats as scipy_stats  # type: ignore

        return float(2.0 * scipy_stats.t.sf(abs(t_stat), df)), "scipy_t"
    except Exception:
        # Fallback normal approximation when scipy is unavailable.
        return float(2.0 * _normal_sf(abs(t_stat))), "normal_approx"


def welch_ttest_and_effect_size(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n_x = x.size
    n_y = y.size
    if n_x < 2 or n_y < 2:
        return {
            "n_pinn": int(n_x),
            "n_rpinn": int(n_y),
            "mean_pinn": float(np.mean(x)) if n_x else float("nan"),
            "mean_rpinn": float(np.mean(y)) if n_y else float("nan"),
            "mean_diff_pinn_minus_rpinn": float(np.mean(x) - np.mean(y)) if n_x and n_y else float("nan"),
            "relative_change_rpinn_vs_pinn_percent": float("nan"),
            "t_stat": float("nan"),
            "df": float("nan"),
            "p_value": float("nan"),
            "p_method": "insufficient_samples",
            "cohens_d": float("nan"),
            "hedges_g": float("nan"),
            "ci95_diff": [float("nan"), float("nan")],
        }

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

    # 95% CI for mean difference. Use t critical if scipy exists, else normal 1.96.
    try:
        from scipy import stats as scipy_stats  # type: ignore

        t_crit = float(scipy_stats.t.ppf(0.975, df)) if np.isfinite(df) else 1.96
    except Exception:
        t_crit = 1.96

    diff = mean_x - mean_y
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se

    rel = ((mean_y - mean_x) / mean_x * 100.0) if mean_x != 0 else float("nan")

    return {
        "n_pinn": int(n_x),
        "n_rpinn": int(n_y),
        "mean_pinn": mean_x,
        "mean_rpinn": mean_y,
        "mean_diff_pinn_minus_rpinn": diff,
        "relative_change_rpinn_vs_pinn_percent": rel,
        "t_stat": float(t_stat),
        "df": float(df),
        "p_value": float(p_value),
        "p_method": p_method,
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "ci95_diff": [float(ci_low), float(ci_high)],
    }


def build_stat_tests(records):
    metrics = ["train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "best_loss", "reaction_rate_abs_error"]
    results = {}
    for metric in metrics:
        x = [r[metric] for r in records if r["method"] == "PINN"]
        y = [r[metric] for r in records if r["method"] == "rPINN"]
        results[metric] = welch_ttest_and_effect_size(x, y)
    return results


def save_stats_text_report(stat_tests, out_path):
    lines = []
    lines.append("PINN vs rPINN Statistical Report")
    lines.append("Columns: mean(PINN), mean(rPINN), diff=PINN-rPINN, p-value, Hedges g")
    lines.append("")

    for metric in ["train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "best_loss", "reaction_rate_abs_error"]:
        s = stat_tests[metric]
        lines.append(
            f"{metric}: mean_pinn={s['mean_pinn']:.6g}, mean_rpinn={s['mean_rpinn']:.6g}, "
            f"diff={s['mean_diff_pinn_minus_rpinn']:.6g}, p={s['p_value']:.6g}, g={s['hedges_g']:.6g}, "
            f"ci95=[{s['ci95_diff'][0]:.6g}, {s['ci95_diff'][1]:.6g}] ({s['p_method']})"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def plot_statistical_report(stat_tests, out_path):
    metrics = ["train_time_s", "epochs_used", "outlet_mse", "spatial_mse", "best_loss", "reaction_rate_abs_error"]
    pretty = {
        "train_time_s": "Train time",
        "epochs_used": "Epochs used",
        "outlet_mse": "Outlet MSE",
        "spatial_mse": "Spatial MSE",
        "best_loss": "Best loss",
        "reaction_rate_abs_error": "Reaction-rate abs. error",
    }

    g_vals = np.array([stat_tests[m]["hedges_g"] for m in metrics], dtype=np.float64)
    p_vals = np.array([stat_tests[m]["p_value"] for m in metrics], dtype=np.float64)

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
    for m, p in zip(metrics, p_vals):
        s = stat_tests[m]
        sig = "*" if np.isfinite(p) and p < 0.05 else "ns"
        lines.append(
            f"{pretty[m]}: p={p:.3g} ({sig}), t={s['t_stat']:.3g}, df={s['df']:.1f}, "
            f"diff={s['mean_diff_pinn_minus_rpinn']:.3g}"
        )
    lines.append("")
    lines.append("Interpretation: negative g favors rPINN")
    lines.append("(lower is better for all reported metrics)")
    ax1.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.suptitle("PINN vs rPINN Significance Summary")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_csv(records, out_csv_path):
    fieldnames = [
        "run_idx",
        "seed",
        "method",
        "train_time_s",
        "epochs_used",
        "outlet_mse",
        "spatial_mse",
        "best_loss",
        "reaction_rate",
        "reaction_rate_abs_error",
        "stage_epoch_counts",
        "stage_durations",
    ]

    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            row_out = dict(row)
            row_out.pop("loss_history", None)
            row_out["stage_epoch_counts"] = ",".join(str(v) for v in row.get("stage_epoch_counts", []))
            row_out["stage_durations"] = ",".join(f"{v:.6f}" for v in row.get("stage_durations", []))
            writer.writerow(row_out)


def _scatter_jittered(ax, x_center, y_values, color, marker):
    if len(y_values) == 0:
        return
    jitter = (np.random.rand(len(y_values)) - 0.5) * 0.15
    xs = np.full((len(y_values),), x_center, dtype=np.float64) + jitter
    ax.scatter(xs, y_values, alpha=0.8, s=32, c=color, marker=marker)


def plot_metric_distributions(records, out_path):
    methods = ["PINN", "rPINN"]
    colors = {"PINN": "#1f77b4", "rPINN": "#d62728"}
    markers = {"PINN": "o", "rPINN": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    metric_specs = [
        ("train_time_s", "Training Time [s]", True),
        ("outlet_mse", "Outlet MSE", True),
        ("spatial_mse", "Spatial MSE", True),
    ]

    for ax, (key, title, log_scale) in zip(axes, metric_specs):
        data = [np.asarray([r[key] for r in records if r["method"] == method], dtype=np.float64) for method in methods]

        bp = ax.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True)
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.25)

        for idx, method in enumerate(methods, start=1):
            _scatter_jittered(ax, idx, data[idx - 1], colors[method], markers[method])

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


def plot_convergence(records, out_path, n_points=300):
    methods = ["PINN", "rPINN"]
    colors = {"PINN": "#1f77b4", "rPINN": "#d62728"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for method in methods:
        curves = [resample_curve(r["loss_history"], n_points=n_points) for r in records if r["method"] == method]
        if len(curves) == 0:
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


def save_convergence_data(records, out_path, n_points=300):
    methods = ["PINN", "rPINN"]
    method_curves = {}

    for method in methods:
        curves = [resample_curve(r["loss_history"], n_points=n_points) for r in records if r["method"] == method]
        if len(curves) == 0:
            method_curves[method] = {
                "mean": np.full((n_points,), np.nan, dtype=np.float64),
                "std": np.full((n_points,), np.nan, dtype=np.float64),
            }
            continue

        stacked = np.vstack(curves)
        method_curves[method] = {
            "mean": np.nanmean(stacked, axis=0),
            "std": np.nanstd(stacked, axis=0),
        }

    normalized_progress = np.linspace(0.0, 1.0, n_points)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "normalized_progress",
            "pinn_mean_loss",
            "pinn_std_loss",
            "rpinn_mean_loss",
            "rpinn_std_loss",
        ])
        for idx in range(n_points):
            writer.writerow([
                float(normalized_progress[idx]),
                float(method_curves["PINN"]["mean"][idx]),
                float(method_curves["PINN"]["std"][idx]),
                float(method_curves["rPINN"]["mean"][idx]),
                float(method_curves["rPINN"]["std"][idx]),
            ])


def plot_speed_accuracy_scatter(records, out_path):
    methods = ["PINN", "rPINN"]
    colors = {"PINN": "#1f77b4", "rPINN": "#d62728"}
    markers = {"PINN": "o", "rPINN": "s"}

    fig, ax = plt.subplots(figsize=(8, 6))

    for method in methods:
        subset = [r for r in records if r["method"] == method]
        xs = np.asarray([r["train_time_s"] for r in subset], dtype=np.float64)
        ys = np.asarray([r["outlet_mse"] for r in subset], dtype=np.float64)

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
    ax.set_ylabel("Outlet MSE")
    ax.set_title("Speed-Accuracy Tradeoff")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_rpinn_stage_stats(records, rpinn_schedule, out_path):
    subset = [r for r in records if r["method"] == "rPINN"]
    if len(subset) == 0:
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


def parse_int_list(csv_text):
    return [int(token.strip()) for token in csv_text.split(",") if token.strip()]


def parse_float_list(csv_text):
    return [float(token.strip()) for token in csv_text.split(",") if token.strip()]


def build_tensors_for_training(data, num_delays, device):
    cin_history, cin_current, target_profile = build_training_tensors(data, num_delays=num_delays)
    cout_target = torch.tensor(data["Cout"][0, :], dtype=torch.float32).view(-1, 1)

    time_vec_out = data["timeOutVec"]
    dt = float(time_vec_out[1] - time_vec_out[0]) if len(time_vec_out) > 1 else 1.0

    zeros_row = torch.zeros(1, cin_history.shape[1], dtype=cin_history.dtype)
    cin_history_prev = torch.cat((zeros_row, cin_history[:-1, :]), dim=0)

    return (
        cin_history.to(device),
        cin_current.to(device),
        target_profile.to(device),
        cout_target.to(device),
        dt,
        cin_history_prev.to(device),
    )


def print_summary(summary):
    for method_name in ["PINN", "rPINN"]:
        m = summary[method_name]
        print(f"\n{method_name} ({m['runs']} runs)")
        print(
            "  train_time_s: "
            f"mean={m['train_time_s']['mean']:.3f}, std={m['train_time_s']['std']:.3f}, "
            f"min={m['train_time_s']['min']:.3f}, max={m['train_time_s']['max']:.3f}"
        )
        print(
            "  outlet_mse:   "
            f"mean={m['outlet_mse']['mean']:.3e}, std={m['outlet_mse']['std']:.3e}, "
            f"min={m['outlet_mse']['min']:.3e}, max={m['outlet_mse']['max']:.3e}"
        )
        print(
            "  spatial_mse:  "
            f"mean={m['spatial_mse']['mean']:.3e}, std={m['spatial_mse']['std']:.3e}, "
            f"min={m['spatial_mse']['min']:.3e}, max={m['spatial_mse']['max']:.3e}"
        )
        print(
            "  reaction_rate: "
            f"mean={m['reaction_rate']['mean']:.6g}, std={m['reaction_rate']['std']:.6g}, "
            f"min={m['reaction_rate']['min']:.6g}, max={m['reaction_rate']['max']:.6g}"
        )
        print(
            "  k_abs_error:   "
            f"mean={m['reaction_rate_abs_error']['mean']:.3e}, std={m['reaction_rate_abs_error']['std']:.3e}, "
            f"min={m['reaction_rate_abs_error']['min']:.3e}, max={m['reaction_rate_abs_error']['max']:.3e}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark PINN vs rPINN training speed and accuracy.")
    parser.add_argument("--data-path", type=str, default="./Data/flowReactorData.pkl")
    parser.add_argument("--output-dir", type=str, default="./Data/rpinn_benchmark")
    parser.add_argument("--runs", type=int, default=5)
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
        help="Shared supervised term for both methods. strict_boundary_phys is the strict apples-to-apples mode.",
    )

    parser.add_argument("--pinn-epochs", type=int, default=2900)
    parser.add_argument("--pinn-threshold", type=float, default=5e-5)
    parser.add_argument("--pinn-patience", type=int, default=120)

    parser.add_argument("--rpinn-schedule", type=str, default="4,20,70,100")
    parser.add_argument("--rpinn-epochs", type=str, default="300,500,900,1200")
    parser.add_argument("--rpinn-thresholds", type=str, default="5e-4,2e-4,1e-4,5e-5")
    parser.add_argument("--rpinn-patience", type=int, default=60)

    parser.add_argument("--min-epochs-before-stop", type=int, default=20)
    parser.add_argument("--force-fixed-epochs", action="store_true", help="Force both PINN and rPINN to run their full configured epoch budgets")
    parser.add_argument("--fixed-dispersion", type=float, default=1e-3)
    parser.add_argument("--fixed-reaction-rate", type=float, default=0.25)
    parser.add_argument("--no-trainable-params", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--k-finetune-epochs", type=int, default=0, help="Number of epochs to finetune only k_param after early stopping (default 0)")

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
    loss_hist_dir = os.path.join(args.output_dir, "loss_histories")
    os.makedirs(loss_hist_dir, exist_ok=True)

    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    cin_history, cin_current, target_profile, cout_target, dt, cin_history_prev = build_tensors_for_training(
        data=data,
        num_delays=args.num_delays,
        device=device,
    )

    base_config = {
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

    pinn_config = dict(base_config)
    rpinn_config = dict(base_config)
    if args.force_fixed_epochs:
        total_rpinn_epochs = sum(rpinn_epochs)
        pinn_config.update(
            {
                "pinn_epochs": int(total_rpinn_epochs),
                "pinn_threshold": args.pinn_threshold,
                "pinn_patience": args.pinn_patience,
                "force_fixed_epochs": True,
            }
        )
        rpinn_config.update(
            {
                "rpinn_schedule": rpinn_schedule,
                "rpinn_epochs": rpinn_epochs,
                "rpinn_thresholds": rpinn_thresholds,
                "rpinn_patience": args.rpinn_patience,
                "force_fixed_epochs": True,
            }
        )
    else:
        pinn_config.update(
            {
                "pinn_epochs": args.pinn_epochs,
                "pinn_threshold": args.pinn_threshold,
                "pinn_patience": args.pinn_patience,
                "force_fixed_epochs": False,
            }
        )
        rpinn_config.update(
            {
                "rpinn_schedule": rpinn_schedule,
                "rpinn_epochs": rpinn_epochs,
                "rpinn_thresholds": rpinn_thresholds,
                "rpinn_patience": args.rpinn_patience,
                "force_fixed_epochs": False,
            }
        )

    tensors = (cin_history, cin_current, target_profile, cout_target, dt)

    run_records = []

    print(f"Running benchmark on {device} | runs={args.runs}")
    for run_idx in range(args.runs):
        seed = args.base_seed + run_idx
        print(f"Run {run_idx + 1}/{args.runs} | seed={seed}")

        pinn_result = train_pinn_once(pinn_config, tensors, device, seed)
        rpinn_result = train_rpinn_once(rpinn_config, tensors, device, seed)

        run_records.append(
            {
                "run_idx": run_idx,
                "seed": seed,
                "method": pinn_result["method"],
                "train_time_s": pinn_result["train_time_s"],
                "epochs_used": pinn_result["epochs_used"],
                "outlet_mse": pinn_result["outlet_mse"],
                "spatial_mse": pinn_result["spatial_mse"],
                "best_loss": pinn_result["best_loss"],
                "reaction_rate": pinn_result["reaction_rate"],
                "reaction_rate_abs_error": pinn_result["reaction_rate_abs_error"],
                "loss_history": pinn_result["loss_history"],
            }
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
        np.save(os.path.join(loss_hist_dir, f"pinn_run_{run_idx}.npy"), np.asarray(pinn_result["loss_history"], dtype=np.float64))
        np.save(os.path.join(loss_hist_dir, f"rpinn_run_{run_idx}.npy"), np.asarray(rpinn_result["loss_history"], dtype=np.float64))

        print(
            "  PINN: "
            f"time={pinn_result['train_time_s']:.2f}s, "
            f"outlet_mse={pinn_result['outlet_mse']:.3e}, "
            f"spatial_mse={pinn_result['spatial_mse']:.3e}, "
            f"k={pinn_result['reaction_rate']:.6g}"
        )
        print(
            "  rPINN: "
            f"time={rpinn_result['train_time_s']:.2f}s, "
            f"outlet_mse={rpinn_result['outlet_mse']:.3e}, "
            f"spatial_mse={rpinn_result['spatial_mse']:.3e}, "
            f"k={rpinn_result['reaction_rate']:.6g}"
        )

    summary = {
        "PINN": summarize_method(run_records, "PINN"),
        "rPINN": summarize_method(run_records, "rPINN"),
        "stat_tests": build_stat_tests(run_records),
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
        },
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

    plot_metric_distributions(
        run_records,
        out_path=os.path.join(args.output_dir, "metric_distributions.png"),
    )
    plot_convergence(
        run_records,
        out_path=os.path.join(args.output_dir, "convergence_mean_std.png"),
        n_points=300,
    )
    convergence_data_path = os.path.join(args.output_dir, "convergence_tikz_data.csv")
    save_convergence_data(
        run_records,
        out_path=convergence_data_path,
        n_points=300,
    )
    plot_speed_accuracy_scatter(
        run_records,
        out_path=os.path.join(args.output_dir, "speed_vs_accuracy.png"),
    )
    plot_rpinn_stage_stats(
        run_records,
        rpinn_schedule=rpinn_schedule,
        out_path=os.path.join(args.output_dir, "rpinn_stage_stats.png"),
    )
    plot_statistical_report(
        summary["stat_tests"],
        out_path=os.path.join(args.output_dir, "statistical_report.png"),
    )

    print_summary(summary)
    print("\nSaved outputs:")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {stat_json_path}")
    print(f"  - {stat_txt_path}")
    print(f"  - {os.path.join(args.output_dir, 'metric_distributions.png')}")
    print(f"  - {os.path.join(args.output_dir, 'convergence_mean_std.png')}")
    print(f"  - {convergence_data_path}")
    print(f"  - {os.path.join(args.output_dir, 'speed_vs_accuracy.png')}")
    print(f"  - {os.path.join(args.output_dir, 'rpinn_stage_stats.png')}")
    print(f"  - {os.path.join(args.output_dir, 'statistical_report.png')}")


if __name__ == "__main__":
    main()
