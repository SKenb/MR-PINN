import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image
import numpy as np


torch.manual_seed(0)
np.random.seed(0)


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


def build_training_tensors(data, reactor_length=5.0, num_delays=10):
    concVecIn = data["Cin"]
    cSpatial = data["Cspatial"]

    n_samples = concVecIn.shape[1]
    n_space = cSpatial.shape[1] - 1

    cin_current = torch.tensor(concVecIn[0, :], dtype=torch.float32).view(n_samples, 1)
    cin_history = build_lagged_cin_features(concVecIn[0, :], num_delays=num_delays)
    target_profile = torch.tensor(cSpatial[0, 1:, :].T, dtype=torch.float32)

    return cin_history, cin_current, target_profile


def physics_loss(predicted_profile, cin_current, flow_rate, reaction_rate, reactor_length=5.0):
    dx = reactor_length / predicted_profile.shape[1]

    left_state = torch.cat((cin_current, predicted_profile[:, :-1]), dim=1)
    dcdx = (predicted_profile - left_state) / dx
    residual = flow_rate * dcdx + reaction_rate * predicted_profile

    inlet_residual = torch.mean((flow_rate * (predicted_profile[:, 0] - cin_current[:, 0]) / dx + reaction_rate * predicted_profile[:, 0]) ** 2)
    interior_residual = torch.mean(residual[:, 1:] ** 2)

    return inlet_residual + interior_residual


def select_anchor_indices(total_points, out_dim):
    if out_dim <= 1:
        return np.array([total_points - 1], dtype=int)
    return np.unique(np.round(np.linspace(0, total_points - 1, out_dim)).astype(int))


def reconstruct_full_profile(anchor_profile, anchor_indices, total_points, interpolation_method: str = "linear"):
    anchor_profile = np.asarray(anchor_profile, dtype=np.float32)
    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    reconstructed = np.zeros((anchor_profile.shape[0], total_points), dtype=np.float32)
    full_x = np.arange(total_points, dtype=np.float32)
    for row_idx in range(anchor_profile.shape[0]):
        row_vals = anchor_profile[row_idx, :]
        if interpolation_method == "nearest":
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


def anchor_mse(predicted, target):
    return torch.mean((predicted - target) ** 2)


def plot_stage_progress(
    stage_idx,
    out_dim,
    anchor_indices,
    anchor_pred,
    target_profile,
    full_pred_profile,
    real_profile,
    time_axis,
    stage_loss=None,
    best_loss=None,
    epoch=None,
    output_dir="./Data/rpinn_plots",
):
    stage_label = stage_idx + 1
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")

    rep_anchor = target_profile[:, anchor_indices]
    anchor_time, anchor_space = np.meshgrid(time_axis, anchor_indices, indexing="ij")
    full_time, full_space = np.meshgrid(time_axis, np.arange(real_profile.shape[1]), indexing="ij")

    ax0.plot_surface(anchor_space, anchor_time, rep_anchor, cmap="viridis", alpha=0.55, linewidth=0, antialiased=False)
    ax0.plot_surface(anchor_space, anchor_time, anchor_pred, cmap="plasma", alpha=0.45, linewidth=0, antialiased=False)
    ax0.set_title(f"Stage {stage_label}: Anchor fit ({out_dim} outputs)")
    ax0.set_xlabel("Spatial index")
    ax0.set_ylabel("Time")
    ax0.set_zlabel("Concentration")
    if stage_loss is not None:
        loss_text = f"last loss={stage_loss:.3e}"
        if best_loss is not None:
            loss_text += f"\nbest={best_loss:.3e}"
        if epoch is not None:
            loss_text += f"\nepoch={epoch}"
        ax0.text2D(0.02, 0.98, loss_text, transform=ax0.transAxes, va="top")

    ax1.plot_surface(full_space, full_time, real_profile, cmap="viridis", alpha=0.55, linewidth=0, antialiased=False)
    ax1.plot_surface(full_space, full_time, full_pred_profile, cmap="plasma", alpha=0.45, linewidth=0, antialiased=False)
    ax1.set_title(f"Stage {stage_label}: Interpolated full profile")
    ax1.set_xlabel("Spatial index")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Concentration")

    fig.suptitle(f"rPINN Stage {stage_label} Progress")
    plt.tight_layout()
    stage_path = os.path.join(output_dir, f"stage_{stage_label:02d}_{out_dim:03d}_anchors.png")
    fig.savefig(stage_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stage progress figure to {stage_path}")
    return stage_path


def plot_lifted_diagnostics(
    stage_idx,
    previous_full_profile,
    lifted_full_profile,
    real_profile,
    time_axis,
    output_dir="./Data/rpinn_plots",
):
    stage_label = stage_idx + 1
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")

    space_axis = np.arange(real_profile.shape[1])
    tt, xx = np.meshgrid(time_axis, space_axis, indexing="ij")

    surf0 = ax0.plot_surface(xx, tt, previous_full_profile, cmap="viridis", linewidth=0, antialiased=False)
    ax0.set_title(f"Stage {stage_label}: previous trained profile")
    ax0.set_xlabel("Spatial index")
    ax0.set_ylabel("Time")
    ax0.set_zlabel("Concentration")
    fig.colorbar(surf0, ax=ax0, shrink=0.6)

    surf1 = ax1.plot_surface(xx, tt, lifted_full_profile, cmap="plasma", linewidth=0, antialiased=False)
    ax1.set_title(f"Stage {stage_label}: lifted profile before retraining")
    ax1.set_xlabel("Spatial index")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Concentration")
    fig.colorbar(surf1, ax=ax1, shrink=0.6)

    surf2 = ax2.plot_surface(xx, tt, lifted_full_profile - previous_full_profile, cmap="coolwarm", linewidth=0, antialiased=False)
    ax2.set_title(f"Stage {stage_label}: lift delta (lifted - previous)")
    ax2.set_xlabel("Spatial index")
    ax2.set_ylabel("Time")
    ax2.set_zlabel("Delta concentration")
    fig.colorbar(surf2, ax=ax2, shrink=0.6)

    surf3 = ax3.plot_surface(xx, tt, lifted_full_profile - real_profile, cmap="magma", linewidth=0, antialiased=False)
    ax3.set_title(f"Stage {stage_label}: lifted error (lifted - real)")
    ax3.set_xlabel("Spatial index")
    ax3.set_ylabel("Time")
    ax3.set_zlabel("Error")
    fig.colorbar(surf3, ax=ax3, shrink=0.6)

    fig.suptitle(f"rPINN Stage {stage_label} Lift Diagnostics")
    plt.tight_layout()
    lifted_path = os.path.join(output_dir, f"stage_{stage_label:02d}_lifted_diagnostics.png")
    fig.savefig(lifted_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lifted diagnostics figure to {lifted_path}")
    return lifted_path


def plot_stage_difference(
    stage_idx,
    previous_full_profile,
    current_full_profile,
    real_profile,
    time_axis,
    output_dir="./Data/rpinn_plots",
):
    stage_label = stage_idx + 1
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    ax3 = fig.add_subplot(2, 2, 4, projection="3d")

    space_axis = np.arange(real_profile.shape[1])
    tt, xx = np.meshgrid(time_axis, space_axis, indexing="ij")

    surf0 = ax0.plot_surface(xx, tt, previous_full_profile, cmap="viridis", linewidth=0, antialiased=False)
    ax0.set_title(f"Stage {stage_label}: previous trained profile")
    ax0.set_xlabel("Spatial index")
    ax0.set_ylabel("Time")
    ax0.set_zlabel("Concentration")
    fig.colorbar(surf0, ax=ax0, shrink=0.6)

    surf1 = ax1.plot_surface(xx, tt, current_full_profile, cmap="plasma", linewidth=0, antialiased=False)
    ax1.set_title(f"Stage {stage_label}: current trained profile")
    ax1.set_xlabel("Spatial index")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Concentration")
    fig.colorbar(surf1, ax=ax1, shrink=0.6)

    surf2 = ax2.plot_surface(xx, tt, current_full_profile - previous_full_profile, cmap="coolwarm", linewidth=0, antialiased=False)
    ax2.set_title(f"Stage {stage_label}: stage delta (current - previous)")
    ax2.set_xlabel("Spatial index")
    ax2.set_ylabel("Time")
    ax2.set_zlabel("Delta concentration")
    fig.colorbar(surf2, ax=ax2, shrink=0.6)

    surf3 = ax3.plot_surface(xx, tt, current_full_profile - real_profile, cmap="magma", linewidth=0, antialiased=False)
    ax3.set_title(f"Stage {stage_label}: stage error (current - real)")
    ax3.set_xlabel("Spatial index")
    ax3.set_ylabel("Time")
    ax3.set_zlabel("Error")
    fig.colorbar(surf3, ax=ax3, shrink=0.6)

    fig.suptitle(f"rPINN Stage {stage_label} Version Difference")
    plt.tight_layout()
    diff_path = os.path.join(output_dir, f"stage_{stage_label:02d}_version_difference.png")
    fig.savefig(diff_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved version difference figure to {diff_path}")
    return diff_path


def plot_training_snapshot(
    stage_idx,
    epoch,
    full_pred_profile,
    time_axis,
    cin_current=None,
    cout_target=None,
    output_dir="./Data/rpinn_plots",
):
    stage_label = stage_idx + 1
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 6))
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2)

    space_axis = np.arange(full_pred_profile.shape[1])
    tt, xx = np.meshgrid(time_axis, space_axis, indexing="ij")

    surf0 = ax0.plot_surface(xx, tt, full_pred_profile, cmap="plasma", linewidth=0, antialiased=False)
    ax0.set_title(f"Stage {stage_label} Epoch {epoch}: predicted profile")
    ax0.set_xlabel("Spatial index")
    ax0.set_ylabel("Time")
    ax0.set_zlabel("Concentration")
    fig.colorbar(surf0, ax=ax0, shrink=0.6)

    outlet_pred = full_pred_profile[:, -1]
    ax1.plot(time_axis, outlet_pred, label="Predicted outlet")
    ax1.plot(time_axis, cin_current.cpu().numpy().reshape(-1), label="Measured Cin", linestyle="--")
    ax1.plot(time_axis, cout_target.cpu().numpy().reshape(-1), label="Measured Cout", linestyle="--")
    ax1.set_title(f"Stage {stage_label} Epoch {epoch}: boundary traces")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Concentration")
    ax1.legend()

    fig.suptitle(f"rPINN Stage {stage_label} Training Snapshot")
    plt.tight_layout()
    snapshot_path = os.path.join(output_dir, f"stage_{stage_label:02d}_epoch_{epoch:04d}.png")
    fig.savefig(snapshot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return snapshot_path


def save_stage_artifacts(output_dir, stage_idx, anchor_indices, anchor_pred, target_profile, time_axis):
    stage_label = stage_idx + 1
    artifact_path = os.path.join(output_dir, f"stage_{stage_label:02d}_anchor_artifacts.npz")
    np.savez(
        artifact_path,
        anchor_indices=np.asarray(anchor_indices, dtype=np.int64),
        anchor_pred=np.asarray(anchor_pred, dtype=np.float32),
        target_profile=np.asarray(target_profile, dtype=np.float32),
        time_axis=np.asarray(time_axis, dtype=np.float32),
    )
    print(f"Saved stage artifacts to {artifact_path}")
    return artifact_path


def save_gif(image_paths, gif_path, duration=1200):
    if not image_paths:
        return None
    frames = [Image.open(path).convert("RGB") for path in image_paths]
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    for frame in frames:
        frame.close()
    print(f"Saved animation to {gif_path}")
    return gif_path


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


PINN = RPinN


def main():
    print("Starting rPINN run...", flush=True)
    with open("./Data/flowReactorData.pkl", "rb") as f:
        data = pickle.load(f)
    print("Loaded flow reactor data.", flush=True)

    time_vec_out = data["timeOutVec"]
    c_spatial = data["Cspatial"]
    c_out = data["Cout"]
    cin_series = data["Cin"]

    reactor_length = 5.0
    num_delays = 30
    cin_history, _, target_profile = build_training_tensors(
        data,
        reactor_length=reactor_length,
        num_delays=num_delays,
    )

    n_samples = cin_series.shape[1]
    cin_current = torch.tensor(cin_series[0, :], dtype=torch.float32).view(n_samples, 1)
    cout_target = torch.tensor(c_out[0, :], dtype=torch.float32).view(n_samples, 1)

    total_points = target_profile.shape[1]
    schedule = [4, 20, 70, 100]
    stage_epochs = [300, 500, 900, 1200]
    stage_thresholds = [5e-4, 2e-4, 1e-4, 5e-5]
    time_axis = np.asarray(time_vec_out)
    real_profile = c_spatial[0, 1:, :].T
    dt = float(time_vec_out[1] - time_vec_out[0]) if len(time_vec_out) > 1 else 1.0
    cin_history_prev = torch.cat((torch.zeros(1, cin_history.shape[1], dtype=cin_history.dtype), cin_history[:-1, :]), dim=0)

    model = RPinN(input_dim=cin_history.shape[1], output_dim=schedule[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    cin_history = cin_history.to(device)
    cin_current = cin_current.to(device)
    cout_target = cout_target.to(device)
    cin_history_prev = cin_history_prev.to(device)
    target_profile = target_profile.to(device)

    use_trainable_params = True
    if use_trainable_params:
        init_D = 1e-3
        raw_init_D = np.log(np.expm1(init_D))
        model.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32))
        model.k_param = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))

    checkpoint_path = "./Data/rpinn_model.pkl"
    plot_dir = "./Data/rpinn_plots"
    os.makedirs(plot_dir, exist_ok=True)
    stage_image_paths = []
    overall_training_image_paths = []
    previous_full_pred_profile = None
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as f:
                loaded_state = pickle.load(f)
            model.load_state_dict(loaded_state)
            print("Loaded existing rPINN checkpoint.", flush=True)
        except Exception:
            print("Existing rPINN checkpoint was not compatible; training from scratch.", flush=True)
    else:
        print("No existing checkpoint found; training from scratch.", flush=True)

    for stage_idx, out_dim in enumerate(schedule):
        if stage_idx > 0:
            print(f"Expanding output layer {schedule[stage_idx - 1]} -> {out_dim}")
            model.expand_output(out_dim)
            model = model.to(device)

            # Diagnose the lifted model before retraining on the new output size.
            with torch.no_grad():
                lifted_anchor_pred = model(cin_history).cpu().numpy()
            lifted_full_pred = reconstruct_full_profile(lifted_anchor_pred, select_anchor_indices(total_points, out_dim), total_points)
            plot_lifted_diagnostics(
                stage_idx=stage_idx,
                previous_full_profile=previous_full_pred_profile,
                lifted_full_profile=lifted_full_pred,
                real_profile=real_profile,
                time_axis=time_axis,
                output_dir=plot_dir,
            )

        anchor_indices = select_anchor_indices(total_points, out_dim)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        print(f"Stage {stage_idx + 1}/{len(schedule)} | out_dim={out_dim} | boundary-supervised", flush=True)

        best_loss = float("inf")
        patience = 60
        stale = 0
        stage_training_image_paths = []
        snapshot_interval = max(1, stage_epochs[stage_idx] // 10)

        for epoch in range(stage_epochs[stage_idx]):
            model.train()
            optimizer.zero_grad()
            predicted_profile = model(cin_history)
            predicted_prev = model(cin_history_prev)

            # boundary supervision only: inlet Cin and outlet Cout
            boundary_loss = torch.mean((predicted_profile[:, 0] - cin_current[:, 0]) ** 2) + torch.mean((predicted_profile[:, -1] - cout_target[:, 0]) ** 2)

            # time-dependent physics residual from the predicted profile only
            dCdt = (predicted_profile - predicted_prev) / dt
            Nx = predicted_profile.shape[1]
            dx = reactor_length / (Nx - 1) if Nx > 1 else reactor_length

            dCdx = torch.zeros_like(predicted_profile)
            d2Cdx2 = torch.zeros_like(predicted_profile)
            if Nx > 2:
                dCdx[:, 1:-1] = (predicted_profile[:, 2:] - predicted_profile[:, :-2]) / (2.0 * dx)
                d2Cdx2[:, 1:-1] = (predicted_profile[:, 2:] - 2.0 * predicted_profile[:, 1:-1] + predicted_profile[:, :-2]) / (dx ** 2)
            dCdx[:, 0] = (predicted_profile[:, 1] - predicted_profile[:, 0]) / dx
            dCdx[:, -1] = (predicted_profile[:, -1] - predicted_profile[:, -2]) / dx
            d2Cdx2[:, 0] = (predicted_profile[:, 2] - 2.0 * predicted_profile[:, 1] + predicted_profile[:, 0]) / (dx ** 2) if Nx > 2 else 0.0
            d2Cdx2[:, -1] = (predicted_profile[:, -1] - 2.0 * predicted_profile[:, -2] + predicted_profile[:, -3]) / (dx ** 2) if Nx > 2 else 0.0

            if use_trainable_params:
                D = F.softplus(model.D_param)
                k = F.softplus(model.k_param)
            else:
                D = float(data.get("dispersionCoefficient", 1e-3))
                k = 0.25
            q = 1.0
            residual = dCdt - (D * d2Cdx2 - q * dCdx - k * predicted_profile)
            phys_loss = torch.mean(residual ** 2)

            loss = boundary_loss + 0.1 * phys_loss
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            if loss_value + 1e-12 < best_loss:
                best_loss = loss_value
                stale = 0
            else:
                stale += 1

            if epoch % max(1, stage_epochs[stage_idx] // 5) == 0 or epoch == stage_epochs[stage_idx] - 1:
                print(f"  epoch {epoch:04d} | boundary+phys_loss={loss_value:.6e} | best={best_loss:.6e}", flush=True)

            if best_loss < stage_thresholds[stage_idx] and epoch > 20:
                print(f"  good fit reached at epoch {epoch}; expanding next stage", flush=True)
                break
            if stale >= patience:
                print(f"  plateau detected at epoch {epoch}; expanding next stage", flush=True)
                break

            if epoch % snapshot_interval == 0 or epoch == stage_epochs[stage_idx] - 1:
                with torch.no_grad():
                    snapshot_full_pred = predicted_profile.detach().cpu().numpy()
                snapshot_path = plot_training_snapshot(
                    stage_idx=stage_idx,
                    epoch=epoch,
                    full_pred_profile=snapshot_full_pred,
                    time_axis=time_axis,
                    cin_current=cin_current,
                    cout_target=cout_target,
                    output_dir=plot_dir,
                )
                stage_training_image_paths.append(snapshot_path)
                overall_training_image_paths.append(snapshot_path)

        with torch.no_grad():
            stage_anchor_pred = model(cin_history).cpu().numpy()
        stage_full_pred = reconstruct_full_profile(stage_anchor_pred, anchor_indices, total_points)
        if previous_full_pred_profile is not None:
            plot_stage_difference(
                stage_idx=stage_idx,
                previous_full_profile=previous_full_pred_profile,
                current_full_profile=stage_full_pred,
                real_profile=real_profile,
                time_axis=time_axis,
                output_dir=plot_dir,
            )
        stage_path = plot_stage_progress(
            stage_idx=stage_idx,
            out_dim=out_dim,
            anchor_indices=anchor_indices,
            anchor_pred=stage_anchor_pred,
            target_profile=target_profile.cpu().numpy(),
            full_pred_profile=stage_full_pred,
            real_profile=real_profile,
            time_axis=time_axis,
            stage_loss=loss_value,
            best_loss=best_loss,
            epoch=epoch,
            output_dir=plot_dir,
        )
        stage_image_paths.append(stage_path)
        previous_full_pred_profile = stage_full_pred
        save_stage_artifacts(
            output_dir=plot_dir,
            stage_idx=stage_idx,
            anchor_indices=anchor_indices,
            anchor_pred=stage_anchor_pred,
            target_profile=target_profile.cpu().numpy(),
            time_axis=time_axis,
        )

        stage_gif_path = os.path.join(plot_dir, f"stage_{stage_idx + 1:02d}_training.gif")
        save_gif(stage_training_image_paths, stage_gif_path, duration=900)

    with torch.no_grad():
        final_anchor_pred = model(cin_history).cpu().numpy()

    full_pred_profile = reconstruct_full_profile(
        final_anchor_pred,
        select_anchor_indices(total_points, model.fc3.out_features),
        total_points,
    )
    real_profile = c_spatial[0, 1:, :].T
    real_cout = c_out[0, :]
    pred_cout = full_pred_profile[:, -1]
    cout_mse = float(np.mean((pred_cout - real_cout) ** 2))
    print(f"Final outlet Cout MSE: {cout_mse:.6e}")

    time_axis = np.asarray(time_vec_out)
    space_axis = np.arange(1, c_spatial.shape[1])
    diff_profile = full_pred_profile - real_profile

    # Save outlet concentration comparison.
    fig_out, ax_out = plt.subplots(figsize=(12, 4))
    ax_out.plot(time_axis, real_cout, label="Measured Cout")
    ax_out.plot(time_axis, pred_cout, label="Predicted Cout", linestyle="--")
    ax_out.set_title(f"Outlet Concentration (MSE={cout_mse:.3e})")
    ax_out.set_xlabel("Time")
    ax_out.set_ylabel("Concentration")
    ax_out.legend()
    fig_out.tight_layout()
    outlet_plot_path = os.path.join(plot_dir, "final_outlet_comparison.png")
    fig_out.savefig(outlet_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig_out)
    print(f"Saved outlet comparison figure to {outlet_plot_path}")

    # Final comparison surface triptych.
    fig = plt.figure(figsize=(18, 6))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")
    tt, xx = np.meshgrid(time_axis, space_axis, indexing="ij")

    surf0 = ax0.plot_surface(xx, tt, real_profile, cmap="viridis", linewidth=0, antialiased=False)
    ax0.set_title("Real Cspatial")
    ax0.set_xlabel("Spatial index")
    ax0.set_ylabel("Time")
    ax0.set_zlabel("Concentration")
    fig.colorbar(surf0, ax=ax0, shrink=0.6)

    surf1 = ax1.plot_surface(xx, tt, full_pred_profile, cmap="plasma", linewidth=0, antialiased=False)
    ax1.set_title(f"rPINN Predicted Cspatial ({model.fc3.out_features} anchors)")
    ax1.set_xlabel("Spatial index")
    ax1.set_ylabel("Time")
    ax1.set_zlabel("Concentration")
    fig.colorbar(surf1, ax=ax1, shrink=0.6)

    surf2 = ax2.plot_surface(xx, tt, diff_profile, cmap="coolwarm", linewidth=0, antialiased=False)
    ax2.set_title("Difference (Pred - Real)")
    ax2.set_xlabel("Spatial index")
    ax2.set_ylabel("Time")
    ax2.set_zlabel("Concentration")
    fig.colorbar(surf2, ax=ax2, shrink=0.6)

    fig.suptitle("rPINN Spatial Surface Comparison")
    plt.tight_layout()
    final_plot_path = os.path.join(plot_dir, "final_surface_comparison.png")
    fig.savefig(final_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved final surface comparison figure to {final_plot_path}")

    if stage_image_paths:
        gif_path = os.path.join(plot_dir, "stage_progress.gif")
        save_gif(stage_image_paths, gif_path, duration=1200)

    if overall_training_image_paths:
        overall_gif_path = os.path.join(plot_dir, "training_progress.gif")
        save_gif(overall_training_image_paths, overall_gif_path, duration=800)


if __name__ == "__main__":
    main()