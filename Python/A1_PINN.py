import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pickle


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


def main():
    with open("./Data/flowReactorData.pkl", "rb") as f:
        data = pickle.load(f)

    timeVecIn = data["timeInVec"]
    timeVecOut = data["timeOutVec"]
    concVecIn = data["Cin"]
    concVecOut = data["Cout"]

    cSpatial = data["Cspatial"]

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(timeVecIn, concVecIn[0,:], label="C_A In")
    plt.plot(timeVecOut, concVecOut[0,:], label="C_A Out")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.title("Concentration Profiles")

    fig = plt.gcf()
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    x = np.arange(cSpatial.shape[1])   # 101 (space)

    X, T = np.meshgrid(x, timeVecOut)

    ax.plot_surface(X, T, cSpatial[0, :, :].T)

    ax.set_xlabel("Spatial Coordinate")
    ax.set_ylabel("Time")
    ax.set_zlabel("Concentration")
    ax.set_title("Spatial Concentration Profile")

    #plt.show()

    # Train a PINN that maps Cin -> C(x) across the reactor length.
    # The physics model is a fixed-flow advection/reaction balance:
    #   q * dC/dx + k * C = 0
    # so Cout is the last spatial value of the predicted profile.

    reactor_length = 5.0
    num_delays = 30
    flow_rate = float(data.get("flowRate", 1.0))
    reaction_rate = 0.25

    # Trainable physical parameters (will be registered on the model)
    use_trainable_params = True

    cin_history, cin_current, target_profile = build_training_tensors(
        data,
        reactor_length=reactor_length,
        num_delays=num_delays,
    )

    # Build previous-step history features (shifted by one timestep)
    zeros_row = torch.zeros(1, cin_history.shape[1], dtype=cin_history.dtype)
    cin_history_prev = torch.cat((zeros_row, cin_history[:-1, :]), dim=0)

    # time step for finite-difference temporal derivative
    dt = float(timeVecOut[1] - timeVecOut[0]) if len(timeVecOut) > 1 else 1.0

    pinn = PINN(input_dim=num_delays + 1)

    # attach trainable parameters to the model so they are included in pinn.parameters()
    if use_trainable_params:
        # initialize dispersion D to a very small value (e.g. 1e-3)
        init_D = 1e-3
        # we initialize the raw parameter so that softplus(raw) ~= init_D
        raw_init_D = np.log(np.expm1(init_D))
        pinn.D_param = nn.Parameter(torch.tensor(raw_init_D, dtype=torch.float32))
        pinn.k_param = nn.Parameter(torch.tensor(reaction_rate, dtype=torch.float32))
        # fix flow velocity q to 1.0 (do not register as a trainable parameter)
        # pinn.q_param intentionally not created so q remains fixed below

    optimizer = optim.Adam(pinn.parameters(), lr=0.001)
    num_epochs = int(10000)
    print(f"Training will run for {num_epochs} epochs (num_epochs variable).")

    # Load pinn if it exists
    pinn_loaded = False
    try:
        with open("./Data/pinn_model.pkl", "rb") as f:
            pinn_state_dict = pickle.load(f)
            pinn.load_state_dict(pinn_state_dict)
        pinn_loaded = True
        print("Loaded existing PINN model.")
    except (FileNotFoundError, RuntimeError):
        print("No compatible existing PINN model found. Training a new one.")


    if not pinn_loaded:
        print(f"Starting training for {num_epochs} epochs...")
        try:
            for epoch in range(num_epochs):
                optimizer.zero_grad()

                predicted_profile = pinn(cin_history)

                # data loss (profile supervision)
                data_loss = torch.mean((predicted_profile - target_profile) ** 2)

                # temporal finite-difference: predict at previous time using shifted history
                predicted_prev = pinn(cin_history_prev)
                dCdt = (predicted_profile - predicted_prev) / dt

                # spatial derivatives (finite differences)
                Nx = predicted_profile.shape[1]
                dx = reactor_length / Nx

                dCdx = torch.zeros_like(predicted_profile)
                # central differences for interior
                if Nx > 2:
                    dCdx[:, 1:-1] = (predicted_profile[:, 2:] - predicted_profile[:, :-2]) / (2.0 * dx)
                # inlet derivative using cin_current
                dCdx[:, 0] = (predicted_profile[:, 1] - cin_current[:, 0]) / dx
                # outlet derivative (backward)
                dCdx[:, -1] = (predicted_profile[:, -1] - predicted_profile[:, -2]) / dx

                d2Cdx2 = torch.zeros_like(predicted_profile)
                if Nx > 2:
                    d2Cdx2[:, 1:-1] = (predicted_profile[:, 2:] - 2 * predicted_profile[:, 1:-1] + predicted_profile[:, :-2]) / (dx ** 2)
                # left boundary second derivative (use cin_current at inlet)
                d2Cdx2[:, 0] = (predicted_profile[:, 1] - 2 * predicted_profile[:, 0] + cin_current[:, 0]) / (dx ** 2)
                # right boundary: one-sided approx
                d2Cdx2[:, -1] = (predicted_profile[:, -2] - 2 * predicted_profile[:, -1] + predicted_profile[:, -1]) / (dx ** 2)

                if use_trainable_params:
                    D = F.softplus(pinn.D_param)
                    k = F.softplus(pinn.k_param)
                    q = 1.0
                else:
                    D = float(data.get("dispersionCoefficient", 0.0))
                    q = flow_rate
                    k = reaction_rate

                residual = dCdt - (D * d2Cdx2 - q * dCdx - k * predicted_profile)

                # physics loss: temporal+spatial residual + inlet BC
                phys_loss = torch.mean(residual ** 2) + torch.mean((predicted_profile[:, 0] - cin_current[:, 0]) ** 2)

                loss = data_loss + 0.1 * phys_loss

                loss.backward()
                optimizer.step()

                if epoch % 200 == 0 or epoch == num_epochs - 1:
                    with torch.no_grad():
                        cout_pred = predicted_profile[:, -1]
                        cout_target = target_profile[:, -1]
                        cout_loss = torch.mean((cout_pred - cout_target) ** 2)
                    print(
                        f"Epoch {epoch:04d} | loss={loss.item():.6f} | data={data_loss.item():.6f} "
                        f"| phys={phys_loss.item():.6f} | cout={cout_loss.item():.6f}"
                    )

        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}")
        finally:
            with torch.no_grad():
                predicted_profile = pinn(cin_history)
            print(f"Training loop finished; last epoch index was {epoch}")

    with torch.no_grad():
        predicted_profile = pinn(cin_history)
        predicted_profile_np = predicted_profile.cpu().numpy()
        target_profile_np = target_profile.cpu().numpy()

    # extract predicted outlet
    real_cout = torch.tensor(concVecOut[0, :], dtype=torch.float32)
    predicted_cout = torch.tensor(predicted_profile_np[:, -1], dtype=torch.float32)
    cout_mse = torch.mean((predicted_cout - real_cout) ** 2).item()
    print(f"Outlet Cout MSE: {cout_mse:.6f}")

    # report learned physical params
    if use_trainable_params:
        D_learn = float(F.softplus(pinn.D_param).detach().cpu().numpy())
        k_learn = float(F.softplus(pinn.k_param).detach().cpu().numpy())
        # q may be fixed (no pinn.q_param) — default to 1.0 if missing
        if hasattr(pinn, "q_param"):
            q_learn = float(F.softplus(pinn.q_param).detach().cpu().numpy())
        else:
            q_learn = 1.0
        print(f"Learned params: D={D_learn:.6g}, k={k_learn:.6g}, q={q_learn:.6g}")
    else:
        D_learn = float(data.get("dispersionCoefficient", 0.0))
        k_learn = reaction_rate
        q_learn = flow_rate

    plt.figure(figsize=(12, 4))
    plt.plot(timeVecOut, real_cout.cpu().numpy(), label="Measured Cout")
    plt.plot(timeVecOut, predicted_cout.cpu().numpy(), label="Predicted Cout", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Outlet Concentration")
    plt.title(f"Measured vs Predicted Outlet Concentration — D={D_learn:.3g}, k={k_learn:.3g}, q={q_learn:.3g}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Store trained pinn
    with open("./Data/pinn_model.pkl", "wb") as f:
        pickle.dump(pinn.state_dict(), f)
    
    # Compare the predicted and real spatial profiles over time and space.
    time_axis = np.asarray(timeVecOut)
    space_axis = np.arange(1, cSpatial.shape[1])
    real_cspatial = cSpatial[0, 1:, :].T
    predicted_cspatial = predicted_profile_np
    diff_cspatial = predicted_cspatial - real_cspatial

    # Create a 3x2 grid: left column = spatial fields (Real / Pred / Diff),
    # right column = output concentration comparisons (time-series, parity, residuals)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)

    vmin = min(real_cspatial.min(), predicted_cspatial.min())
    vmax = max(real_cspatial.max(), predicted_cspatial.max())

    # Row 0: Real spatial field
    ax00 = axes[0, 0]
    mesh0 = ax00.pcolormesh(space_axis, time_axis, real_cspatial, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax00.set_title("Real Cspatial")
    ax00.set_xlabel("Spatial index")
    ax00.set_ylabel("Time")
    fig.colorbar(mesh0, ax=ax00)

    # Row 1: Predicted spatial field
    ax10 = axes[1, 0]
    mesh1 = ax10.pcolormesh(space_axis, time_axis, predicted_cspatial, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax10.set_title("Predicted Cspatial")
    ax10.set_xlabel("Spatial index")
    ax10.set_ylabel("Time")
    fig.colorbar(mesh1, ax=ax10)

    # Row 2: Difference
    ax20 = axes[2, 0]
    mesh2 = ax20.pcolormesh(space_axis, time_axis, diff_cspatial, shading="auto", cmap="coolwarm")
    ax20.set_title("Difference (Pred - Real)")
    ax20.set_xlabel("Spatial index")
    ax20.set_ylabel("Time")
    fig.colorbar(mesh2, ax=ax20)

    # Right column: output concentration comparisons
    # Top-right: time-series measured vs predicted Cout
    ax01 = axes[0, 1]
    ax01.plot(time_axis, real_cout.cpu().numpy(), label="Measured Cout")
    ax01.plot(time_axis, predicted_cout.cpu().numpy(), label="Predicted Cout", linestyle="--")
    ax01.set_title(f"Outlet Concentration (MSE={cout_mse:.3e})")
    ax01.set_xlabel("Time")
    ax01.set_ylabel("Concentration")
    ax01.legend()

    # Middle-right: parity scatter (predicted vs measured)
    ax11 = axes[1, 1]
    ax11.scatter(real_cout.cpu().numpy(), predicted_cout.cpu().numpy(), s=10)
    lims = [min(real_cout.min().item(), predicted_cout.min().item()), max(real_cout.max().item(), predicted_cout.max().item())]
    ax11.plot(lims, lims, color="k", linestyle="--", linewidth=1)
    ax11.set_title("Predicted vs Measured Cout")
    ax11.set_xlabel("Measured Cout")
    ax11.set_ylabel("Predicted Cout")

    # Bottom-right: residual histogram
    ax21 = axes[2, 1]
    residuals = (predicted_cout - real_cout).cpu().numpy()
    ax21.hist(residuals, bins=30, color="gray")
    ax21.set_title("Residuals of Cout (Pred - Meas)")
    ax21.set_xlabel("Residual")
    ax21.set_ylabel("Count")

    fig.suptitle("Real vs Predicted Cspatial and Outlet Comparisons")
    plt.tight_layout()
    plt.show()




class PINN(nn.Module):
    def __init__(self, input_dim=11):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 100)

    def forward(self, t):
        input_tensor = t

        out = torch.tanh(self.fc1(input_tensor))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == "__main__":
    main()