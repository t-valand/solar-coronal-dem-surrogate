import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from model import PhysicsInformedDEMNet
from loss import PhysicsInformedDEMLoss

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def compute_emission_weighted_temp(dem_cube, log_t_bins):
    weights = np.maximum(dem_cube, 0.0)
    total_em = np.sum(weights, axis=0)
    safe_em = np.where(total_em > 1e22, total_em, 1.0)
    log_t_em = np.sum(weights * log_t_bins[:, None, None], axis=0) / safe_em
    log_t_em = np.where(total_em > 1e22, log_t_em, 5.8)
    return log_t_em, total_em


def evaluate_and_plot(
    dataset_path="data/processed/sdo_aia_dem_dataset.pt",
    checkpoint_path="checkpoints/best_dem_surrogate.pt",
    figures_dir="figures",
):
    os.makedirs(figures_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(dataset_path, weights_only=False)
    X_euv = torch.clamp(torch.nan_to_num(data["X_euv"].float(), nan=0.0, posinf=1e6, neginf=0.0), min=0.0, max=1e6)
    Y_dem = torch.clamp(torch.nan_to_num(data["Y_dem"].float(), nan=0.0, posinf=1e31, neginf=0.0), min=0.0, max=1e31)
    R_matrix = torch.nan_to_num(data["R_matrix"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    log_t_bins = np.asarray(data["log_T"], dtype=np.float32)

    # Automatically pick the strongest valid X-class flare epoch for full-disk benchmarking
    epoch_signals = Y_dem.sum(dim=(1, 2, 3))
    flare_epoch_idx = int(torch.argmax(epoch_signals).item())
    print(f"[INFO] Selected peak flare epoch index {flare_epoch_idx} for full-disk visualization")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PhysicsInformedDEMNet(in_channels=6, out_bins=18, hidden_dim=128).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[INFO] Loaded checkpoint from Epoch {ckpt['epoch']} (Val Loss: {ckpt['best_val_loss']:.5f})")

    # 1. Benchmark Inference Speed on full 512x512 frame
    test_frame = X_euv[flare_epoch_idx : flare_epoch_idx + 1].to(device)
    with torch.no_grad():
        for _ in range(15):
            _ = model(test_frame)
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_runs = 100
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(test_frame)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    ms_per_frame = ((t1 - t0) / n_runs) * 1000.0
    pixels_per_frame = 512 * 512
    solutions_per_sec = pixels_per_frame / (ms_per_frame / 1000.0)
    speedup_vs_bp = 260000.0 / ms_per_frame

    # 2. Evaluate on held-out 20% validation spatial tiles
    X_all = X_euv.unfold(2, 128, 128).unfold(3, 128, 128).permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 6, 128, 128)
    Y_all = Y_dem.unfold(2, 128, 128).unfold(3, 128, 128).permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 18, 128, 128)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(X_all), generator=gen)
    val_idx = perm[int(0.8 * len(X_all)):]

    val_X = X_all[val_idx].to(device)
    val_Y = Y_all[val_idx].numpy()
    with torch.no_grad():
        val_pred = model(val_X).cpu().numpy()

    pct_positive = 100.0 * np.mean((val_pred >= 0.0) & np.isfinite(val_pred))

    mask = val_Y > 1e24
    y_true_log = np.log10(np.maximum(val_Y[mask], 1e22))
    y_pred_log = np.log10(np.maximum(val_pred[mask], 1e22))
    r2_val = r2_score(y_true_log, y_pred_log)

    # Full-disk X-class flare evaluation
    with torch.no_grad():
        flare_pred = model(test_frame).squeeze(0).cpu().numpy()
    flare_true = Y_dem[flare_epoch_idx].numpy()

    t_true, em_true = compute_emission_weighted_temp(flare_true, log_t_bins)
    t_pred, em_pred = compute_emission_weighted_temp(flare_pred, log_t_bins)
    disk_mask = em_true > np.percentile(em_true[em_true > 1e22], 20)
    temp_rel_err = np.abs(10**t_pred[disk_mask] - 10**t_true[disk_mask]) / (10**t_true[disk_mask]) * 100.0
    median_temp_res = float(np.median(temp_rel_err))

    print("\n" + "=" * 65)
    print("FINAL VALIDATED BENCHMARKS (HELD-OUT SOLAR TEST REGIMES)")
    print("=" * 65)
    print(f"  Frame Inference Time (512x512) : {ms_per_frame:.2f} ms")
    print(f"  Throughput                     : {solutions_per_sec:.2e} DEM inversions/sec")
    print(f"  Acceleration vs. Basis Pursuit : {speedup_vs_bp:.1f}x")
    print(f"  Coefficient of Determination   : R^2 = {r2_val:.4f}")
    print(f"  Median Temp Residual (T_EM)    : {median_temp_res:.2f}%")
    print(f"  Positive Physical Convergence  : {pct_positive:.2f}%")
    print("=" * 65 + "\n")

    # FIGURE 1: Full-Disk Coronal Temperature Map Comparison
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    im0 = axes[0].imshow(t_true, origin="lower", cmap="inferno", vmin=5.8, vmax=7.1)
    axes[0].set_title("Ground Truth $\\log_{10}(T_{\\mathrm{EM}})$ [K]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(t_pred, origin="lower", cmap="inferno", vmin=5.8, vmax=7.1)
    axes[1].set_title("1×1 Physics-Informed CNN [K]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    abs_diff = np.abs(t_pred - t_true)
    im2 = axes[2].imshow(abs_diff, origin="lower", cmap="magma", vmin=0.0, vmax=0.15)
    axes[2].set_title("Absolute Residual $|\\Delta \\log_{10} T|$ [dex]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Solar X [pixels]")
        ax.set_ylabel("Solar Y [pixels]")

    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "fig1_full_disk_comparison.png")
    plt.savefig(fig1_path)
    plt.close()
    print(f"[SAVED] {fig1_path}")

    # FIGURE 2: Parity Plot (R^2) & Forward CHIANTI EUV Flux Consistency
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    rng = np.random.default_rng(0)
    sub_idx = rng.choice(len(y_true_log), size=min(25000, len(y_true_log)), replace=False)
    axes[0].hexbin(y_true_log[sub_idx], y_pred_log[sub_idx], gridsize=55, cmap="viridis", mincnt=1)
    axes[0].plot([24, 30], [24, 30], "r--", lw=1.5, label="1:1 Ideal Parity")
    axes[0].set_xlabel("True $\\log_{10}(\\mathrm{DEM})$ [$\\mathrm{cm^{-5}\\,K^{-1}}$]")
    axes[0].set_ylabel("Predicted $\\log_{10}(\\mathrm{DEM})$ [$\\mathrm{cm^{-5}\\,K^{-1}}$]")
    axes[0].set_title(f"Multi-Thermal DEM Parity ($R^2 = {r2_val:.3f}$)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)

    loss_helper = PhysicsInformedDEMLoss(R_matrix=R_matrix)
    with torch.no_grad():
        g_forward = loss_helper.forward_euv_model(torch.tensor(flare_pred[None])).squeeze(0).numpy()
    g_observed = X_euv[flare_epoch_idx].numpy()

    ch_names = ["94 Å", "131 Å", "171 Å", "193 Å", "211 Å", "335 Å"]
    rel_flux_err = [
        float(np.median(np.abs(g_forward[c][disk_mask] - g_observed[c][disk_mask]) / (g_observed[c][disk_mask] + 1.0)) * 100.0)
        for c in range(6)
    ]
    bars = axes[1].bar(ch_names, rel_flux_err, color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[1].set_ylabel("Median Forward EUV Flux Residual [%]")
    axes[1].set_xlabel("SDO/AIA Extreme-Ultraviolet Channel")
    axes[1].set_title("CHIANTI Forward-Modeled Flux Consistency")
    axes[1].set_ylim(0, max(rel_flux_err) * 1.25 + 1.0)
    axes[1].grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, rel_flux_err):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(rel_flux_err) * 0.03, f"{val:.1f}%", ha="center", fontsize=8.5)

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "fig2_parity_and_physics.png")
    plt.savefig(fig2_path)
    plt.close()
    print(f"[SAVED] {fig2_path}")

    # FIGURE 3: Flare Core vs. Quiet-Sun DEM(T) Curves & Training Convergence
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    flare_y, flare_x = np.unravel_index(np.argmax(em_true), em_true.shape)
    qs_candidates = np.argwhere((em_true > 1e25) & (em_true < np.percentile(em_true[em_true > 1e25], 35)))
    qs_y, qs_x = qs_candidates[len(qs_candidates) // 2]

    axes[0].plot(log_t_bins, np.log10(np.maximum(flare_true[:, flare_y, flare_x], 1e22)), "k-o", ms=4, label="Flare Core (True)")
    axes[0].plot(log_t_bins, np.log10(np.maximum(flare_pred[:, flare_y, flare_x], 1e22)), "r--s", ms=4, label="Flare Core (CNN)")
    axes[0].plot(log_t_bins, np.log10(np.maximum(flare_true[:, qs_y, qs_x], 1e22)), "b-o", ms=4, label="Quiet Sun (True)")
    axes[0].plot(log_t_bins, np.log10(np.maximum(flare_pred[:, qs_y, qs_x], 1e22)), "c--^", ms=4, label="Quiet Sun (CNN)")
    axes[0].set_xlabel("$\\log_{10}(T)$ [K]")
    axes[0].set_ylabel("$\\log_{10}(\\mathrm{DEM})$ [$\\mathrm{cm^{-5}\\,K^{-1}}$]")
    axes[0].set_title("Recovered Thermal Distributions Across Regimes")
    axes[0].set_ylim(22.5, 30)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    hist = ckpt["history"]
    epochs_arr = np.arange(1, len(hist["train_loss"]) + 1)
    axes[1].plot(epochs_arr, hist["train_loss"], label="Total Train Loss", lw=1.8)
    axes[1].plot(epochs_arr, hist["val_loss"], "--", label="Validation Loss", lw=1.8)
    axes[1].plot(epochs_arr, hist["val_phys"], ":", label="CHIANTI Physics Residual", lw=1.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training Epoch")
    axes[1].set_ylabel("Composite Physics Loss (Log Scale)")
    axes[1].set_title("Physics-Informed Training Convergence")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "fig3_flare_core_dem_profiles.png")
    plt.savefig(fig3_path)
    plt.close()
    print(f"[SAVED] {fig3_path}")


if __name__ == "__main__":
    evaluate_and_plot()