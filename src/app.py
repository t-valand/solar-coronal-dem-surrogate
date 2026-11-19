import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from model import PhysicsInformedDEMNet
from loss import PhysicsInformedDEMLoss

# 1. Load trained physics-informed surrogate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysicsInformedDEMNet(in_channels=6, out_bins=18, hidden_dim=128).to(device)
ckpt_path = os.path.join(ROOT_DIR, "checkpoints", "best_dem_surrogate.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 2. Load calibrated dataset & CHIANTI physics matrix
data_path = os.path.join(ROOT_DIR, "data", "processed", "sdo_aia_dem_dataset.pt")
data = torch.load(data_path, weights_only=False)
X_euv = torch.clamp(torch.nan_to_num(data["X_euv"].float(), nan=0.0, posinf=1e6, neginf=0.0), min=0.0, max=1e6)
Y_dem = torch.clamp(torch.nan_to_num(data["Y_dem"].float(), nan=0.0, posinf=1e31, neginf=0.0), min=0.0, max=1e31)
R_matrix = torch.nan_to_num(data["R_matrix"].float(), nan=0.0, posinf=0.0, neginf=0.0)
log_t_bins = np.asarray(data["log_T"], dtype=np.float32)
loss_helper = PhysicsInformedDEMLoss(R_matrix=R_matrix)

# 3. Extract real calibrated pixel presets from the peak flare epoch
flare_epoch = int(torch.argmax(Y_dem.sum(dim=(1, 2, 3))).item())
em_map = Y_dem[flare_epoch].sum(dim=0).numpy()
fy, fx = np.unravel_index(np.argmax(em_map), em_map.shape)

valid_mask = em_map > 1e25
qs_pts = np.argwhere(valid_mask & (em_map < np.percentile(em_map[valid_mask], 30)))
qy, qx = qs_pts[len(qs_pts) // 2]

ar_pts = np.argwhere(
    valid_mask
    & (em_map > np.percentile(em_map[valid_mask], 70))
    & (em_map < np.percentile(em_map[valid_mask], 95))
)
ay, ax = ar_pts[len(ar_pts) // 2]

qs_vals = [round(float(v), 2) for v in X_euv[flare_epoch, :, qy, qx]]
ar_vals = [round(float(v), 2) for v in X_euv[flare_epoch, :, ay, ax]]
fl_vals = [round(float(v), 2) for v in X_euv[flare_epoch, :, fy, fx]]


def run_all_diagnostics(ch94, ch131, ch171, ch193, ch211, ch335):
    vals = [float(ch94), float(ch131), float(ch171), float(ch193), float(ch211), float(ch335)]
    euv_tensor = torch.tensor(
        [[[[vals[0]]], [[vals[1]]], [[vals[2]]], [[vals[3]]], [[vals[4]]], [[vals[5]]]]],
        dtype=torch.float32,
        device=device,
    )

    # Single-pixel inference + full-disc 512x512 inference
    frame = X_euv[flare_epoch : flare_epoch + 1].to(device)
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dem_pred = model(euv_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        dem_disc = model(frame).squeeze(0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()

        g_recon = loss_helper.forward_euv_model(dem_pred.cpu()).squeeze().numpy()

    pixel_ms = (t1 - t0) * 1000.0
    disc_ms = (t3 - t2) * 1000.0

    dem_vals = dem_pred.squeeze().cpu().numpy()
    log_dem_vals = np.log10(np.maximum(dem_vals, 1e22))
    total_em = float(np.sum(dem_vals))
    log_t_em = float(np.sum(dem_vals * log_t_bins) / (total_em + 1e-12))
    temp_mk = (10.0 ** log_t_em) / 1e6

    # Graph 1: 18-Bin Thermal Distribution Curve
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.8), dpi=140)
    ax1.plot(log_t_bins, log_dem_vals, "r-o", lw=2.2, ms=5, label="Predicted DEM(T)")
    ax1.fill_between(log_t_bins, 22.0, log_dem_vals, color="orangered", alpha=0.18)
    ax1.axvline(log_t_em, color="navy", linestyle="--", lw=1.6, label=f"Mean Temp: {temp_mk:.2f} MK")
    ax1.set_xlabel("Logarithmic Plasma Temperature log10(T) [K]")
    ax1.set_ylabel("log10(DEM) [cm^-5 K^-1]")
    ax1.set_title("Graph 1: 18-Bin Reconstructed Thermal Profile")
    ax1.set_xlim(5.5, 7.2)
    ax1.set_ylim(22.0, 30.5)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    plt.tight_layout()

    # Graph 2: CHIANTI Forward Radiative Flux Consistency
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8), dpi=140)
    ch_names = ["94 Å", "131 Å", "171 Å", "193 Å", "211 Å", "335 Å"]
    x = np.arange(len(ch_names))
    ax2.bar(x - 0.18, np.maximum(vals, 1e-2), width=0.35, label="Queried EUV Input", color="#1f77b4")
    ax2.bar(x + 0.18, np.maximum(g_recon, 1e-2), width=0.35, label="CHIANTI Reconstructed", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ch_names)
    ax2.set_yscale("log")
    ax2.set_ylabel("Calibrated EUV Intensity (Log Scale)")
    ax2.set_title("Graph 2: CHIANTI Forward Flux Consistency")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    plt.tight_layout()

    # Graph 3: Full-Disc 512x512 Solar Temperature Map
    weights = np.maximum(dem_disc, 0.0)
    tot_disc_em = np.sum(weights, axis=0)
    log_t_map = np.sum(weights * log_t_bins[:, None, None], axis=0) / np.where(tot_disc_em > 1e22, tot_disc_em, 1.0)
    log_t_map = np.where(tot_disc_em > 1e22, log_t_map, 5.8)

    fig3, ax3 = plt.subplots(figsize=(5.5, 3.8), dpi=140)
    im = ax3.imshow(log_t_map, origin="lower", cmap="inferno", vmin=5.8, vmax=7.1)
    ax3.set_title(f"Graph 3: Full-Disc 512×512 Temp Map ({disc_ms:.1f} ms)")
    ax3.set_xlabel("Solar X [pixels]")
    ax3.set_ylabel("Solar Y [pixels]")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="log10(T_EM) [K]")
    plt.tight_layout()

    summary = (
        f"### Real-Time Coronal Plasma Diagnostics\n"
        f"- **Mean Coronal Temperature (T_EM):** `{temp_mk:.2f} Million K` (`log10 T = {log_t_em:.3f}`)\n"
        f"- **Total Emission Measure:** `{total_em:.2e} cm⁻⁵` | "
        f"**Single-Pixel Time:** `{pixel_ms:.2f} ms` | **Full-Disc (262,144 px) Time:** `{disc_ms:.1f} ms` on `{str(device).upper()}`"
    )
    return summary, fig1, fig2, fig3


with gr.Blocks(title="Real-Time Solar Coronal DEM Surrogate") as demo:
    gr.Markdown("# Real-Time Physics-Informed Solar Coronal DEM Surrogate (SDO/AIA)")
    gr.Markdown("Enter any 6 SDO/AIA EUV channel intensities below (recommended range `0.1` to `1,000`) or click a preset row:")

    with gr.Row():
        c94 = gr.Number(label="AIA 94 Å", value=fl_vals[0])
        c131 = gr.Number(label="AIA 131 Å", value=fl_vals[1])
        c171 = gr.Number(label="AIA 171 Å", value=fl_vals[2])
        c193 = gr.Number(label="AIA 193 Å", value=fl_vals[3])
        c211 = gr.Number(label="AIA 211 Å", value=fl_vals[4])
        c335 = gr.Number(label="AIA 335 Å", value=fl_vals[5])

    btn = gr.Button("Run Live DEM Inversion & Generate All 3 Graphs", variant="primary")

    gr.Examples(
        examples=[qs_vals, ar_vals, fl_vals],
        inputs=[c94, c131, c171, c193, c211, c335],
        label="Click a Preset Row to Auto-Fill: Row 1 = Quiet Sun | Row 2 = Active Region | Row 3 = X-Class Flare Core",
    )

    out_md = gr.Markdown()
    with gr.Row():
        out_g1 = gr.Plot(label="Graph 1: 18-Bin DEM Curve")
        out_g2 = gr.Plot(label="Graph 2: CHIANTI Forward Flux Consistency")
        out_g3 = gr.Plot(label="Graph 3: Full-Disc 512×512 Temperature Map")

    btn.click(
        run_all_diagnostics,
        inputs=[c94, c131, c171, c193, c211, c335],
        outputs=[out_md, out_g1, out_g2, out_g3],
    )

if __name__ == "__main__":
    demo.launch()