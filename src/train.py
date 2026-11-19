import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import PhysicsInformedDEMNet
from loss import PhysicsInformedDEMLoss


def extract_patches(X_frames, Y_frames, patch_size=128):
    N, C_in, H, W = X_frames.shape
    _, C_out, _, _ = Y_frames.shape
    X_patches = X_frames.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    X_patches = X_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C_in, patch_size, patch_size)

    Y_patches = Y_frames.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    Y_patches = Y_patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, C_out, patch_size, patch_size)
    return X_patches, Y_patches


def train_model(
    dataset_path="data/processed/sdo_aia_dem_dataset.pt",
    checkpoint_path="checkpoints/best_dem_surrogate.pt",
    epochs=60,
    batch_size=8,
    lr=2e-3,
    lambda_phys=0.35,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    data = torch.load(dataset_path, weights_only=False)
    X_euv = torch.clamp(torch.nan_to_num(data["X_euv"].float(), nan=0.0, posinf=1e6, neginf=0.0), min=0.0, max=1e6)
    Y_dem = torch.clamp(torch.nan_to_num(data["Y_dem"].float(), nan=0.0, posinf=1e31, neginf=0.0), min=0.0, max=1e31)
    R_matrix = torch.nan_to_num(data["R_matrix"].float(), nan=0.0, posinf=0.0, neginf=0.0)

    # Extract 160 spatial tiles (128x128) across all 10 solar epochs and use a clean 80/20 split
    X_all, Y_all = extract_patches(X_euv, Y_dem, patch_size=128)
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(X_all), generator=gen)
    n_train = int(0.8 * len(X_all))

    train_ds = TensorDataset(X_all[perm[:n_train]], Y_all[perm[:n_train]])
    val_ds = TensorDataset(X_all[perm[n_train:]], Y_all[perm[n_train:]])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = PhysicsInformedDEMNet(in_channels=6, out_bins=18, hidden_dim=128).to(device)
    criterion = PhysicsInformedDEMLoss(R_matrix=R_matrix, dlog_t=0.1, lambda_phys=lambda_phys).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_phys": []}

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for g_batch, d_batch in train_loader:
            g_batch = g_batch.to(device, non_blocking=True)
            d_batch = d_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            d_pred = model(g_batch)
            loss, _, _ = criterion(d_pred, d_batch, g_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * g_batch.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        val_phys = 0.0
        with torch.no_grad():
            for g_batch, d_batch in val_loader:
                g_batch = g_batch.to(device, non_blocking=True)
                d_batch = d_batch.to(device, non_blocking=True)
                d_pred = model(g_batch)
                loss, _, l_phys = criterion(d_pred, d_batch, g_batch)
                val_loss += loss.item() * g_batch.size(0)
                val_phys += l_phys.item() * g_batch.size(0)

        val_loss /= len(val_ds)
        val_phys /= len(val_ds)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_phys"].append(val_phys)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                checkpoint_path,
            )

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"Phys Residual: {val_phys:.5f} | Elapsed: {elapsed:.1f}s"
            )

    print(f"\n[COMPLETE] Best validation loss: {best_val_loss:.5f} saved to {checkpoint_path}")


if __name__ == "__main__":
    train_model()