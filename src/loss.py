import torch
import torch.nn as nn


class PhysicsInformedDEMLoss(nn.Module):
    """
    Composite Physics-Informed Loss Function operating in balanced multi-decade log10 space:
        L = L_log10_DEM(d_hat, d_true) + lambda_phys * L_CHIANTI_flux(R * d_hat * dlogT, g_obs)
    """

    def __init__(self, R_matrix, dlog_t=0.1, lambda_phys=0.35):
        super().__init__()
        self.dlog_t = dlog_t
        self.lambda_phys = lambda_phys
        self.huber = nn.SmoothL1Loss(beta=0.25)

        R_clean = torch.nan_to_num(R_matrix.float(), nan=0.0, posinf=0.0, neginf=0.0) * dlog_t
        self.register_buffer("R_kernel", R_clean)

    def forward_euv_model(self, dem_pred):
        # Scale by 1e-27 before einsum to keep float32 registers completely overflow-safe
        dem_scaled = torch.clamp(dem_pred * 1e-27, min=0.0, max=1e4)
        R_scaled = self.R_kernel * 1e27
        return torch.einsum("ik,bkhw->bihw", R_scaled, dem_scaled)

    def forward(self, dem_pred, dem_true, g_obs):
        log_pred = torch.log10(torch.clamp(dem_pred, min=1e22, max=1e31))
        log_true = torch.log10(torch.clamp(dem_true, min=1e22, max=1e31))
        loss_data = self.huber(log_pred, log_true)

        g_pred = self.forward_euv_model(dem_pred)
        g_pred_log = torch.log10(torch.clamp(g_pred, min=1.0, max=1e7))
        g_obs_log = torch.log10(torch.clamp(g_obs, min=1.0, max=1e7))
        loss_phys = self.huber(g_pred_log, g_obs_log)

        total_loss = loss_data + self.lambda_phys * loss_phys
        return total_loss, loss_data.detach(), loss_phys.detach()