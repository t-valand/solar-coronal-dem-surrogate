import torch
import torch.nn as nn


class PhysicsInformedDEMNet(nn.Module):
    """
    Physics-Informed 1x1 2D Convolutional Neural Network Surrogate for real-time
    Solar Coronal Differential Emission Measure (DEM) inversion.
    """

    def __init__(self, in_channels=6, out_bins=18, hidden_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.out_bins = out_bins

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.GELU(),

            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, out_bins, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.05, inplace=True)
        )

    def forward_log10(self, g_euv):
        x_norm = torch.log1p(torch.clamp(g_euv, min=0.0)) / 12.0
        raw_out = self.net(x_norm)
        return torch.clamp(22.0 + raw_out, min=22.0, max=30.5)

    def forward(self, g_euv):
        log10_dem = self.forward_log10(g_euv)
        return torch.pow(10.0, log10_dem)