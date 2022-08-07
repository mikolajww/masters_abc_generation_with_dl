import torch
import torch.nn as nn


class VAEMuLogVar(nn.Module):

    def __init__(
            self,
            in_features,
            latent_dim
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.mu = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2 * latent_dim),
            nn.SELU(),
            nn.Linear(in_features=2 * latent_dim, out_features=latent_dim)
        )

        self.logvar = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=2 * latent_dim),
            nn.SELU(),
            nn.Linear(in_features=2 * latent_dim, out_features=latent_dim)
        )

    def forward(self, X, device):

        batch_size = X.size(0)

        mu = self.mu(X)
        logvar = self.logvar(X)

        std_dev = torch.exp(0.5 * logvar)
        z = mu + torch.randn((batch_size, self.latent_dim), device=device) * std_dev
        return z, mu, logvar