import torch
import torch.nn as nn
import torch.nn.functional as F


class FedFedBetaVAEGenerator(nn.Module):
    """Image-space beta-VAE generator q(x); FedFed shares x - q(x)."""

    def __init__(self, in_channels, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.mu = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.logvar = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.last_kl = None

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h).clamp(min=-8.0, max=8.0)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        self.last_kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return recon


class FedFedAutoEncoderGenerator(nn.Module):
    """Deterministic image-space autoencoder q(x), used as a no-KL generator ablation."""

    def __init__(self, in_channels, latent_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.last_kl = None

    def forward(self, x):
        recon = self.decoder(self.encoder(x))
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        self.last_kl = None
        return recon


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.block(x), inplace=True)


class FedFedResNetGenerator(nn.Module):
    """ResNet-style image generator q(x), used to compare with beta-VAE."""

    def __init__(self, in_channels, latent_channels=64):
        super().__init__()
        width = max(int(latent_channels), 32)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            ResidualBlock(width),
            ResidualBlock(width),
            ResidualBlock(width),
            nn.Conv2d(width, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.last_kl = None

    def forward(self, x):
        self.last_kl = None
        return self.net(x)


def build_fedfed_generator(generator_type, in_channels, latent_channels=64):
    generator_type = str(generator_type or 'beta_vae').lower()
    if generator_type == 'beta_vae':
        return FedFedBetaVAEGenerator(in_channels, latent_channels=latent_channels)
    if generator_type == 'resnet':
        return FedFedResNetGenerator(in_channels, latent_channels=latent_channels)
    if generator_type == 'autoencoder':
        return FedFedAutoEncoderGenerator(in_channels, latent_channels=latent_channels)
    raise ValueError('Unsupported FedFed generator type: {}'.format(generator_type))


FedFedGenerator = FedFedBetaVAEGenerator
