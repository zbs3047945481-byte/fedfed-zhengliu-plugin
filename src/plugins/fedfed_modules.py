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


class FedFedPaperResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class FedFedPaperBetaVAEGenerator(nn.Module):
    """Beta-VAE generator matching the public FedFed FL_CVAE_cifar backbone."""

    def __init__(self, in_channels, latent_channels=32, z_dim=2048):
        super().__init__()
        d = int(latent_channels)
        self.d = d
        self.f = 8
        self.z = int(z_dim)
        self.encoder_former = nn.Conv2d(
            in_channels, d // 2, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.encoder = nn.Sequential(
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            FedFedPaperResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            FedFedPaperResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            FedFedPaperResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            FedFedPaperResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2),
            nn.LeakyReLU(inplace=True),
        )
        self.decoder_last = nn.ConvTranspose2d(
            d // 2, in_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.xi_bn = nn.BatchNorm2d(in_channels)
        self.fc11 = nn.Linear(d * self.f ** 2, self.z)
        self.fc12 = nn.Linear(d * self.f ** 2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f ** 2)
        self.sigmoid = nn.Sigmoid()
        self.last_kl = None
        self.last_mu = None
        self.last_logvar = None

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.view(-1, self.d * self.f ** 2)
        return h, self.fc11(h_flat), self.fc12(h_flat)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return torch.randn_like(std).mul(std).add_(mu)
        return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        return torch.tanh(self.decoder(z))

    def forward(self, x):
        h0 = self.encoder_former(x)
        _, mu, logvar = self.encode(h0)
        z = self.reparameterize(mu, logvar)
        projected = self.fc21(z)
        recon = self.decoder_last(self.decode(projected))
        recon = self.sigmoid(self.xi_bn(recon))
        if recon.shape[-2:] != x.shape[-2:]:
            recon = F.interpolate(recon, size=x.shape[-2:], mode='bilinear', align_corners=False)
        self.last_mu = mu
        self.last_logvar = logvar
        self.last_kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        self.last_kl = self.last_kl / max(x.size(0) * x.size(1) * self.z, 1)
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


def build_fedfed_generator(generator_type, in_channels, latent_channels=64, z_dim=2048):
    generator_type = str(generator_type or 'beta_vae').lower()
    if generator_type == 'beta_vae':
        return FedFedBetaVAEGenerator(in_channels, latent_channels=latent_channels)
    if generator_type == 'paper_beta_vae':
        return FedFedPaperBetaVAEGenerator(in_channels, latent_channels=latent_channels, z_dim=z_dim)
    if generator_type == 'resnet':
        return FedFedResNetGenerator(in_channels, latent_channels=latent_channels)
    if generator_type == 'autoencoder':
        return FedFedAutoEncoderGenerator(in_channels, latent_channels=latent_channels)
    raise ValueError('Unsupported FedFed generator type: {}'.format(generator_type))


FedFedGenerator = FedFedBetaVAEGenerator
