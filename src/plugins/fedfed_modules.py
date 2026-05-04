import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Beta-VAE generator q(x), aligned with FedFed's FL_CVAE_cifar."""

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
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
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


def build_fedfed_generator(generator_type, in_channels, latent_channels=32, z_dim=2048):
    generator_type = str(generator_type or 'paper_beta_vae').lower()
    if generator_type != 'paper_beta_vae':
        raise ValueError('Only paper_beta_vae is supported in the FedFed plugin.')
    return FedFedPaperBetaVAEGenerator(
        in_channels,
        latent_channels=latent_channels,
        z_dim=z_dim,
    )
