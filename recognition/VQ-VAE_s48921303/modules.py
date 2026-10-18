import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            Encoder(1, 32),
            Encoder(32, 64),
            nn.Flatten()
        )

        self.latent = latent
        self.mean = nn.Linear(64 * 64 * 36, latent)
        self.log_var = nn.Linear(64 * 64 * 36, latent)
        self.fc = nn.Sequential(
            nn.Linear(latent, 64 * 64 * 36),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            Decoder(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.enc(x)

        mean = self.mean(x)
        log_var = torch.clamp(self.log_var(x), min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        
        x = mean + std * epsilon
        x = self.fc(x)
        x = x.view(-1, 64, 64, 36)
        
        x = self.dec(x)

        return x, mean, log_var

    def sample(self, samples, device):
        with torch.no_grad():
            z = torch.randn(samples, self.latent, device=device)
            gen = self.fc(z)
            gen = gen.view(-1, 64, 64, 36)
            gen = self.dec(gen)

            return gen
