import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.encoder = nn.Sequential(
            Encoder(1, 32),
            Encoder(32, 64),
            nn.Flatten()
        )

        self.latent = latent
        self.mean = nn.Linear(64 * 64 * 32, latent)
        self.log_var = nn.Linear(64 * 64 * 32, latent)
        self.fc = nn.Sequential(
            nn.Linear(latent, 64 * 64 * 32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            Decoder(64, 32),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)

        mean = self.mean(x)
        log_var = torch.clamp(self.log_var(x), min=-10, max=10)
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        
        x = mean + std * epsilon
        x = self.fc(x)
        x = x.view(-1, 64, 64, 32)
        
        x = self.decoder(x)

        return x, mean, log_var

    def sample(self, samples, device):
        with torch.no_grad():
            z = torch.randn(samples, self.latent, device=device)
            gen = self.fc(z)
            gen = gen.view(-1, 64, 64, 32)
            gen = self.decoder(gen)

            return gen


class VectorQuantiser(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook dictionary: K vectors of dimension 
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        # z_e shape: [B, D, H, W] -> convert to [B, H, W, D] -> [B*H*W, D]
        z_e_perm = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_perm.view(-1, self.embedding_dim)

        # Pairwise Euclidean Distance: ||z_e - e||^2 = ||z_e||^2 + ||e||^2 - 2 * (z_e . e^T)
        distances = (
            torch.sum(flat_z_e ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2.0 * torch.matmul(flat_z_e, self.embedding.weight.t())
        ) # [B*H*W, K]

        # 1. Quantization: nearest neighbor argmin inde
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z_e_perm.shape)

        # 2. VQ Loss = Codebook Loss + beta * Commitment Loss
        loss = F.mse_loss(z_q, z_e_perm.detach()) + self.commitment_cost * F.mse_loss(z_e_perm, z_q.detach())

        # 3. Straight-Through Estimator (STE)
        z_q = z_e_perm + (z_q - z_e_perm).detach()

        # Reshape back to [B, D, H, W]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        indices = encoding_indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])

        return z_q, loss, indices


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=128, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            Encoder(1, 64),
            Encoder(64, 128),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
        )

        self.vq = VectorQuantiser(num_embeddings, embedding_dim, commitment_cost)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            Decoder(128, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

    def decode_indices(self, indices):
        z_q = self.vq.embedding(indices) 
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)


class MaskedConv2d(nn.Module):
    def __init__(self):
        super().__init__()