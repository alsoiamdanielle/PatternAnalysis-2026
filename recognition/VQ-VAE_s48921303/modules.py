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
    def __init__(self, latent=64, embedding_dim=64):
        super().__init__()
        self.latent = latent
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            Encoder(1, 64),
            Encoder(64, 128),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.Flatten()
        )

        self.mean = nn.Linear(embedding_dim * 64 * 32, latent)
        self.log_var = nn.Linear(embedding_dim * 64 * 32, latent)
        self.fc = nn.Sequential(
            nn.Linear(latent, embedding_dim * 64 * 32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            Decoder(128, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
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
        x = x.view(-1, self.embedding_dim, 64, 32)
        
        x = self.decoder(x)

        return x, mean, log_var

    def sample(self, samples, device):
        with torch.no_grad():
            z = torch.randn(samples, self.latent, device=device)
            gen = self.fc(z)
            gen = gen.view(-1, self.embedding_dim, 64, 32)
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

        # 1. Quantisation: nearest neighbor argmin inde
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
    def __init__(self, num_embeddings=64, embedding_dim=64, commitment_cost=0.25):
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

    def get_codebook_indices(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices


class MaskedConv2d(nn.Module):
    def __init__(self, type, in_channels, out_channels, kernel_size=5):
        super().__init__()
        centre = kernel_size // 2
        
        if type == 'V':
            mask = torch.ones(kernel_size, kernel_size)
            mask[centre + 1:, :] = 0
            k_size = (kernel_size, kernel_size)
            padding = (centre, centre)
        elif type in ('A', 'B'):
            mask = torch.ones(1, kernel_size)
            if type == 'A':
                mask[0, centre:] = 0
            else:
                mask[0, centre + 1:] = 0
            k_size = (1, kernel_size)
            padding = (0, centre)
        else:
            raise ValueError('type parameter for MaskedConv2d must either be "V", "A" or "B"')
            
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=padding)

    def forward(self, x):
        return F.conv2d(x, self.conv2d.weight * self.mask, self.conv2d.bias, padding=self.conv2d.padding)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, type, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.v_conv = MaskedConv2d('V', in_channels, out_channels * 2, kernel_size=kernel_size)
        self.h_conv = MaskedConv2d(type, in_channels, out_channels * 2, kernel_size=kernel_size)
        self.v_to_h = nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=1)
        self.h_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, v, h, pos):
        v_feat = self.v_conv(v)
        v_shift = F.pad(v_feat[:, :, :-1, :], (0, 0, 1, 0))
        h_feat = self.h_conv(h) + self.v_to_h(v_shift) + pos

        v_f, v_g = torch.chunk(v_feat, 2, dim=1)
        v_out = torch.tanh(v_f) * torch.sigmoid(v_g)

        h_f, h_g = torch.chunk(h_feat, 2, dim=1)
        h_out = self.h_out(torch.tanh(h_f) * torch.sigmoid(h_g))

        return v_out, h_out


class MaskedConv2dLayer(nn.Module):
    def __init__(self, type, in_channel, out_channel, kernel_size=5):
        super().__init__()
        self.type = type
        self.layer = GatedMaskedConv2d(type, in_channel, out_channel, kernel_size=kernel_size)

    def forward(self, x):
        v, h, pos = x
        v_out, h_out = self.layer(v, h, pos)
        if self.type == 'A':
            return v_out, h_out, pos
        else:
            return v + v_out, h + h_out, pos


class PixelCNN(nn.Module):
    def __init__(self, num_embeddings=128, hidden_dim=64, B_layers=8, latent_h=64, latent_w=32):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.hidden_dim = hidden_dim
        self.latent_h = latent_h
        self.latent_w = latent_w
        
        self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        self.row_embed = nn.Parameter(torch.randn(1, hidden_dim * 2, latent_h, 1) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, hidden_dim * 2, 1, latent_w) * 0.02)
        
        self.net = nn.Sequential(
            MaskedConv2dLayer('A', hidden_dim, hidden_dim, kernel_size=5),
            *[MaskedConv2dLayer('B', hidden_dim, hidden_dim, kernel_size=5) for _ in range(B_layers)]
        )
        self.out_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, num_embeddings, kernel_size=1)
        )

    def forward(self, x):
        b, h, w = x.shape
        embeddings = self.embedding(x.long()).permute(0, 3, 1, 2).contiguous()
        pos = self.row_embed[:, :, :h, :] + self.col_embed[:, :, :, :w]
        
        _, features, _ = self.net((embeddings, embeddings, pos))
        return self.out_conv(features)
        
    def sample(self, vqvae, samples, device, temperature=0.9, top_k=0):
        self.eval()
        vqvae.eval()
        
        with torch.no_grad():
            tokens = torch.zeros(samples, self.latent_h, self.latent_w, device=device, dtype=torch.long)
            for r in range(self.latent_h):
                for c in range(self.latent_w):
                    logits = self.forward(tokens[:, :r + 1, :])
                    logits_rc = logits[:, :, r, c] / temperature

                    if top_k > 0:
                        v, _ = torch.topk(logits_rc, min(top_k, logits_rc.size(-1)), dim=-1)
                        logits_rc[logits_rc < v[:, [-1]]] = -float('inf')
                    
                    probs = F.softmax(logits_rc, dim=1)
                    sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
                    
                    tokens[:, r, c] = sampled_tokens

        return vqvae.decode_indices(tokens)