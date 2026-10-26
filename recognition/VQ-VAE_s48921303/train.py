import dataset
import modules
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure
import csv
import argparse
import numpy as np
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingData:
    def __init__(self):
        self.data = {}

    def add(self, epoch, values):
        self.data[epoch] = values

    def get(self, epoch):
        return self.data[epoch]


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def load_data(split, data_dir, batch_size=32, early_stop=False):
    if split == 'train':
        dir = os.path.expanduser(os.path.join(data_dir, 'keras_slices_train'))
    elif split == 'test':
        dir = os.path.expanduser(os.path.join(data_dir, 'keras_slices_test'))
    elif split == 'validate':
        dir = os.path.expanduser(os.path.join(data_dir, 'keras_slices_validate'))
    else:
        raise ValueError('Input into load_data must either be "train", "test" or "validate"')
    
    augment = (split == 'train')
    data = dataset.HipMRIDataset(dir, early_stop=early_stop, augment=augment)
    return DataLoader(data, batch_size=batch_size, shuffle=(split == 'train'))


def plot_metric(epochs, values, title, ylabel, save_path):
    plt.figure()
    plt.plot(epochs, values)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


def vae_loss(pred, target, mean, log_var, beta=1):
    mse = F.mse_loss(pred, target, reduction='sum') / pred.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp()) / pred.size(0)
    return mse + beta * kl


def validate_vae(vae, val_loader, device):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_total = 0
    slices = 0
    vae.eval()
    
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            pred, _, _ = vae(images)
            
            batch_size = images.size(0)
            ssim_total += ssim(pred, images).item() * batch_size
            slices += batch_size

    return ssim_total / slices


def validate_vqvae(vqvae, val_loader, device, num_embeddings=128):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_total = 0
    slices = 0
    num_embeddings = vqvae.vq.num_embeddings
    total_token_counts = torch.zeros(num_embeddings, device=device)
    vqvae.eval()
    
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            pred, _, indices = vqvae(images)
            
            batch_size = images.size(0)
            ssim_total += ssim(pred, images).item() * batch_size
            slices += batch_size
            
            total_token_counts += torch.bincount(indices.flatten(), minlength=num_embeddings).float()

    probs = total_token_counts / total_token_counts.sum()
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
    normalised_perplexity = (torch.exp(entropy) / num_embeddings).item()

    return (ssim_total / slices), normalised_perplexity


def train_vae(vae, train_loader, val_loader, device, epochs, lr, patience=7):
    optimiser = optim.Adam(vae.parameters(), lr=lr)
    training_data = TrainingData()
    start = time.time()
    
    best_ssim = -1.0
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        vae.train()
        running_loss = 0
        peak_vram_mps = 0
        epoch_start = time.time()
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
                
        for images in train_loader:
            images = images.to(device)

            optimiser.zero_grad()

            pred, mean, log_var = vae(images)
            loss = vae_loss(pred, images, mean, log_var)

            if device.type == 'mps':
                alloc = torch.mps.current_allocated_memory()
                if alloc > peak_vram_mps:
                    peak_vram_mps = alloc
                
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        ssim = validate_vae(vae, val_loader, device)

        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            peak_vram_mb = peak_vram_mps / (1024 ** 2)
        else:
            peak_vram_mb = 0

        training_data.add(epoch, [avg_loss, epoch_time, ssim, peak_vram_mb])
        
        print('------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Average loss: {avg_loss:.4f} | Validation SSIM: {ssim:.4f}')
        print(f'Elapsed time for epoch: {epoch_time:.2f}s | Total: {time.time() - start:.2f}s')
        print(f'Peak VRAM: {peak_vram_mb:.2f} MB')

        if ssim > best_ssim:
            best_ssim = ssim
            best_weights = {k: v.cpu() for k, v in vae.state_dict().items()}
            patience_counter = 0
            print(f"--> Achieved new best Val SSIM: {best_ssim:.4f} (Weights cached)")
        else:
            patience_counter += 1
            print(f"--> No SSIM improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_weights is not None:
        vae.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        print(f"Restored best VAE model with Val SSIM: {best_ssim:.4f}")

    print('------------------------------------------------------------------------------')
    return training_data


def train_vqvae(vqvae, train_loader, val_loader, device, epochs, lr, patience=7):
    optimiser = optim.Adam(vqvae.parameters(), lr=lr)
    training_data = TrainingData()
    start = time.time()
    
    best_ssim = -1.0
    patience_counter = 0
    best_weights = None
    
    for epoch in range(epochs):
        vqvae.train()
        running_loss = 0
        peak_vram_mps = 0
        epoch_start = time.time()
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
                
        for images in train_loader:
            images = images.to(device)

            optimiser.zero_grad()

            pred, vq_loss, _ = vqvae(images)
            loss = F.mse_loss(pred, images) + vq_loss

            if device.type == 'mps':
                alloc = torch.mps.current_allocated_memory()
                if alloc > peak_vram_mps:
                    peak_vram_mps = alloc
                
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        ssim, norm_perplexity = validate_vqvae(vqvae, val_loader, device)

        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            peak_vram_mb = peak_vram_mps / (1024 ** 2)
        else:
            peak_vram_mb = 0

        training_data.add(epoch, [avg_loss, epoch_time, ssim, norm_perplexity, peak_vram_mb])
        
        print('------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Average loss: {avg_loss:.4f} | Validation SSIM: {ssim:.4f} | Perplexity: {norm_perplexity:.4f}')

        if ssim > best_ssim:
            best_ssim = ssim
            best_weights = {k: v.cpu() for k, v in vqvae.state_dict().items()}
            patience_counter = 0
            print(f"--> Achieved new best Val SSIM: {best_ssim:.4f} (Weights cached)")
        else:
            patience_counter += 1
            print(f"--> No SSIM improvement for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_weights is not None:
        vqvae.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        print(f"Restored best VQ-VAE model with Val SSIM: {best_ssim:.4f}")

    print('------------------------------------------------------------------------------')
    return training_data


def train_pixelCNN(pixelCNN, vqvae, train_loader, device, epochs, lr):
    optimiser = optim.Adam(pixelCNN.parameters(), lr=lr)
    training_data = TrainingData()
    start = time.time()
    pixelCNN.train()
    vqvae.eval()
    
    for epoch in range(epochs):
        running_loss = 0
        peak_vram_mps = 0
        epoch_start = time.time()
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
                
        for images in train_loader:
            images = images.to(device)
            with torch.no_grad():
                tokens = vqvae.get_codebook_indices(images)

            optimiser.zero_grad()
            
            logits = pixelCNN(tokens)
            loss = F.cross_entropy(logits, tokens)

            if device.type == 'mps':
                alloc = torch.mps.current_allocated_memory()
                if alloc > peak_vram_mps:
                    peak_vram_mps = alloc
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pixelCNN.parameters(), max_norm=1.0)
            optimiser.step()

            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)

        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            peak_vram_mb = peak_vram_mps / (1024 ** 2)
        else:
            peak_vram_mb = 0

        training_data.add(epoch, [avg_loss, epoch_time, peak_vram_mb])
        
        print('------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Average loss: {avg_loss:.4f}')
        print(f'Elapsed time for epoch: {epoch_time:.2f}s | Total: {time.time() - start:.2f}s')
        print(f'Peak VRAM: {peak_vram_mb:.2f} MB')

    return training_data


def save_vae(vae, training_data, epochs):
    os.makedirs('train', exist_ok=True)
    save_dir = os.path.join('train', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(save_dir, exist_ok=True)
    torch.save(vae.state_dict(), os.path.join(save_dir, 'VAE.pth'))
    
    actual_epochs = len(training_data.data)
    
    loss_vals, ssim_vals = [], []
    with open(os.path.join(save_dir, 'training_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'loss', 'epoch_time_secs', 'val_ssim', 'peak_vram'])
        
        for epoch in range(actual_epochs):
            row = training_data.get(epoch)
            loss_vals.append(row[0])
            ssim_vals.append(row[2])
            writer.writerow([epoch] + row)

    plot_metric(range(1, actual_epochs + 1), loss_vals, 'VAE Training Loss', 'Loss (MSE + KL)', os.path.join(save_dir, 'loss_curve.png'))
    plot_metric(range(1, actual_epochs + 1), ssim_vals, 'VAE Validation SSIM', 'SSIM', os.path.join(save_dir, 'ssim_curve.png'))
    print('VAE model, curves, and data saved in', save_dir, 'directory')


def save_vqvae(vqvae, training_data, epochs, timestamp):
    os.makedirs('train', exist_ok=True)
    save_dir = os.path.join('train', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(vqvae.state_dict(), os.path.join(save_dir, 'VQVAE.pth'))
    
    # Use actual completed epochs in case early stopping triggered
    actual_epochs = len(training_data.data)
    
    loss_vals, ssim_vals, perp_vals = [], [], []
    with open(os.path.join(save_dir, 'vqvae_training_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'loss', 'epoch_time_secs', 'val_ssim', 'norm_perplexity', 'peak_vram'])
        
        for epoch in range(actual_epochs):
            row = training_data.get(epoch)
            loss_vals.append(row[0])
            ssim_vals.append(row[2])
            perp_vals.append(row[3])
            writer.writerow([epoch] + row)

    plot_metric(range(1, actual_epochs + 1), loss_vals, 'VQ-VAE Training Loss', 'Loss (Recon + VQ)', os.path.join(save_dir, 'loss_curve.png'))
    plot_metric(range(1, actual_epochs + 1), ssim_vals, 'VQ-VAE Validation SSIM', 'SSIM', os.path.join(save_dir, 'ssim_curve.png'))
    plot_metric(range(1, actual_epochs + 1), perp_vals, 'Codebook Utilisation', 'Normalised Perplexity', os.path.join(save_dir, 'perplexity_curve.png'))
    print('VQ-VAE model, curves, and data saved in', save_dir, 'directory')
    

def save_pixelCNN(pixelCNN, training_data, epochs, timestamp):
    os.makedirs('train', exist_ok=True)
    save_dir = os.path.join('train', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(pixelCNN.state_dict(), os.path.join(save_dir, 'pixelCNN.pth'))
    
    loss_vals = []
    with open(os.path.join(save_dir, 'pixelcnn_training_data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'loss', 'epoch_time_secs', 'peak_vram'])
        
        for epoch in range(epochs):
            row = training_data.get(epoch)
            loss_vals.append(row[0])
            writer.writerow([epoch] + row)

    plot_metric(range(1, epochs + 1), loss_vals, 'PixelCNN Training Loss', 'Cross-Entropy Loss', os.path.join(save_dir, 'pixelcnn_loss_curve.png'))
    print('PixelCNN model, curves, and data saved in', save_dir, 'directory')
    

def main(args):
    if args.model not in ['vae', 'vqvae']:
        raise ValueError("--model argument must either be 'vae' or 'vqvae'")
        
    set_seed(args.seed)
    device = get_device()
    train_loader = load_data('train', data_dir=args.data_dir, batch_size=args.batch_size, early_stop=args.test)
    val_loader = load_data('validate', data_dir=args.data_dir, batch_size=args.batch_size, early_stop=args.test)
    
    if args.model == 'vae':
        vae = modules.VAE(embedding_dim=args.embedding_dim).to(device)
        training_data = train_vae(vae, train_loader, val_loader, device, epochs=args.vae_epochs, lr=args.lr)
        save_vae(vae, training_data, args.vae_epochs)

    elif args.model == 'vqvae':
        vqvae = modules.VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim).to(device)
        vqvae_training_data = train_vqvae(vqvae, train_loader, val_loader, device, epochs=args.vqvae_epochs, lr=args.lr)

        print('VQ-VAE training is done. PixelCNN training commence')

        pixelCNN = modules.PixelCNN(num_embeddings=args.num_embeddings).to(device)
        pixelcnn_training_data = train_pixelCNN(pixelCNN, vqvae, train_loader, device, epochs=args.pixelcnn_epochs, lr=args.lr)
        
        timestamp = datetime.now().strftime('%H-%M-%S_%d-%m-%y')
        save_vqvae(vqvae, vqvae_training_data, args.vqvae_epochs, timestamp)
        save_pixelCNN(pixelCNN, pixelcnn_training_data, args.pixelcnn_epochs, timestamp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['vae', 'vqvae'], help="Model type: 'vae' or 'vqvae'")
    parser.add_argument('--data_dir', required=True, type=str, help="Path to HipMRI root directory")
    parser.add_argument('--vae_epochs', type=int, default=30, help="Number of VAE training epochs")
    parser.add_argument('--vqvae_epochs', type=int, default=30, help="Number of VQ-VAE training epochs")
    parser.add_argument('--pixelcnn_epochs', type=int, default=60, help="Number of PixelCNN training epochs")
    parser.add_argument('--batch_size', type=int, default=64, help="DataLoader batch size")
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001, help="Adam optimizer learning rate")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--test', action='store_true', help="Early stopping flag for smoke-testing scripts")
    args = parser.parse_args()
    
    main(args)