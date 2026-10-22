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


vae_epochs = 30
vqvae_epochs = 30
pixelcnn_epochs = 30


class TrainingData():
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


def load_data(split, early_stop=False):
    if split == 'train':
        dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_train')
    elif split == 'test':
        dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_test')
    elif split == 'validate':
        dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_validate')
    else:
        raise ValueError('Input into load_data must either be "train", "test" or "validate"')
    data = dataset.HipMRIDataset(dir, early_stop=early_stop)
    return DataLoader(data, batch_size=32, shuffle=True)


def vae_loss(pred, target, mean, log_var, beta=1):
    mse = F.mse_loss(pred, target, reduction='sum') / pred.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp()) / pred.size(0)
    return mse + beta * kl


def validate(vae, val_loader, device):
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


def train_vae(vae, train_loader, val_loader, device):
    optimiser = optim.Adam(vae.parameters(), lr=0.0001)
    training_data = TrainingData()
    start = time.time()
    
    for epoch in range(vae_epochs):
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

        train_time = time.time()
        epoch_time = train_time - epoch_start
        ssim = validate(vae, val_loader, device)
        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            peak_vram_mb = peak_vram_mps / (1024 ** 2)
        else:
            peak_vram_mb = 0

        training_data.add(epoch, [epoch_time, ssim, peak_vram_mb])
        
        print('------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}:')
        print(f'Average loss is {running_loss / len(train_loader)}')
        print(f'Elapsed time for epoch is {epoch_time} seconds')
        print(f'Total elapsed time is {time.time() - start} seconds')
        print(f'SSIM is {ssim}')
        print(f'Peak VRAM is {peak_vram_mb} MB')

    return training_data


def train_vqvae(vqvae, train_loader, val_loader, device):
    optimiser = optim.Adam(vqvae.parameters(), lr=0.0001)
    training_data = TrainingData()
    start = time.time()
    
    for epoch in range(vqvae_epochs):
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

        train_time = time.time()
        epoch_time = train_time - epoch_start
        ssim = validate(vqvae, val_loader, device)
        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            peak_vram_mb = peak_vram_mps / (1024 ** 2)
        else:
            peak_vram_mb = 0

        training_data.add(epoch, [epoch_time, ssim, peak_vram_mb])
        
        print('------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}:')
        print(f'Average loss is {running_loss / len(train_loader)}')
        print(f'Elapsed time for epoch is {epoch_time} seconds')
        print(f'Total elapsed time is {time.time() - start} seconds')
        print(f'SSIM is {ssim}')
        print(f'Peak VRAM is {peak_vram_mb} MB')

    print('------------------------------------------------------------------------------')
    return training_data


def train_pixelCNN(pixelCNN, vqvae, train_loader, device):
    optimiser = optim.Adam(pixelCNN.parameters(), lr=0.0001)
    training_data = TrainingData()
    start = time.time()
    pixelCNN.train()
    
    for epoch in range(pixelcnn_epochs):
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
            optimiser.step()

            running_loss += loss.item()

        train_time = time.time()
        epoch_time = train_time - epoch_start
        if device.type == 'cuda':
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        elif device.type == 'mps':
            peak_vram_mb = peak_vram_mps / (1024 ** 2)
        else:
            peak_vram_mb = 0

        training_data.add(epoch, [epoch_time, peak_vram_mb])
        
        print('------------------------------------------------------------------------------')
        print(f'Epoch {epoch + 1}:')
        print(f'Average loss is {running_loss / len(train_loader)}')
        print(f'Elapsed time for epoch is {epoch_time} seconds')
        print(f'Total elapsed time is {time.time() - start} seconds')
        print(f'Peak VRAM is {peak_vram_mb} MB')

    return training_data


def save_vae(vae, training_data):
    os.makedirs('train', exist_ok=True)
    dir = os.path.join('train', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(dir)
    torch.save(vae.state_dict(), os.path.join(dir, 'VAE.pth'))
    
    with open(os.path.join(dir, 'training data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'epoch_time_secs', 'val_ssim', 'peak_vram'])
        
        for epoch in range(vae_epochs):
            writer.writerow([epoch] + training_data.get(epoch))

    print('VAE model and data saved in', dir, 'directory')


def save_vqvae(vqvae, training_data, time):
    os.makedirs('train', exist_ok=True)
    dir = os.path.join('train', time)
    os.makedirs(dir, exist_ok=True)
    torch.save(vqvae.state_dict(), os.path.join(dir, 'VQVAE.pth'))
    
    with open(os.path.join(dir, 'training data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'epoch_time_secs', 'val_ssim', 'peak_vram'])
        
        for epoch in range(vqvae_epochs):
            writer.writerow([epoch] + training_data.get(epoch))

    print('VQVAE model and data saved in', dir, 'directory')
    

def save_pixelCNN(pixelCNN, training_data, time):
    os.makedirs('train', exist_ok=True)
    dir = os.path.join('train', time)
    os.makedirs(dir, exist_ok=True)
    torch.save(pixelCNN.state_dict(), os.path.join(dir, 'pixelCNN.pth'))
    
    with open(os.path.join(dir, 'training data.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'epoch_time_secs', 'peak_vram'])
        
        for epoch in range(pixelcnn_epochs):
            writer.writerow([epoch] + training_data.get(epoch))

    print('PixelCNN model and data saved in', dir, 'directory')
    

def main(args):
    if args.model != 'vae' and args.model != 'vqvae':
        raise ValueError("--model argument must either be 'vae' or 'vqvae'")
        
    device = get_device()
    train_loader = load_data('train', early_stop=args.test)
    val_loader = load_data('validate', early_stop=args.test)
    
    if args.model == 'vae':
        vae = modules.VAE().to(device)
        training_data = train_vae(vae, train_loader, val_loader, device)
        save_vae(vae, training_data)
    elif args.model == 'vqvae':
        vqvae = modules.VQVAE().to(device)
        vqvae_training_data = train_vqvae(vqvae, train_loader, val_loader, device)

        print('VQ-VAE training is done. PixelCNN training commence')

        pixelCNN = modules.PixelCNN().to(device)
        pixelcnn_training_data = train_pixelCNN(pixelCNN, vqvae, train_loader, device)
        
        time = datetime.now().strftime('%H-%M-%S_%d-%m-%y')
        save_vqvae(vqvae, vqvae_training_data, time)
        save_pixelCNN(pixelCNN, pixelcnn_training_data, time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    main(args);
