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


epochs = 10


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


def load_data(split):
    if split == 'train':
        dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_train')
    elif split == 'test':
        dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_test')
    elif split == 'validate':
        dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_validate')
    else:
        raise ValueError('Input into load_data must either be "train", "test" or "validate"')
    data = dataset.HipMRIDataset(dir, early_stop=False)
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
    
    for epoch in range(epochs):
        vae.train()
        running_loss = 0
        peak_vram_mps = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
                
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
    
    for epoch in range(epochs):
        vqvae.train()
        running_loss = 0
        peak_vram_mps = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
                
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

    return training_data


def save(model, training_data, args):
    os.makedirs('train', exist_ok=True)
    dir = 'train/' + datetime.now().strftime('%H-%M-%S_%d-%m-%y')
    os.makedirs(dir)
    torch.save(model.state_dict(), dir + '/' + args.model + '.pth')
    
    with open(dir + '/training data.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'epoch_time_secs', 'val_ssim', 'peak_vram'])
        
        for epoch in range(epochs):
            writer.writerow([epoch] + training_data.get(epoch))

    print('Saved in', dir, 'directory')
    

def main(args):
    device = get_device()
    train_loader = load_data('train')
    val_loader = load_data('validate')
    if args.model == 'vae':
        model = modules.VAE().to(device)
        training_data = train_vae(model, train_loader, val_loader, device)
    elif args.model == 'vq-vae':
        model = modules.VQVAE().to(device)
        training_data = train_vqvae(model, train_loader, val_loader, device)
    else:
        raise ValueError("--model argument must either be 'vae' or 'vq-vae'")
    
    save(model, training_data, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    
    main(args);
