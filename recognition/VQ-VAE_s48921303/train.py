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


epochs = 3


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


def loss_fn(pred, target, mean, log_var, beta=10):
    mse = F.mse_loss(pred, target, reduction='sum') / pred.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp()) / pred.size(0)
    return mse + beta * kl


def test(vae, test_loader, device):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    ssim_total = 0
    slices = 0
    vae.eval()
    
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)

            pred, _, _ = vae(images)
            
            batch_size = images.size(0)
            ssim_total += ssim(pred, images).item() * batch_size
            slices += batch_size

    return ssim_total / slices


def train(vae, train_loader, test_loader, device):
    optimiser = optim.Adam(vae.parameters(), lr=0.0001)
    training_data = TrainingData()
    vae.train()
    start = time.time()
    
    for epoch in range(epochs):
        running_loss = 0
        for images in train_loader:
            images = images.to(device)

            optimiser.zero_grad()

            pred, mean, log_var = vae(images)
            loss = loss_fn(pred, images, mean, log_var)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        train_time = time.time()
        
        if epoch == 0:
            epoch_time = train_time - start
        else:
            epoch_time = train_time - prev_train_time

        print(f'Epoch {epoch + 1} has average loss of {running_loss / len(train_loader)} at elapsed time {train_time - start} secs with this epoch taking {epoch_time}')
        
        prev_train_time = train_time
        ssim = test(vae, test_loader, device)
        training_data.add(epoch, [epoch_time, ssim])

    return training_data


def save(vae, training_data):
    os.makedirs('train', exist_ok=True)
    dir = 'train/' + datetime.now().strftime('%T_%d-%m-%y')
    os.makedirs(dir)
    torch.save(vae.state_dict(), dir + '/vae' + '.pth')
    with open(dir + '/training data.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['epoch', 'epoch_time_secs', 'val_ssim'])
        for epoch in range(epochs):
            writer.writerow([epoch] + training_data.get(epoch))
    

def main():
    device = get_device()
    train_loader = load_data('train')
    vae = modules.VAE().to(device)
    
    test_loader = load_data('test')
    training_data = train(vae, train_loader, test_loader, device)
    
    save(vae, training_data)


if __name__ == "__main__":
    main();
