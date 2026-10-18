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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def load_data():
    dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_train')
    data = dataset.HipMRIDataset(dir, early_stop=False)
    return DataLoader(data, batch_size=32, shuffle=True)


def loss_fn(pred, target, mean, log_var, beta=10):
    mse = F.mse_loss(pred, target, reduction='sum') / pred.size(0)
    kl = -0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp()) / pred.size(0)
    return mse + beta * kl


def train(vae, train_loader, device):
    optimiser = optim.Adam(vae.parameters(), lr=0.0001)
    epochs = 30
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

        print(f'epoch {epoch + 1} has avg loss {running_loss / len(train_loader)} at elapsed time {time.time() - start} secs')


def save_model(vae):
    os.makedirs('models', exist_ok=True)
    torch.save(vae.state_dict(), 'models/vae: ' + datetime.now().strftime('%T_%d-%m-%y') + '.pth')
    

def main():
    device = get_device()
    train_loader = load_data()
    vae = modules.VAE().to(device)
    train(vae, train_loader, device)
    save_model(vae)


if __name__ == "__main__":
    main();
