import dataset
import modules
import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def load_data():
    dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_test')
    data = dataset.HipMRIDataset(dir, early_stop=False)
    return DataLoader(data, batch_size=32, shuffle=True)


def load_vae(args, device):
    vae = modules.VAE().to(device)
    weights = torch.load(os.path.join(args.path, 'VAE.pth'), weights_only=True)
    vae.load_state_dict(weights)
    return vae
    
    
def load_vqvae(args, device):
    vae = modules.VQVAE().to(device)
    weights = torch.load(os.path.join(args.path, 'VQVAE.pth'), weights_only=True)
    vae.load_state_dict(weights)
    return vae

    
def load_pixelcnn(args, device):
    vae = modules.PixelCNN().to(device)
    weights = torch.load(os.path.join(args.path, 'pixelCNN.pth'), weights_only=True)
    vae.load_state_dict(weights)
    return vae

    
def vae_recon_gen(vae, test_loader, device, samples=8):
    vae.eval()
    os.makedirs('eval', exist_ok=True)
    path = os.path.join('eval', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    with torch.no_grad():
        images = next(iter(test_loader))[:samples].to(device)
    
        pred, _, _ = vae(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, os.path.join(path, 'recon.png'), nrow=samples)
        print("Average SSIM is", ssim(pred, images).item())
    
        generated = vae.sample(samples, device)
        save_image(generated, os.path.join(path, 'generated.png'), nrow=samples)

    print('Reconstructed and generated images saved in', path, 'directory')


def vqvae_recon_gen(vqvae, pixelcnn, test_loader, device, samples=8):
    vqvae.eval()
    pixelcnn.eval()
    os.makedirs('eval', exist_ok=True)
    path = os.path.join('eval', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    with torch.no_grad():
        images = next(iter(test_loader))[:samples].to(device)
    
        pred, _, _ = vqvae(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, os.path.join(path, 'recon.png'), nrow=samples)
        print("Average SSIM is", ssim(pred, images).item())
    
        generated = pixelcnn.sample(vqvae, samples, device)
        save_image(generated, os.path.join(path, 'generated.png'), nrow=samples)

    print('Reconstructed and generated images saved in', path, 'directory')


def eval_vae(args, test_loader, device):
    vae = load_vae(args,  device)
    vae_recon_gen(vae, test_loader, device)
    
    
def eval_vqvae(args, test_loader, device):
    vqvae = load_vqvae(args,  device)
    pixelcnn = load_pixelcnn(args,  device)
    vqvae_recon_gen(vqvae, pixelcnn, test_loader, device)


def main(args):
    device = get_device()
    test_loader = load_data()
    
    if args.model == 'vae':
        eval_vae(args, test_loader, device)
    elif args.model == 'vqvae':
        eval_vqvae(args, test_loader, device)
    else:
        raise ValueError("--model argument must either be 'vae' or 'vqvae'")


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    
    main(args)