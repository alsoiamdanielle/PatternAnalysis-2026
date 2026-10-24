import dataset
import modules
import torch
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datetime import datetime
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def load_data(test):
    dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_test')
    data = dataset.HipMRIDataset(dir, early_stop=test)
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


def images_convert_fid(images):
  images = torch.clamp(images, 0.0, 1.0)
  if images.shape[1] == 1:
    images = images.repeat(1, 3, 1, 1)

  return (images * 255.0).to(torch.uint8)


def init_fid(test_loader, feature, device):
    fid = FrechetInceptionDistance(feature=feature).to(device)
    
    for images in test_loader:
        images = images.to(device)
        converted_images = images_convert_fid(images)
        fid.update(converted_images, real=True)
        
    return fid
    

def init_kid(test_loader, device):
    kid = KernelInceptionDistance(subset_size=50).to(device)
    
    for images in test_loader:
        images = images.to(device)
        converted_images = images_convert_fid(images)
        kid.update(converted_images, real=True)
        
    return kid
    
    
def vae_recon_gen(vae, test_loader, fid_2048, fid_192, kid, device, samples=12, fid_samples=200, batch_size=32):
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
        print("Average SSIM of samples is", ssim(pred, images).item())
    
        generated_samples = vae.sample(samples, device)
        save_image(generated_samples, os.path.join(path, 'generated.png'), nrow=samples)

        print('Reconstructed and generated images saved in', path, 'directory')
        
        generated_count = 0
        while generated_count < fid_samples:
            curr_batch = min(batch_size, fid_samples - generated_count)
            generated = vae.sample(curr_batch, device)
            fid_2048.update(images_convert_fid(generated), real=False)
            fid_192.update(images_convert_fid(generated), real=False)
            kid.update(images_convert_fid(generated), real=False)
            generated_count += curr_batch
    
        fid_2048_score = fid_2048.compute().item()
        fid_192_score = fid_192.compute().item()
        kid_mean, kid_std = kid.compute()
        
        print('For VQ-VAE:')
        print('FID with feature=2048 is', fid_2048_score)
        print('FID with feature=192 is', fid_192_score)
        print('KID (for VQ-VAE)', kid_mean.item(), '+-', kid_std.item())


def vqvae_recon_gen(vqvae, pixelcnn, test_loader, fid_2048, fid_192, kid, device, samples=12, fid_samples=200, batch_size=32):
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
        print("Average SSIM of samples is", ssim(pred, images).item())
    
        generated_samples = pixelcnn.sample(vqvae, samples, device)
        save_image(generated_samples, os.path.join(path, 'generated.png'), nrow=samples)

        print('Reconstructed and generated images saved in', path, 'directory')
        
        generated_count = 0
        while generated_count < fid_samples:
            curr_batch = min(batch_size, fid_samples - generated_count)
            generated = pixelcnn.sample(vqvae, curr_batch, device)
            fid_2048.update(images_convert_fid(generated), real=False)
            fid_192.update(images_convert_fid(generated), real=False)
            kid.update(images_convert_fid(generated), real=False)
            generated_count += curr_batch
    
        fid_2048_score = fid_2048.compute().item()
        fid_192_score = fid_192.compute().item()
        kid_mean, kid_std = kid.compute()
        
        print('For VQ-VAE:')
        print('FID with feature=2048 is', fid_2048_score)
        print('FID with feature=192 is', fid_192_score)
        print('KID (for VQ-VAE)', kid_mean.item(), '+-', kid_std.item())


def eval_vae(args, test_loader, fid_2048, fid_192, kid, device):
    vae = load_vae(args, device)
    vae_recon_gen(vae, test_loader, fid_2048, fid_192, kid, device)
    
    
def eval_vqvae(args, test_loader, fid_2048, fid_192, kid, device):
    vqvae = load_vqvae(args, device)
    pixelcnn = load_pixelcnn(args, device)
    vqvae_recon_gen(vqvae, pixelcnn, test_loader, fid_2048, fid_192, kid, device)


def main(args):
    device = get_device()
    test_loader = load_data(args.test)
    fid_2048 = init_fid(test_loader, 2048, device)
    fid_192 = init_fid(test_loader, 192, device)
    kid = init_kid(test_loader, device)
    
    if args.model == 'vae':
        eval_vae(args, test_loader, fid_2048, fid_192, kid, device)
    elif args.model == 'vqvae':
        eval_vqvae(args, test_loader, fid_2048, fid_192, kid, device)
    else:
        raise ValueError("--model argument must either be 'vae' or 'vqvae'")


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    main(args)