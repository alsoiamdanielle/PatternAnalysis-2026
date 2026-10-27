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
import time
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


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def count_parameters(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params


def load_data(data_dir, batch_size=32, test=False):
    dir = os.path.expanduser(os.path.join(data_dir, 'keras_slices_test'))
    data = dataset.HipMRIDataset(dir, early_stop=test)
    return DataLoader(data, batch_size=batch_size, shuffle=False)
    
    
def get_sample_images(test_loader, samples=8, seed=42):
    total_slices = len(test_loader.dataset)
    rng = np.random.RandomState(seed)
    chosen_indices = sorted(rng.choice(total_slices, size=min(samples, total_slices), replace=False))
    sampled_tensors = [test_loader.dataset[i] for i in chosen_indices]
    return torch.stack(sampled_tensors)


def load_vae(args, device):
    vae = modules.VAE(embedding_dim=args.embedding_dim).to(device)
    weights = torch.load(os.path.join(args.path, 'VAE.pth'), weights_only=True)
    vae.load_state_dict(weights)
    return vae
    
    
def load_vqvae(args, device):
    vqvae = modules.VQVAE(num_embeddings=args.num_embeddings, embedding_dim=args.embedding_dim).to(device)
    weights = torch.load(os.path.join(args.path, 'VQVAE.pth'), weights_only=True)
    vqvae.load_state_dict(weights)
    return vqvae

    
def load_pixelcnn(args, device):
    pixelcnn = modules.PixelCNN(num_embeddings=args.num_embeddings).to(device)
    weights = torch.load(os.path.join(args.path, 'pixelCNN.pth'), weights_only=True)
    pixelcnn.load_state_dict(weights)
    return pixelcnn


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


def vae_recon_gen(vae, test_loader, fid_2048, fid_192, kid, device, samples=8, fid_samples=200, batch_size=32, seed=42):
    vae.eval()
    os.makedirs('eval', exist_ok=True)
    path = os.path.join('eval', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    print('==============================================================================')
    print('VAE RESOURCE PROFILING')
    print(f'Total Trainable Parameters: {count_parameters(vae):,}')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.time()
    _ = vae.sample(batch_size, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_32 = (time.time() - start_time) * 1000.0

    if device.type == 'cuda':
        inference_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        inference_vram = 0.0

    print(f'Batch Inference Latency (Batch Size={batch_size}): {latency_32:.2f} ms')
    print(f'Peak Inference VRAM: {inference_vram:.2f} MB')
    print('==============================================================================')

    with torch.no_grad():
        images = get_sample_images(test_loader, samples=samples, seed=seed).to(device)
    
        pred, _, _ = vae(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, os.path.join(path, 'recon.png'), nrow=samples)
        print("Average SSIM of test sample batch is", ssim(pred, images).item())
    
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
        
        print('For VAE:')
        print('FID with feature=2048 is', fid_2048_score)
        print('FID with feature=192 is', fid_192_score)
        print('KID is', kid_mean.item(), '+-', kid_std.item())


def vqvae_recon_gen(vqvae, pixelcnn, test_loader, fid_2048, fid_192, kid, device, samples=8, fid_samples=200, batch_size=32, temperature=0.9, seed=42):
    vqvae.eval()
    pixelcnn.eval()
    os.makedirs('eval', exist_ok=True)
    path = os.path.join('eval', datetime.now().strftime('%H-%M-%S_%d-%m-%y'))
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    total_params = count_parameters(vqvae) + count_parameters(pixelcnn)
    print('==============================================================================')
    print('VQ-VAE + PIXELCNN RESOURCE PROFILING')
    print(f'VQ-VAE Parameters: {count_parameters(vqvae):,}')
    print(f'PixelCNN Parameters: {count_parameters(pixelcnn):,}')
    print(f'Total Pipeline Parameters: {total_params:,}')

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    start_time = time.time()
    _ = pixelcnn.sample(vqvae, batch_size, device, temperature=temperature)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    latency_32 = (time.time() - start_time) * 1000.0

    if device.type == 'cuda':
        inference_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        inference_vram = 0.0

    print(f'Batch Inference Latency (Batch Size={batch_size}): {latency_32:.2f} ms ({latency_32 / batch_size:.2f} ms/sample)')
    print(f'Peak Inference VRAM: {inference_vram:.2f} MB')
    print('==============================================================================')

    with torch.no_grad():
        images = get_sample_images(test_loader, samples=samples, seed=seed).to(device)
    
        pred, _, _ = vqvae(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, os.path.join(path, 'recon.png'), nrow=samples)
        print("Average SSIM of test sample batch is", ssim(pred, images).item())
    
        print(f"Sampling {samples} synthetic scans via PixelCNN prior (temperature={temperature})...")
        generated_samples = pixelcnn.sample(vqvae, samples, device, temperature=temperature)
        save_image(generated_samples, os.path.join(path, 'generated.png'), nrow=samples)

        print('Reconstructed and generated images saved in', path, 'directory')
        
        generated_count = 0
        while generated_count < fid_samples:
            curr_batch = min(batch_size, fid_samples - generated_count)
            generated = pixelcnn.sample(vqvae, curr_batch, device, temperature=temperature)
            fid_2048.update(images_convert_fid(generated), real=False)
            fid_192.update(images_convert_fid(generated), real=False)
            kid.update(images_convert_fid(generated), real=False)
            generated_count += curr_batch
    
        fid_2048_score = fid_2048.compute().item()
        fid_192_score = fid_192.compute().item()
        kid_mean, kid_std = kid.compute()
        
        print('For VQ-VAE + PixelCNN:')
        print('FID with feature=2048 is', fid_2048_score)
        print('FID with feature=192 is', fid_192_score)
        print('KID is', kid_mean.item(), '+-', kid_std.item())


def eval_vae(args, test_loader, fid_2048, fid_192, kid, device):
    vae = load_vae(args, device)
    vae_recon_gen(vae, test_loader, fid_2048, fid_192, kid, device, samples=args.samples, fid_samples=args.fid_samples, batch_size=args.batch_size, seed=args.seed)
    
    
def eval_vqvae(args, test_loader, fid_2048, fid_192, kid, device):
    vqvae = load_vqvae(args, device)
    pixelcnn = load_pixelcnn(args, device)
    vqvae_recon_gen(vqvae, pixelcnn, test_loader, fid_2048, fid_192, kid, device, samples=args.samples, fid_samples=args.fid_samples, batch_size=args.batch_size, temperature=args.temperature, seed=args.seed)


def main(args):
    set_seed(args.seed)
    device = get_device()
    test_loader = load_data(args.data_dir, batch_size=args.batch_size, test=args.test)
    fid_2048 = init_fid(test_loader, 2048, device)
    fid_192 = init_fid(test_loader, 192, device)
    kid = init_kid(test_loader, device)
    
    if args.model == 'vae':
        eval_vae(args, test_loader, fid_2048, fid_192, kid, device)
    elif args.model == 'vqvae':
        eval_vqvae(args, test_loader, fid_2048, fid_192, kid, device)
    else:
        raise ValueError("--model argument must either be 'vae' or 'vqvae'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['vae', 'vqvae'])
    parser.add_argument('--path', required=True)
    parser.add_argument('--data_dir', default='~/Downloads/HipMRI')
    parser.add_argument('--num_embeddings', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--fid_samples', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    
    main(args)