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


def load_model(args, device):
    if args.model == 'vae':
        model = modules.VAE().to(device)
    elif args.model == 'vq-vae':
        model = modules.VQVAE().to(device)
    else:
        raise ValueError("--model argument must either be 'vae' or 'vq-vae'")
        
    weights = torch.load(args.path, weights_only=True)
    model.load_state_dict(weights)
    return model

    
def eval_model(model, test_loader, device, samples=8):
    model.eval()
    os.makedirs('eval', exist_ok=True)
    path = 'eval/' + datetime.now().strftime('%H-%M-%S_%d-%m-%y')
    os.makedirs(path, exist_ok=True)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    with torch.no_grad():
        images = next(iter(test_loader))[:samples].to(device)
    
        pred, _, _ = model(images)
        cmp = torch.cat([images, pred])
        save_image(cmp, path + '/recon.png', nrow=samples)
        print("Average SSIM is", ssim(pred, images).item())
    
        # generated = model.sample(samples, device)
        # save_image(generated, path + '/generated.png', nrow=samples)

    print('Saved in', path, 'directory')


def main(args):
    device = get_device()
    test_loader = load_data()
    model = load_model(args,  device)
    eval_model(model, test_loader, device)


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    
    main(args)