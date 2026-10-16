import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import os
import glob


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1

    return res


# load medical image functions
def load_data_2D(imageNames, shape, normImage, dtype, early_stop):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    # get fixed size
    num = len(imageNames)
    target_rows = shape[0]
    target_cols = shape[1]
    images = np.zeros((num, 1, target_rows, target_cols), dtype=dtype)
    
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged')  # read disk only
        
        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        
        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)
            
        rows, cols = inImage.shape
        
        row_pad = max(0, target_rows - rows)
        top_pad = row_pad // 2
        bot_pad = row_pad - top_pad
        
        col_pad = max(0, target_cols - cols)
        left_pad = col_pad // 2
        right_pad = col_pad - left_pad
        
        if row_pad > 0 or col_pad > 0:
            inImage = np.pad(inImage, ((top_pad, bot_pad), (left_pad, right_pad)), 'constant', constant_values=0)
            
        images[i, 0, :, :] = inImage
        
        if i > 20 and early_stop:
            break
            
    return images


class HipMRIDataset(Dataset):
    def __init__(self, dir, shape=(256, 144), normImage=True, dtype=np.float32, early_stop=False):
        self.img_names = sorted(glob.glob(os.path.join(dir, '*.nii*')))
        self.imgs = load_data_2D(self.img_names, shape, normImage, dtype, early_stop)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        return self.imgs[idx]
