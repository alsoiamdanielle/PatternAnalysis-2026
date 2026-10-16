import dataset
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_data():
    dir = os.path.expanduser('~/Downloads/HipMRI/keras_slices_train')
    data = dataset.HipMRIDataset(dir, early_stop=True)
    
    sample = data.__getitem__(0)
    plt.imshow(sample)
    plt.show()
    
    return DataLoader(data, batch_size=32, shuffle=True)

def main():
    train_loader = load_data()

if __name__ == "__main__":
    main();
