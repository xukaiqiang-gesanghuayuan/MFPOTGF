import os
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

class TripleNiftiDataset(Dataset):
    def __init__(self, root_dir1, root_dir2, root_dir3):
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.root_dir3 = root_dir3
        self.paths1 = [os.path.join(root_dir1, f) for f in os.listdir(root_dir1) if f.endswith('.npy')]
        self.paths2 = [os.path.join(root_dir2, f) for f in os.listdir(root_dir2) if f.endswith('.npy')]
        self.paths3 = [os.path.join(root_dir3, f) for f in os.listdir(root_dir3) if f.endswith('.npy')]
        self.labels = [int(os.path.basename(path).split('-')[1].split('.')[0]) - 1 for path in self.paths1]

    def __len__(self):
        return len(self.paths1)

    def __getitem__(self, idx):
        image_path1 = self.paths1[idx]
        image_path2 = self.paths2[idx]
        image_path3 = self.paths3[idx]
        
        image_tensor1 = torch.tensor(np.load(image_path1), dtype=torch.float32).unsqueeze(0)
        image_tensor2 = torch.tensor(np.load(image_path2), dtype=torch.float32).unsqueeze(0)
        image_tensor3 = torch.tensor(np.load(image_path3), dtype=torch.float32).unsqueeze(0)
        
        label = self.labels[idx]
        return (image_tensor1, image_tensor2, image_tensor3), torch.tensor(label, dtype=torch.long)

def main():
    root_dir1 = '/home/xukaiqiang/shuju/yanchaogan/VBM/mwc1npy'  # Path to the first set of 3D data
    root_dir2 = '/home/xukaiqiang/shuju/yanchaogan/VBM/mwc2npy'  # Path to the second set of 3D data
    root_dir3 = '/home/xukaiqiang/shuju/yanchaogan/VBM/mwc3npy'   # Path to the third set of 3D data

    dataset = TripleNiftiDataset(root_dir1, root_dir2, root_dir3)
    labels = []

    # Loop over the dataset and collect labels
    for (image_tensor1, image_tensor2, image_tensor3), label in dataset:
        labels.append(label.item())

    label_counts = Counter(labels)
    print("Label counts:", label_counts)

if __name__ == '__main__':
    main()