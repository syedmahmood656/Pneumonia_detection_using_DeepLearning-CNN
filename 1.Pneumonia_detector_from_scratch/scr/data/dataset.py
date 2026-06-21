import numpy as np  
import torch
from torch.utils.data import Dataset
import h5py

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

class ChestXrayDataset(Dataset):
    """HDF5-backed dataset. Images are pre-CLAHE'd and resized."""

    def __init__(self, hdf5_path: str, split: str, transform=None):
        self.hdf5_path = hdf5_path
        self.split = split
        self.transform = transform
        self._h5 = None  # Lazy open (required for multi-worker DataLoader)

        # Build label matrix from string labels stored in HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            raw_labels = [s.decode() if isinstance(s, bytes) else s
                          for s in hf[split]['labels'][:]]
            self.length = len(raw_labels)

        self.label_matrix = self._parse_labels(raw_labels)

    def _parse_labels(self, raw_labels):
        matrix = np.zeros((len(raw_labels), len(LABELS)), dtype=np.float32)
        for i, findings in enumerate(raw_labels):
            if findings != 'No Finding':
                for f in findings.split('|'):
                    if f in LABELS:
                        matrix[i, LABELS.index(f)] = 1.0
        return matrix

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open HDF5 lazily per worker
        if self._h5 is None:
            self._h5 = h5py.File(self.hdf5_path, 'r')

        img = self._h5[self.split]['images'][idx]  # (224, 224) uint8
        label = self.label_matrix[idx]              # (14,) float32

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']              # (1, H, W) float tensor
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, torch.tensor(label, dtype=torch.float32)
