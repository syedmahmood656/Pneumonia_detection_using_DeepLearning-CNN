import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import ChestXrayDataset
from src.data.augmentations import get_train_transforms, get_val_transforms

class ChestXrayDataModule(pl.LightningDataModule):
    def __init__(self, hdf5_path: str, batch_size: int = 32,
                 num_workers: int = 4, image_size: int = 224):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage=None):
        self.train_ds = ChestXrayDataset(
            self.hdf5_path, 'train', transform=get_train_transforms(self.image_size)
        )
        self.val_ds = ChestXrayDataset(
            self.hdf5_path, 'val', transform=get_val_transforms(self.image_size)
        )
        self.test_ds = ChestXrayDataset(
            self.hdf5_path, 'test', transform=get_val_transforms(self.image_size)
        )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self): return self._loader(self.train_ds, shuffle=True)
    def val_dataloader(self):   return self._loader(self.val_ds,   shuffle=False)
    def test_dataloader(self):  return self._loader(self.test_ds,  shuffle=False)