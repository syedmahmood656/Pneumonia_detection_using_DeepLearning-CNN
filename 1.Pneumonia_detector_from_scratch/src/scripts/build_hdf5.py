"""
Pre-apply CLAHE + resize to ALL subset images and store in a single HDF5 file.
Run once before training. Saves ~40% I/O time on Colab.
Target: ~6–7 GB for 16K images at 224×224 uint8.
"""
import h5py
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import os

def apply_clahe(image: np.ndarray, clip_limit=2.0, grid=(8, 8)) -> np.ndarray:
    """CLAHE: Contrast Limited Adaptive Histogram Equalization.
    Rationale: X-rays have very narrow intensity ranges; CLAHE locally
    enhances contrast without over-amplifying noise (unlike global histogram eq).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
    return clahe.apply(image)

def build_hdf5(
    split_csvs: dict,  # {'train': 'data/splits/train.csv', ...}
    image_dir: str,
    output_path: str = 'data/processed/cxr_dataset.h5',
    image_size: int = 224,
):
    all_dfs = {name: pd.read_csv(path) for name, path in split_csvs.items()}
    total = sum(len(df) for df in all_dfs.values())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(output_path, 'w') as hf:
        for split_name, df in all_dfs.items():
            grp = hf.create_group(split_name)
            n = len(df)

            # Pre-allocate datasets
            images_ds = grp.create_dataset(
                'images', shape=(n, image_size, image_size),
                dtype=np.uint8,
                chunks=(32, image_size, image_size),  # chunk by batch size
                compression='lzf',  # fast compression
            )
            labels_list = []

            for i, row in tqdm(df.iterrows(), total=n, desc=f"Building {split_name}"):
                img_path = os.path.join(image_dir, row['Image Index'])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.zeros((image_size, image_size), dtype=np.uint8)
                img = apply_clahe(img)
                img = cv2.resize(img, (image_size, image_size))
                images_ds[i] = img
                labels_list.append(row['Finding Labels'])

            # Store labels as variable-length strings
            dt = h5py.special_dtype(vlen=str)
            grp.create_dataset('labels', data=np.array(labels_list, dtype=object), dtype=dt)

        print(f"\nHDF5 built: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1e9:.2f} GB")

build_hdf5(
    split_csvs={
        'train': 'data/splits/train.csv',
        'val':   'data/splits/val.csv',
        'test':  'data/splits/test.csv',
    },
    image_dir='data/raw/images/',
)
