import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        # Spatial — anatomically plausible for X-rays
        A.HorizontalFlip(p=0.5),                     # Left-right flip is OK
        A.Rotate(limit=10, border_mode=0, p=0.4),    # Small rotation (patient positioning)
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=5, p=0.3),
        A.ElasticTransform(alpha=80, sigma=80 * 0.05,
                           alpha_affine=80 * 0.03, p=0.2),

        # Intensity — simulate exposure variation across machines
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),

        # Normalize and convert
        A.Normalize(mean=0.5, std=0.25),             # Empirical for X-ray pixel dist.
        ToTensorV2(),                                  # (H, W) → (1, H, W) float32
    ])

def get_val_transforms(image_size: int = 224) -> A.Compose:
    """No augmentation at val/test — only normalize."""
    return A.Compose([
        A.Normalize(mean=0.5, std=0.25),
        ToTensorV2(),
    ])
