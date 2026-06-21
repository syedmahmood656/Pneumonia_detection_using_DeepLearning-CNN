import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

RARE_CLASSES = ['Hernia', 'Pneumonia', 'Mass', 'Nodule', 'Emphysema', 'Fibrosis']

def create_stratified_subset(
    metadata_csv: str,
    output_dir: str = 'data/splits/',
    total_samples: int = 16000,
    no_finding_cap: int = 2000,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """
    Intelligent sampling strategy:
    - 100% of rare classes (Hernia, Pneumonia, Mass, Nodule, Emphysema, Fibrosis)
    - Cap 'No Finding' at no_finding_cap to prevent majority class dominance
    - Fill remaining quota proportionally from common pathologies
    - Stratified train/val/test split on primary label
    """
    df = pd.read_csv(metadata_csv)
    selected_indices = set()

    # 1. Collect all rare class samples first
    for cls in RARE_CLASSES:
        mask = df['Finding Labels'].str.contains(cls, na=False)
        rare_idx = df[mask].index.tolist()
        selected_indices.update(rare_idx)
        print(f"  {cls:22s}: {len(rare_idx):5d} samples (100%)")

    # 2. Cap 'No Finding'
    no_finding_mask = df['Finding Labels'] == 'No Finding'
    nf_idx = (
        df[no_finding_mask]
        .sample(n=min(no_finding_cap, no_finding_mask.sum()), random_state=random_state)
        .index.tolist()
    )
    selected_indices.update(nf_idx)
    print(f"  {'No Finding':22s}: {len(nf_idx):5d} samples (capped at {no_finding_cap})")

    # 3. Fill remaining quota from common pathologies
    remaining = total_samples - len(selected_indices)
    common_mask = (
        ~df['Finding Labels'].str.contains('|'.join(RARE_CLASSES), na=False)
        & ~no_finding_mask
    )
    common_candidates = [i for i in df[common_mask].index if i not in selected_indices]

    if remaining > 0 and common_candidates:
        extra = (
            pd.Series(common_candidates)
            .sample(n=min(remaining, len(common_candidates)), random_state=random_state)
            .tolist()
        )
        selected_indices.update(extra)

    subset_df = df.loc[list(selected_indices)].reset_index(drop=True)
    print(f"\n  Total subset: {len(subset_df):,} images")

    # Stratify on primary (first) label
    subset_df['_primary'] = subset_df['Finding Labels'].str.split('|').str[0]

    train_df, temp_df = train_test_split(
        subset_df,
        test_size=val_size + test_size,
        stratify=subset_df['_primary'],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df['_primary'],
        random_state=random_state,
    )

    os.makedirs(output_dir, exist_ok=True)
    for split, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        split.drop(columns=['_primary']).to_csv(f'{output_dir}/{name}.csv', index=False)

    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    return train_df, val_df, test_df