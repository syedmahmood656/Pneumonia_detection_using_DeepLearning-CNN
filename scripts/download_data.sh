#!/bin/bash
set -e

mkdir -p data/raw

echo "Downloading the pre-scaled 224x224 version..."
kaggle datasets download -d khanfashee/nih-chest-x-ray-14-224x224-resized -p data/raw/

echo "Unpacking images..."
unzip -q data/raw/nih-chest-x-ray-14-224x224-resized.zip -d data/raw/
rm data/raw/nih-chest-x-ray-14-224x224-resized.zip

echo "Pre-scaled data pipeline downloaded successfully!"