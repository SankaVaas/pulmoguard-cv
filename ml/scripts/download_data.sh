#!/usr/bin/env bash
# Download the Chest X-Ray Images (Pneumonia) dataset from Kaggle.
#
# Prerequisites:
#   1. pip install kaggle
#   2. Place your Kaggle API token at ~/.kaggle/kaggle.json
#      (Kaggle account -> Settings -> API -> Create New Token)
#
# Usage:
#   bash scripts/download_data.sh [target_dir]
#
# Result:
#   target_dir/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg

set -euo pipefail

TARGET_DIR="${1:-data}"
mkdir -p "${TARGET_DIR}"

echo "Downloading paultimothymooney/chest-xray-pneumonia to ${TARGET_DIR} ..."
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p "${TARGET_DIR}"

echo "Unzipping..."
unzip -q -o "${TARGET_DIR}/chest-xray-pneumonia.zip" -d "${TARGET_DIR}"

# The Kaggle archive sometimes nests an extra chest_xray/chest_xray level -
# normalize so the final layout is always TARGET_DIR/chest_xray/{train,val,test}.
if [ -d "${TARGET_DIR}/chest_xray/chest_xray" ]; then
  echo "Flattening nested chest_xray directory..."
  mv "${TARGET_DIR}/chest_xray/chest_xray"/* "${TARGET_DIR}/chest_xray/"
  rmdir "${TARGET_DIR}/chest_xray/chest_xray"
fi

echo "Done. Dataset ready at ${TARGET_DIR}/chest_xray/"
find "${TARGET_DIR}/chest_xray" -maxdepth 2 -type d | sort
