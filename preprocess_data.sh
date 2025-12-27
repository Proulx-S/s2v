#!/bin/bash
# Preprocess MRI data for slice-to-volume registration
# Performs motion correction and temporal averaging on slice timeseries
# Prepares data in format compatible with NH-RS2V baseline methods

set -e  # Exit on error

# Load AFNI module
ml afni

# Input files
SLICE_INPUT="slice.nii.gz"
VOLUME_INPUT="volume.nii.gz"

# Output files
SLICE_AVERAGED="slice_avg.nii.gz"
SLICE_2D="slice_2d.nii.gz"
VOLUME_3D="volume_3d.nii.gz"

# Python output files (numpy arrays)
SLICE_NPY="slice_2d.npy"
VOLUME_NPY="volume_3d.npy"

echo "=========================================="
echo "Preprocessing data for S2V registration"
echo "=========================================="

# Step 1: Temporal averaging (skip motion correction for now)
echo ""
echo "Step 1: Averaging across time dimension (skipping motion correction)..."
if [ ! -f "$SLICE_AVERAGED" ]; then
    3dTstat \
        -prefix "$SLICE_AVERAGED" \
        -mean \
        "$SLICE_INPUT"
    echo "Temporal averaging complete."
else
    echo "Averaged file already exists, skipping..."
fi

# Step 2: Extract 2D slice (remove singleton time dimension)
echo ""
echo "Step 2: Extracting 2D slice..."
if [ ! -f "$SLICE_2D" ]; then
    # Use 3dTstat to remove time dimension (should already be 1 after averaging, but this ensures it)
    # Then use 3drefit to ensure proper header
    3dTstat -mean -prefix "$SLICE_2D" "$SLICE_AVERAGED"
    3drefit -view orig -space ORIG "$SLICE_2D"
    echo "2D slice extraction complete."
else
    echo "2D slice file already exists, skipping..."
fi

# Step 3: Ensure volume is 3D (remove singleton time dimension if present)
echo ""
echo "Step 3: Preparing 3D volume..."
if [ ! -f "$VOLUME_3D" ]; then
    # Check if volume has time dimension using AFNI
    nvols=$(3dinfo -nv "$VOLUME_INPUT" 2>/dev/null || echo "1")
    if [ "$nvols" -gt 1 ]; then
        echo "Warning: Volume has $nvols time points. Averaging across time..."
        3dTstat -mean -prefix "$VOLUME_3D" "$VOLUME_INPUT"
    else
        # Just copy if already 3D
        cp "$VOLUME_INPUT" "$VOLUME_3D"
    fi
    echo "3D volume preparation complete."
else
    echo "3D volume file already exists, skipping..."
fi

# Step 4: Convert to numpy arrays using Python
echo ""
echo "Step 4: Converting to numpy arrays for NH-RS2V..."
python3 << 'PYTHON_SCRIPT'
import numpy as np
import nibabel as nib
import os

# Load the 2D slice
print("Loading 2D slice...")
slice_img = nib.load("slice_2d.nii.gz")
slice_data = slice_img.get_fdata().squeeze()  # Remove all singleton dimensions

# Ensure it's 2D
if slice_data.ndim > 2:
    # If still has extra dimensions, take the middle slice
    while slice_data.ndim > 2:
        mid_idx = slice_data.shape[-1] // 2
        slice_data = slice_data[..., mid_idx]
    
print(f"Slice shape: {slice_data.shape}")
print(f"Slice dtype: {slice_data.dtype}")
print(f"Slice range: [{slice_data.min():.2f}, {slice_data.max():.2f}]")

# Load the 3D volume
print("\nLoading 3D volume...")
volume_img = nib.load("volume_3d.nii.gz")
volume_data = volume_img.get_fdata().squeeze()

# Ensure it's 3D
if volume_data.ndim < 3:
    raise ValueError(f"Volume should be 3D, got shape {volume_data.shape}")
elif volume_data.ndim > 3:
    # If has time dimension, take first or average
    if volume_data.ndim == 4:
        volume_data = volume_data[..., 0]  # Take first time point
    else:
        raise ValueError(f"Volume has unexpected dimensions: {volume_data.shape}")

print(f"Volume shape: {volume_data.shape}")
print(f"Volume dtype: {volume_data.dtype}")
print(f"Volume range: [{volume_data.min():.2f}, {volume_data.max():.2f}]")

# Convert to float32 for compatibility with NH-RS2V
slice_data = slice_data.astype(np.float32)
volume_data = volume_data.astype(np.float32)

# Save as numpy arrays
np.save("slice_2d.npy", slice_data, allow_pickle=False)
np.save("volume_3d.npy", volume_data, allow_pickle=False)

print("\n✓ Numpy arrays saved:")
print(f"  - slice_2d.npy: shape {slice_data.shape}, dtype {slice_data.dtype}")
print(f"  - volume_3d.npy: shape {volume_data.shape}, dtype {volume_data.dtype}")

# Also save metadata (convert numpy types to native Python types for JSON)
def convert_to_python_type(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return [convert_to_python_type(item) for item in obj.flatten()] if obj.size == 1 else obj.tolist()
    elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_python_type(value) for key, value in obj.items()}
    return obj

# Get voxel sizes and convert to lists first
slice_voxel_size = [float(x) for x in slice_img.header.get_zooms()[:3]]
volume_voxel_size = [float(x) for x in volume_img.header.get_zooms()[:3]]

metadata = {
    'slice_shape': list(slice_data.shape),
    'volume_shape': list(volume_data.shape),
    'slice_affine': convert_to_python_type(slice_img.affine.tolist()),
    'volume_affine': convert_to_python_type(volume_img.affine.tolist()),
    'slice_voxel_size': slice_voxel_size,
    'volume_voxel_size': volume_voxel_size,
}

import json
with open('data_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata saved to data_metadata.json")
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Preprocessing complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - slice_2d.nii.gz: 2D temporally averaged slice"
echo "  - volume_3d.nii.gz: 3D volume"
echo "  - slice_2d.npy: 2D slice as numpy array (for NH-RS2V)"
echo "  - volume_3d.npy: 3D volume as numpy array (for NH-RS2V)"
echo "  - data_metadata.json: Metadata about the data"
echo ""
echo "You can now use these files with the NH-RS2V baseline methods."

