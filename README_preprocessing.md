# Preprocessing and Registration Guide

This directory contains scripts to preprocess your MRI data and run slice-to-volume registration using the NH-RS2V baseline methods.

## Overview

The preprocessing pipeline:
1. **Motion correction**: Corrects for motion in the slice timeseries using AFNI's `3dvolreg`
2. **Temporal averaging**: Averages the motion-corrected timeseries across time
3. **Data conversion**: Converts NIfTI files to numpy arrays compatible with NH-RS2V methods

## Prerequisites

### Required software:
- **AFNI**: For motion correction and temporal averaging
  ```bash
  ml afni
  ```

### Required Python packages:
- `nibabel`: For reading/writing NIfTI files
- `numpy`: For array operations

Install with:
```bash
pip install nibabel numpy
```

### NH-RS2V baseline methods (optional, for registration):
If you want to run the registration methods, you'll need to install the NH-RS2V-baselines package:
```bash
cd /path/to/NH-RS2V-baselines
poetry install  # or pip install -e .
```

## Usage

### Step 1: Preprocess the data

Run the preprocessing script:
```bash
ml afni
./preprocess_data.sh
```

This will:
- Motion correct `slice.nii.gz` → `slice_mc.nii.gz`
- Average across time → `slice_avg.nii.gz`
- Extract 2D slice → `slice_2d.nii.gz` and `slice_2d.npy`
- Prepare 3D volume → `volume_3d.nii.gz` and `volume_3d.npy`
- Save metadata → `data_metadata.json`

### Step 2: Run registration (optional)

Once the data is preprocessed, you can run registration using the optimization-based methods:

```bash
python3 run_s2v_registration.py \
    --method scipy_SLSQP \
    --loss MAE \
    --population 2048 \
    --backend numpy
```

Available methods:
- `scipy_SLSQP`: Sequential Least Squares Programming (recommended, fastest)
- `scipy_LBFGSB`: Limited-memory BFGS
- `nlopt_SBPLX`: Subplex algorithm
- `nlopt_COBYLA`: Constrained Optimization BY Linear Approximation
- `random`: Random search (baseline)

Available loss functions:
- `MAE`: Mean Absolute Error
- `MSE`: Mean Squared Error
- `ZNCC`: Zero-Normalized Cross-Correlation

## Output Files

After preprocessing:
- `slice_2d.npy`: 2D slice as numpy array (ready for NH-RS2V)
- `volume_3d.npy`: 3D volume as numpy array (ready for NH-RS2V)
- `slice_2d.nii.gz`: 2D slice as NIfTI (for visualization)
- `volume_3d.nii.gz`: 3D volume as NIfTI (for visualization)
- `data_metadata.json`: Metadata including shapes, affines, voxel sizes
- `motion_params.1D`: Motion correction parameters from AFNI

## Notes

1. **Motion correction**: The script uses the first volume as the reference for motion correction. You can modify the `-base` parameter in the script if needed.

2. **Data normalization**: The NH-RS2V methods automatically normalize the data to [0, 1] range, so you don't need to worry about intensity scaling.

3. **LoFTR-S2V**: The deep learning method (LoFTR-S2V) requires more setup and the full NH-RS2V dataset infrastructure. For now, use the optimization-based methods, or refer to the NH-RS2V-baselines examples for LoFTR-S2V usage.

4. **GPU acceleration**: If you have CuPy installed and a CUDA-capable GPU, you can use `--backend cupy` for faster computation.

## Troubleshooting

### nibabel not found
```bash
pip install nibabel
```

### AFNI commands not found
```bash
ml afni
```

### Volume has unexpected dimensions
The script handles volumes with time dimensions by averaging. If you encounter issues, check the volume dimensions:
```bash
fslhd volume.nii.gz | grep -E "dim[0-9]"
```

## Next Steps

After preprocessing, you can:
1. Use the numpy arrays directly with NH-RS2V methods
2. Visualize the preprocessed data to verify quality
3. Run registration and evaluate the results

For more advanced usage, see the NH-RS2V-baselines repository examples.

