# Quick Start Guide

## Step 1: Preprocess Your Data

1. Load AFNI module:
   ```bash
   ml afni
   ```

2. Run preprocessing:
   ```bash
   ./preprocess_data.sh
   ```

This will:
- Motion correct your slice timeseries
- Average across time
- Convert to numpy arrays ready for NH-RS2V

## Step 2: Set Up Environment

1. Create conda environment (only needed once):
   ```bash
   ./setup_environment.sh
   ```

2. Activate the environment:
   ```bash
   source activate_env.sh
   # OR
   conda activate nh-rs2v
   ```

## Step 3: Run Registration

Once preprocessing is complete and environment is activated:

**Quick test:**
```bash
./run_registration_quick.sh
```

**Full registration:**
```bash
python3 run_s2v_registration.py \
    --method scipy_SLSQP \
    --loss MAE \
    --population 2048
```

## Files Created

- `slice_2d.npy` - 2D slice (numpy array)
- `volume_3d.npy` - 3D volume (numpy array)
- `data_metadata.json` - Metadata about your data

See `README_preprocessing.md` for detailed documentation.

