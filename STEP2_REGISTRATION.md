# Step 2: Running Slice-to-Volume Registration

Now that your data is preprocessed, you can run the NH-RS2V registration methods.

## Prerequisites

### Step 1: Set up conda environment

First, create and activate the conda environment with all dependencies:

```bash
# Create the environment (only needed once)
./setup_environment.sh

# Activate the environment (needed in each new terminal session)
source activate_env.sh
# OR
conda activate nh-rs2v
```

The setup script will:
- Create a conda environment named `nh-rs2v` with Python 3.10
- Install all required dependencies (numpy, scipy, joblib, etc.)
- Install the NH-RS2V-baselines package
- Handle optional dependencies gracefully (PyTorch is optional for optimization methods)

## Quick Test (Recommended First)

Run a quick test with a small population to verify everything works:

```bash
./run_registration_quick.sh
```

This uses:
- Method: `scipy_SLSQP` (fastest and best performing)
- Population: 100 (for quick testing)
- Loss: `MAE` (Mean Absolute Error)

## Full Registration

For the best results, run with the full population:

```bash
python3 run_s2v_registration.py \
    --method scipy_SLSQP \
    --loss MAE \
    --population 2048 \
    --backend numpy
```

## Available Options

### Methods
- `scipy_SLSQP` - Sequential Least Squares Programming (recommended, fastest)
- `scipy_LBFGSB` - Limited-memory BFGS
- `nlopt_SBPLX` - Subplex algorithm
- `nlopt_COBYLA` - Constrained Optimization BY Linear Approximation
- `random` - Random search (baseline)

### Loss Functions
- `MAE` - Mean Absolute Error (recommended)
- `MSE` - Mean Squared Error
- `ZNCC` - Zero-Normalized Cross-Correlation

### Other Parameters
- `--population` - Number of random initializations (default: 2048)
  - Use 100-500 for quick testing
  - Use 2048+ for best results
- `--backend` - Computation backend
  - `numpy` - CPU only (default, works everywhere)
  - `cupy` - GPU accelerated (requires CUDA and CuPy)
- `--size` - Slice sampling size (default: 256)
- `--slice` - Path to slice numpy array (default: `slice_2d.npy`)
- `--volume` - Path to volume numpy array (default: `volume_3d.npy`)
- `--output` - Output JSON file (default: `registration_result.json`)

## Output Files

After registration completes, you'll get:

1. **`registration_result.json`** - Complete registration result in JSON format
   - Contains rotation matrix and translation vector
   - Includes metadata about the registration

2. **`registration_R.npy`** - 3x3 rotation matrix (numpy array)
   - Use this to transform coordinates

3. **`registration_T.npy`** - 3x1 translation vector (numpy array)
   - Use this to transform coordinates

## Using the Results

The registration result gives you a rigid transformation:
- **Rotation matrix R (3x3)**: Describes the rotation needed to align the slice
- **Translation vector T (3x1)**: Describes the translation needed to align the slice

To transform a point `p` from slice space to volume space:
```
p_volume = R @ p_slice + T
```

Or to transform from volume space to slice space:
```
p_slice = R.T @ (p_volume - T)
```

## Performance Notes

- **Population size**: Larger populations give better results but take longer
  - 100: ~1-2 minutes (quick test)
  - 2048: ~30-60 minutes (recommended for final results)
  - 65536: Several hours (best quality)

- **Method speed** (fastest to slowest):
  1. `scipy_SLSQP` - Fastest, best results
  2. `scipy_LBFGSB` - Similar speed
  3. `nlopt_SBPLX` - Slower
  4. `nlopt_COBYLA` - Slower
  5. `random` - Slowest (exhaustive search)

## Troubleshooting

### Import errors
If you get import errors for `nh_rs2v_baseline`, make sure:
1. The package is installed in your Python environment
2. You're using the correct Python (the one with the package installed)
3. The package path is in your PYTHONPATH

### Memory errors
If you run out of memory:
- Reduce `--population` size
- Use `--backend numpy` instead of `cupy`
- Reduce `--size` parameter

### Slow performance
- Use `scipy_SLSQP` method (fastest)
- Reduce population size for testing
- Use GPU with `--backend cupy` if available

## Next Steps

After registration:
1. Check the rotation matrix and translation vector
2. Visualize the alignment (you may want to create a visualization script)
3. Apply the transformation to your data if needed
4. Evaluate the registration quality

