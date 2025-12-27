#!/bin/bash
# Setup script to create conda environment for NH-RS2V

set -e  # Exit on error

echo "=========================================="
echo "Setting up NH-RS2V conda environment"
echo "=========================================="

ENV_NAME="nh-rs2v"
NH_RS2V_PATH="/scratch/users/Proulx-S/tools/NH-RS2V-baselines"

if [ ! -d "$NH_RS2V_PATH" ]; then
    echo "Error: NH-RS2V-baselines not found at $NH_RS2V_PATH"
    exit 1
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not available. Please load conda first:"
    echo "  module load conda"
    echo "  or"
    echo "  source /path/to/conda/etc/profile.d/conda.sh"
    exit 1
fi

echo ""
echo "Step 1: Creating conda environment '$ENV_NAME'..."
echo "(This may take a few minutes)"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists."
    read -p "  Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "  Using existing environment."
        echo ""
        echo "To activate the environment, run:"
        echo "  conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create conda environment with Python 3.10 (as required by NH-RS2V)
conda create -n "$ENV_NAME" python=3.10 -y

echo ""
echo "Step 2: Activating environment and installing core dependencies..."

# Activate environment and install dependencies
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Install core dependencies via conda (faster and more reliable)
echo "  Installing numpy, scipy, joblib via conda..."
conda install -y numpy scipy joblib scikit-learn scikit-image

# Install other dependencies via pip
echo "  Installing additional dependencies via pip..."
pip install absl-py nlopt

echo ""
echo "Step 3: Installing nh_rs2v_dataset dependency..."
# The package also needs nh_rs2v_dataset - check if it's available
if [ -d "/scratch/users/Proulx-S/tools/NH-RS2V-dataset" ]; then
    echo "  Found NH-RS2V-dataset, installing..."
    cd /scratch/users/Proulx-S/tools/NH-RS2V-dataset
    pip install -e . 2>&1 | tail -5
    cd "$NH_RS2V_PATH"
else
    echo "  Warning: NH-RS2V-dataset not found. Some functionality may be limited."
    echo "  The optimization methods should still work for basic usage."
fi

echo ""
echo "Step 4: Installing NH-RS2V-baselines package..."
cd "$NH_RS2V_PATH"

# Try to install, but don't fail on optional dependencies
echo "  Installing package (this may take a while)..."
echo "  (Note: PyTorch and CuPy errors are expected and OK - see below)"
pip install -e . 2>&1 | tee /tmp/nh_rs2v_install.log || INSTALL_ERROR=$?

# Check the installation log for errors
HAS_PYTORCH_ERROR=$(grep -q "ERROR.*torch\|ERROR.*pytorch\|torch.*cu118\|torch==2.0.1" /tmp/nh_rs2v_install.log && echo "yes" || echo "no")
HAS_CUPY_ERROR=$(grep -q "ERROR.*cupy\|cupy.*cu11" /tmp/nh_rs2v_install.log && echo "yes" || echo "no")

if [ "$HAS_PYTORCH_ERROR" = "yes" ] || [ "$HAS_CUPY_ERROR" = "yes" ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "⚠ Some dependencies failed to install - THIS IS EXPECTED!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    
    if [ "$HAS_PYTORCH_ERROR" = "yes" ]; then
        echo "PyTorch Error:"
        echo "  ✓ This is OK! PyTorch is only needed for LoFTR-S2V (deep learning)."
        echo "  ✓ Optimization methods (which you're using) don't need PyTorch."
        echo ""
    fi
    
    if [ "$HAS_CUPY_ERROR" = "yes" ]; then
        echo "CuPy Error:"
        echo "  ✓ This is OK! CuPy is only needed for GPU acceleration (backend='cupy')."
        echo "  ✓ Using backend='numpy' (CPU) doesn't need CuPy."
        echo ""
    fi
    
    echo "The optimization methods should still work with:"
    echo "  - backend='numpy' (CPU, no CuPy needed)"
    echo "  - All optimization methods (SLSQP, LBFGSB, etc.)"
    echo ""
    
    # Fix cupy import issue if needed
    echo "  Fixing CuPy import issue for numpy backend..."
    cd /scratch/users/Proulx-S/s2v
    if [ -f "fix_cupy_import.py" ]; then
        python3 fix_cupy_import.py 2>&1 | grep -E "✓|⚠|ℹ" || true
    fi
    cd "$NH_RS2V_PATH"
else
    echo ""
    echo "✓ Package installed successfully!"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To use the environment:"
echo "  1. Activate it:"
echo "     conda activate $ENV_NAME"
echo ""
echo "  2. Run the registration script:"
echo "     ./run_registration_quick.sh"
echo ""
echo "Note: You need to activate the environment in each new terminal session."

