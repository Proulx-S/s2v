#!/bin/bash
# Quick script to activate the NH-RS2V conda environment

ENV_NAME="nh-rs2v"

# Try to find and source conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda &> /dev/null; then
    # Conda is in PATH, try to initialize
    eval "$(conda shell.bash hook)"
else
    echo "Error: conda is not available. Please load conda first:"
    echo "  module load conda"
    echo "  or"
    echo "  source /path/to/conda/etc/profile.d/conda.sh"
    exit 1
fi

# Activate environment
if conda env list | grep -q "^${ENV_NAME} "; then
    conda activate "$ENV_NAME"
    echo "✓ Activated conda environment: $ENV_NAME"
    echo "  Python: $(which python3)"
    echo ""
    echo "You can now run:"
    echo "  ./run_registration_quick.sh"
else
    echo "Error: Environment '$ENV_NAME' does not exist."
    echo "Please run ./setup_environment.sh first to create it."
    exit 1
fi

