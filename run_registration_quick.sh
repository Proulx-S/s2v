#!/bin/bash
# Quick test script for registration with smaller population
# This is faster for testing, use full population for final results

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nh-rs2v

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "nh-rs2v" ]; then
    echo "Error: Failed to activate nh-rs2v conda environment."
    echo "Make sure the environment exists: conda env list"
    echo "If it doesn't exist, run: ./setup_environment.sh"
    exit 1
fi

CONDA_PYTHON="/home/sebp/miniconda3/envs/nh-rs2v/bin/python3"
if [ ! -f "$CONDA_PYTHON" ]; then
    CONDA_PYTHON="$(conda info --base)/envs/nh-rs2v/bin/python3"
fi

echo "Using Python from conda environment: $CONDA_PYTHON"
echo ""

"$CONDA_PYTHON" run_s2v_registration.py \
    --method scipy_SLSQP \
    --loss MAE \
    --population 100 \
    --backend numpy \
    --size 256 \
    --output registration_result_quick.json

