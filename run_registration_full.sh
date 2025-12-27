#!/bin/bash
# Run registration with full population size for better results

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nh-rs2v

# Get Python from conda environment
CONDA_PYTHON="/home/sebp/miniconda3/envs/nh-rs2v/bin/python3"

if [ ! -f "$CONDA_PYTHON" ]; then
    echo "Error: Conda Python not found at $CONDA_PYTHON"
    echo "Please make sure the nh-rs2v environment is set up:"
    echo "  ./setup_environment.sh"
    exit 1
fi

echo "Using Python from conda environment: $CONDA_PYTHON"
echo ""
echo "Running registration with full population size (2048)..."
echo "This will take longer but should give better results."
echo ""

# Run registration with full population
"$CONDA_PYTHON" /scratch/users/Proulx-S/s2v/run_s2v_registration.py \
    --method scipy_SLSQP \
    --loss MAE \
    --population 2048 \
    --backend numpy \
    --size 256 \
    --output registration_result_full.json

echo ""
echo "Registration complete! Results saved to registration_result_full.json"
echo ""
echo "To visualize the results:"
echo "  python3 visualize_registration.py --result registration_result_full.json"
echo "  python3 visualize_slice_position.py --result registration_result_full.json"

