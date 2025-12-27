#!/bin/bash
# Try different optimization methods if the default fails
# This script tries methods in order of speed/quality

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nh-rs2v

CONDA_PYTHON="/home/sebp/miniconda3/envs/nh-rs2v/bin/python3"

if [ ! -f "$CONDA_PYTHON" ]; then
    echo "Error: Conda Python not found at $CONDA_PYTHON"
    exit 1
fi

POPULATION=2048
OUTPUT_PREFIX="registration_result"

echo "Using Python from conda environment: $CONDA_PYTHON"
echo ""
echo "Trying different optimization methods with population size: $POPULATION"
echo ""

# Methods to try in order
METHODS=("scipy_SLSQP" "scipy_LBFGSB" "nlopt_SBPLX" "nlopt_COBYLA")

for method in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Trying method: $method"
    echo "=========================================="
    
    OUTPUT_FILE="${OUTPUT_PREFIX}_${method}.json"
    
    "$CONDA_PYTHON" /scratch/users/Proulx-S/s2v/run_s2v_registration.py \
        --method "$method" \
        --loss MAE \
        --population "$POPULATION" \
        --backend numpy \
        --size 256 \
        --output "$OUTPUT_FILE"
    
    echo ""
    echo "Method $method completed. Results saved to $OUTPUT_FILE"
    echo ""
    
    # Visualize the result
    echo "Creating visualizations..."
    "$CONDA_PYTHON" /scratch/users/Proulx-S/s2v/visualize_registration.py \
        --result "$OUTPUT_FILE" \
        --output "registration_vis_${method}" 2>/dev/null || echo "  (Visualization skipped)"
    
    "$CONDA_PYTHON" /scratch/users/Proulx-S/s2v/visualize_slice_position.py \
        --result "$OUTPUT_FILE" \
        --output "slice_position_vis_${method}" 2>/dev/null || echo "  (Position visualization skipped)"
    
    echo ""
done

echo "=========================================="
echo "All methods completed!"
echo "=========================================="
echo "Check the output files to see which method gave the best results:"
for method in "${METHODS[@]}"; do
    echo "  - ${OUTPUT_PREFIX}_${method}.json"
done

