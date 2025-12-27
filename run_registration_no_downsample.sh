#!/bin/bash
# Run registration without downsampling - uses actual slice dimensions (400x400)

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate nh-rs2v

CONDA_PYTHON="/home/sebp/miniconda3/envs/nh-rs2v/bin/python3"

if [ ! -f "$CONDA_PYTHON" ]; then
    echo "Error: Conda Python not found at $CONDA_PYTHON"
    exit 1
fi

# Clear NH-RS2V coordinate grid cache (in case size changed)
# The cache will be automatically cleared by the Python script, but we'll clear it here too for safety
echo "Clearing coordinate grid cache..."
rm -f /tmp/g_s_slice /tmp/g_s_volume 2>/dev/null || true

echo "Using Python from conda environment: $CONDA_PYTHON"
echo ""
echo "Running registration WITHOUT downsampling..."
echo "This will use the actual slice dimensions (400x400) instead of 256x256"
echo "This preserves full resolution and may improve registration accuracy."
echo ""

# Run registration - size will be auto-detected from slice dimensions
"$CONDA_PYTHON" /scratch/users/Proulx-S/s2v/run_s2v_registration.py \
    --method scipy_SLSQP \
    --loss MAE \
    --population 2048 \
    --backend numpy \
    --output registration_result_no_downsample.json

echo ""
echo "Registration complete! Results saved to registration_result_no_downsample.json"
echo ""
echo "To visualize the results:"
echo "  python3 visualize_registration.py --result registration_result_no_downsample.json --size 400"
echo "  python3 visualize_slice_position.py --result registration_result_no_downsample.json --size 400"

