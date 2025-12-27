#!/bin/bash
# Test different optimization methods with size=400 (no downsampling)
# Tries different population sizes to find what works best for each method

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

# Clear cache at start
echo "Clearing coordinate grid cache..."
rm -f /tmp/g_s_slice /tmp/g_s_volume 2>/dev/null || true
echo ""

# Test different methods with different population sizes
# Start with smaller populations to avoid numerical issues
METHODS=("scipy_SLSQP" "scipy_LBFGSB" "nlopt_SBPLX" "nlopt_COBYLA")
POPULATION_SIZES=(512 1024 2048)
SIZE=400

echo "=========================================="
echo "Testing optimization methods with size=$SIZE"
echo "=========================================="
echo ""

for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Testing method: $METHOD"
    echo "=========================================="
    
    # Try with increasing population sizes until one works
    SUCCESS=false
    BEST_POPULATION=""
    
    for POPULATION in "${POPULATION_SIZES[@]}"; do
        echo ""
        echo "Trying population size: $POPULATION"
        echo "----------------------------------------"
        
        OUTPUT_FILE="registration_result_${METHOD}_pop${POPULATION}_size${SIZE}.json"
        
        # Clear cache before each run
        rm -f /tmp/g_s_slice /tmp/g_s_volume 2>/dev/null || true
        
        # Run registration (suppress most output, only show errors and final result)
        if "$CONDA_PYTHON" /scratch/users/Proulx-S/s2v/run_s2v_registration.py \
            --method "$METHOD" \
            --loss MAE \
            --population "$POPULATION" \
            --backend numpy \
            --size "$SIZE" \
            --output "$OUTPUT_FILE" > "registration_log_${METHOD}_pop${POPULATION}.txt" 2>&1; then
            
            # Check if result is identity
            if python3 -c "
import json
import sys
try:
    r = json.load(open('$OUTPUT_FILE'))
    R = r['rotation_matrix']
    T = r['translation_vector']
    is_identity = all(abs(R[i][j] - (1 if i==j else 0)) < 0.01 for i in range(3) for j in range(3))
    is_zero = all(abs(t) < 0.01 for t in T)
    if is_identity and is_zero:
        sys.exit(1)  # Identity result
    else:
        sys.exit(0)  # Non-identity result
except Exception as e:
    sys.exit(1)  # Error reading file
" 2>/dev/null; then
                echo ""
                echo "✓ SUCCESS: Method $METHOD with population $POPULATION produced non-identity result!"
                echo "  Result saved to: $OUTPUT_FILE"
                SUCCESS=true
                BEST_POPULATION=$POPULATION
                break
            else
                echo ""
                echo "✗ Result is identity, trying next population size..."
            fi
        else
            echo ""
            echo "✗ Registration failed (check log: registration_log_${METHOD}_pop${POPULATION}.txt)"
            echo "  Trying next population size..."
        fi
    done
    
    if [ "$SUCCESS" = true ]; then
        echo ""
        echo "✓ Method $METHOD succeeded with population size: $BEST_POPULATION"
    else
        echo ""
        echo "⚠ WARNING: Method $METHOD failed with all population sizes tested"
    fi
    
    echo ""
done

echo "=========================================="
echo "Testing complete!"
echo "=========================================="
echo ""
echo "Summary of results:"
echo "-------------------"
for METHOD in "${METHODS[@]}"; do
    FOUND_SUCCESS=false
    for POPULATION in "${POPULATION_SIZES[@]}"; do
        OUTPUT_FILE="registration_result_${METHOD}_pop${POPULATION}_size${SIZE}.json"
        if [ -f "$OUTPUT_FILE" ]; then
            if python3 -c "
import json
r = json.load(open('$OUTPUT_FILE'))
R = r['rotation_matrix']
T = r['translation_vector']
is_identity = all(abs(R[i][j] - (1 if i==j else 0)) < 0.01 for i in range(3) for j in range(3))
is_zero = all(abs(t) < 0.01 for t in T)
exit(0 if (is_identity and is_zero) else 1)
" 2>/dev/null; then
                echo "  ✓ $METHOD (pop=$POPULATION): SUCCESS"
                FOUND_SUCCESS=true
                break
            fi
        fi
    done
    if [ "$FOUND_SUCCESS" = false ]; then
        echo "  ✗ $METHOD: FAILED (all population sizes)"
    fi
done

echo ""
echo "Best results are saved with pattern: registration_result_<METHOD>_pop<POPULATION>_size${SIZE}.json"
echo ""
echo "To visualize a result:"
echo "  python3 visualize_registration.py --result registration_result_<METHOD>_pop<POPULATION>_size${SIZE}.json --size $SIZE"
echo "  python3 visualize_slice_position.py --result registration_result_<METHOD>_pop<POPULATION>_size${SIZE}.json --size $SIZE"
