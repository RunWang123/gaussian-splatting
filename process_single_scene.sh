#!/bin/bash
#
# Process all cases for a single scene with vanilla 3D Gaussian Splatting
# Usage: bash process_single_scene.sh <scene_name> <json_split_path> <data_base_dir> <output_base_dir>
#
# Simplified version (no segmentation, no LSeg features)

# NOTE: We don't use 'set -e' here so that if one case fails,
# we continue processing other cases.

# ============================================================================
# Configuration
# ============================================================================

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <scene_name> <json_split_path> <data_base_dir> <output_base_dir>"
    echo ""
    echo "Example:"
    echo "  $0 5c8dafad7d /path/to/split.json /scratch/dl3dv_data /scratch/dl3dv_output"
    exit 1
fi

SCENE_NAME="$1"
JSON_SPLIT_PATH="$2"
DATA_BASE_DIR="$3"
OUTPUT_BASE_DIR="$4"

SCENE_DATA_DIR="${DATA_BASE_DIR}/${SCENE_NAME}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VANILLA_3DGS_DIR="${SCRIPT_DIR}"
FEATURE_3DGS_METRICS="$(dirname ${SCRIPT_DIR})/feature-3dgs/metrics.py"  # For depth evaluation

# Training parameters for vanilla 3DGS
ITERATIONS=30000
SAVE_ITERATIONS="7000 30000"
TEST_ITERATIONS="7000 30000"

# ============================================================================
# Validate inputs
# ============================================================================

if [ ! -d "${SCENE_DATA_DIR}" ]; then
    echo "❌ Error: Scene data directory not found: ${SCENE_DATA_DIR}"
    echo ""
    echo "Available scenes in ${DATA_BASE_DIR}:"
    ls -1 "${DATA_BASE_DIR}" 2>/dev/null | head -10 | sed 's/^/  - /'
    [ $(ls -1 "${DATA_BASE_DIR}" 2>/dev/null | wc -l) -gt 10 ] && echo "  ... (and more)"
    exit 1
fi

# Validate scene structure
if [ ! -d "${SCENE_DATA_DIR}/images" ] && [ ! -d "${SCENE_DATA_DIR}/sparse" ]; then
    echo "❌ Error: Scene directory doesn't have expected structure"
    echo "   Expected: ${SCENE_DATA_DIR}/images/ or ${SCENE_DATA_DIR}/sparse/"
    echo "   Found:"
    ls -la "${SCENE_DATA_DIR}" | sed 's/^/     /'
    exit 1
fi

if [ ! -f "${JSON_SPLIT_PATH}" ]; then
    echo "❌ Error: JSON split file not found: ${JSON_SPLIT_PATH}"
    exit 1
fi

# ============================================================================
# Extract number of cases from JSON
# ============================================================================

echo "========================================"
echo "Processing Scene: ${SCENE_NAME}"
echo "========================================"
echo "Data directory: ${SCENE_DATA_DIR}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "JSON split: ${JSON_SPLIT_PATH}"
echo ""

# Get number of cases for this scene
NUM_CASES=$(python3 -c "
import json
import sys

try:
    with open('${JSON_SPLIT_PATH}', 'r') as f:
        data = json.load(f)
    
    scene_name = '${SCENE_NAME}'
    
    # Handle different JSON formats
    if isinstance(data, dict) and 'scenes' in data:
        # Format: {'scenes': {'scene1': [...], 'scene2': [...]}}
        if scene_name in data['scenes']:
            print(len(data['scenes'][scene_name]))
        else:
            print(0)
    elif isinstance(data, dict):
        # Format: {'scene1': [...], 'scene2': [...]}
        if scene_name in data:
            print(len(data[scene_name]))
        else:
            print(0)
    else:
        # List format - can't get cases count
        print(1)  # Assume 1 case per scene
        
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    print(0)
" 2>&1)

# Check if NUM_CASES is a valid number
if ! [[ "${NUM_CASES}" =~ ^[0-9]+$ ]]; then
    echo "❌ Error: Invalid NUM_CASES value: ${NUM_CASES}"
    echo "   This might be a JSON parsing error"
    exit 1
fi

if [ "${NUM_CASES}" -eq 0 ]; then
    echo "❌ Error: Scene ${SCENE_NAME} not found in JSON or has no cases"
    exit 1
fi

echo "Found ${NUM_CASES} cases for scene ${SCENE_NAME}"
echo ""

# ============================================================================
# Process each case
# ============================================================================

SUCCESSFUL_CASES=0
FAILED_CASES=0

for CASE_ID in $(seq 0 $((NUM_CASES - 1))); do
    echo "========================================"
    echo "Processing Case ${CASE_ID} / ${NUM_CASES}"
    echo "========================================"
    
    CASE_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SCENE_NAME}_case${CASE_ID}"
    mkdir -p "${CASE_OUTPUT_DIR}"
    
    # Log file for this case
    CASE_LOG="${CASE_OUTPUT_DIR}/processing.log"
    echo "Log file: ${CASE_LOG}"
    echo ""
    
    # Track if this case has any errors
    CASE_STATUS_FILE="${CASE_OUTPUT_DIR}/.case_status"
    echo "0" > "${CASE_STATUS_FILE}"  # 0 = success, 1 = failed
    
    {
        echo "Starting case ${CASE_ID} at $(date)"
        
        # ----------------------------------------------------------------
        # Step 1: Train
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 1: Training (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        cd "${VANILLA_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        python train.py \
            -s "${SCENE_DATA_DIR}" \
            -m "${CASE_OUTPUT_DIR}" \
            --iterations ${ITERATIONS} \
            --save_iterations ${SAVE_ITERATIONS} \
            --test_iterations ${TEST_ITERATIONS} \
            --eval \
            --json_split_path "${JSON_SPLIT_PATH}" \
            --case_id ${CASE_ID} \
            2>&1
        
        TRAIN_STATUS=$?
        if [ ${TRAIN_STATUS} -ne 0 ]; then
            echo "❌ Training failed for case ${CASE_ID}"
            echo "1" > "${CASE_STATUS_FILE}"
            exit 1
        fi
        
        echo "✅ Training completed"
        
        # ----------------------------------------------------------------
        # Step 2: Render (all saved iterations)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 2: Rendering (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        cd "${VANILLA_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        # Render all saved iterations
        for ITER in ${SAVE_ITERATIONS}; do
            echo "Rendering iteration ${ITER}..."
            
            python render.py \
                -s "${SCENE_DATA_DIR}" \
                -m "${CASE_OUTPUT_DIR}" \
                --iteration ${ITER} \
                --skip_train \
                --json_split_path "${JSON_SPLIT_PATH}" \
                --case_id ${CASE_ID} \
                2>&1
            
            RENDER_STATUS=$?
            if [ ${RENDER_STATUS} -ne 0 ]; then
                echo "⚠️  Rendering failed for iteration ${ITER}"
            else
                echo "✅ Rendering iteration ${ITER} completed"
            fi
        done
        
        echo "✅ All renderings completed"
        
        # ----------------------------------------------------------------
        # Step 3: RGB Metrics (all saved iterations)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 3: RGB Metrics (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        cd "${VANILLA_3DGS_DIR}" || exit 1
        echo "Working directory: $(pwd)"
        
        echo "Computing RGB metrics (PSNR/SSIM/LPIPS)..."
        python metrics.py -m "${CASE_OUTPUT_DIR}" 2>&1
        
        METRICS_STATUS=$?
        if [ ${METRICS_STATUS} -ne 0 ]; then
            echo "⚠️  RGB metrics computation failed for case ${CASE_ID}"
        else
            echo "✅ RGB metrics computed"
            
            # Display results
            if [ -f "${CASE_OUTPUT_DIR}/results.json" ]; then
                echo ""
                echo "Results:"
                cat "${CASE_OUTPUT_DIR}/results.json"
                echo ""
            fi
        fi
        
        # ----------------------------------------------------------------
        # Step 4: Depth Metrics (if GT depth exists)
        # ----------------------------------------------------------------
        echo ""
        echo "----------------------------------------"
        echo "Step 4: Depth Metrics (Case ${CASE_ID})"
        echo "----------------------------------------"
        
        GT_DEPTH_DIR="${SCENE_DATA_DIR}/depths"
        if [ -d "${GT_DEPTH_DIR}" ]; then
            echo "✅ Found GT depth directory: ${GT_DEPTH_DIR}"
            echo "Evaluating depth metrics on TRAINING views (0.1m-100m range, median norm)..."
            
            # Use feature-3dgs metrics.py for depth evaluation
            cd "${VANILLA_3DGS_DIR}" || exit 1
            python "${FEATURE_3DGS_METRICS}" \
                -m "${CASE_OUTPUT_DIR}" \
                --eval_depth \
                --gt_depth_dir "${GT_DEPTH_DIR}" \
                --json_split_path "${JSON_SPLIT_PATH}" \
                --case_id ${CASE_ID} \
                2>&1
            
            DEPTH_STATUS=$?
            if [ ${DEPTH_STATUS} -ne 0 ]; then
                echo "⚠️  Depth metrics computation failed"
            else
                echo "✅ Depth metrics computed"
            fi
        else
            echo "ℹ️  No GT depth directory found: ${GT_DEPTH_DIR}"
            echo "   Skipping depth evaluation (dataset may not provide GT depth)"
        fi
        
        # ----------------------------------------------------------------
        # Case summary
        # ----------------------------------------------------------------
        echo ""
        echo "========================================="
        echo "Case ${CASE_ID} completed at $(date)"
        echo "========================================="
        
    } 2>&1 | tee "${CASE_LOG}"
    
    # Check if case succeeded
    CASE_STATUS=$(cat "${CASE_STATUS_FILE}" 2>/dev/null || echo "1")
    if [ "${CASE_STATUS}" = "0" ]; then
        SUCCESSFUL_CASES=$((SUCCESSFUL_CASES + 1))
        echo "✅ Case ${CASE_ID}: SUCCESS"
    else
        FAILED_CASES=$((FAILED_CASES + 1))
        echo "❌ Case ${CASE_ID}: FAILED (check log: ${CASE_LOG})"
    fi
    echo ""
    
done

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo "========================================"
echo "Scene ${SCENE_NAME} Processing Complete"
echo "========================================"
echo "Total cases: ${NUM_CASES}"
echo "Successful: ${SUCCESSFUL_CASES}"
echo "Failed: ${FAILED_CASES}"
echo ""
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo ""

if [ ${FAILED_CASES} -gt 0 ]; then
    echo "⚠️  Some cases failed. Check individual logs for details."
    exit 1
else
    echo "✅ All cases processed successfully!"
    exit 0
fi

