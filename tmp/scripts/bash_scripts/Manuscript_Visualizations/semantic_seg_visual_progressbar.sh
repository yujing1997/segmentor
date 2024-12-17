#!/bin/bash
# ============================================================================
# Manuscript Method Visualization Generation Script with Progress Bar
    # USE_RANDOM_COMBOS=false # manual X_START, Y_START, PATCH_SIZE selection
    # USE_RANDOM_COMBOS=true # random X_START, Y_START, PATCH_SIZE selection for visual inspection

# Author: Yujing Zou
# Date: Dec, 2024
# ============================================================================

# Load necessary modules
# conda activate segment2

# Define variables for paths and parameters
BASE_SCRIPT_PATH="/home/yujing/dockhome/Multimodality/Segment/tmp/scripts/visualize_semantic_segmentation2.py"
BASE_NPY_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask"
BASE_WSI_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
# BASE_OVERLAY_SAVE_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations"
# BASE_OVERLAY_SAVE_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/tmp/semantic_only"
BASE_OVERLAY_SAVE_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/semantic_seg"
CSV_PARENT_DIR="/Data/Yujing/Segment/tmp/unzipped_cesc_polygon"


# Experiment parameters
EXPERIMENTS=(
    "69037019-9df5-493e-8067-d4078d78e518"
    # "6edad00e-0e5b-42bc-a09d-ea81b1011c20"
)

CASE_NAME="TCGA-MA-AA3X-01Z-00-DX1.44657CDB-53F1-4DED-AE54-2251118565EA.svs"
# CASE_NAME="TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs"

NUM_RANDOM_COMBOS=5  # Number of random combinations to pick

# Toggle Variable
USE_RANDOM_COMBOS=false #true  # Set to 'false' for manual selection

# Define manual combinations (only used if USE_RANDOM_COMBOS=false)
# Format: "X_START Y_START PATCH_SIZE"
MANUAL_COMBOS=(
    "28001 68001 4000"
    "32001 56001 4000"
    "104001 44001 4000"
    # Add more combinations as needed
)

# Function to extract random combinations of X_START, Y_START, and PATCH_SIZE from .csv filenames
get_random_combos() {
    local experiment_id=$1
    local case_name=$2
    local csv_dir="${CSV_PARENT_DIR}/${case_name}/cesc_polygon/${case_name}"

    if [ -d "$csv_dir" ]; then
        COMBOS=$(find "$csv_dir" -name "*.csv" | while read -r csv_file; do
            filename=$(basename "$csv_file")
            echo "Parsing filename: $filename" >&2  # Redirect to stderr
            # Extract X_START, Y_START, PATCH_SIZE
            # Adjust the field numbers if necessary
            X=$(echo "$filename" | awk -F'[_.]' '{print $1}')
            Y=$(echo "$filename" | awk -F'[_.]' '{print $2}')
            PATCH=$(echo "$filename" | awk -F'[_.]' '{print $3}')
            echo "$X $Y $PATCH"
        done | shuf -n "$NUM_RANDOM_COMBOS")

        echo "Random combinations extracted:" >&2
        echo "$COMBOS" >&2

        echo "$COMBOS"  # Only echo combinations to stdout
    else
        echo "Directory $csv_dir does not exist. Skipping." >&2
        return 1
    fi
}

# Progress bar function
progress_bar() {
    local total=$1
    local current=$2
    local width=50
    local percent=0
    if [ "$total" -ne 0 ]; then
        percent=$((current * 100 / total))
    fi
    local filled=0
    if [ "$total" -ne 0 ]; then
        filled=$((width * current / total))
    fi
    local empty=$((width - filled))
    printf "\r[%-${width}s] %d%%" "$(printf "#%.0s" $(seq 1 "$filled"))" "$percent"
}

# Initialize total tasks based on selection mode
if [ "$USE_RANDOM_COMBOS" = true ]; then
    TOTAL_TASKS=$((NUM_RANDOM_COMBOS * ${#EXPERIMENTS[@]}))
else
    TOTAL_TASKS=$(( ${#MANUAL_COMBOS[@]} * ${#EXPERIMENTS[@]} ))
fi

CURRENT_TASK=0

for EXPERIMENT_ID in "${EXPERIMENTS[@]}"
do
    echo "Processing experiment: $EXPERIMENT_ID"
    
    if [ "$USE_RANDOM_COMBOS" = true ]; then
        RANDOM_COMBOS=$(get_random_combos "$EXPERIMENT_ID" "$CASE_NAME")
        if [ $? -ne 0 ]; then
            echo "Failed to get random combinations for experiment: $EXPERIMENT_ID" >&2
            continue
        fi
        echo "Random COMBOS:"
        echo "$RANDOM_COMBOS"
    else
        # Use manual combinations
        RANDOM_COMBOS="${MANUAL_COMBOS[@]}"
        echo "Manual COMBOS:"
        for combo in "${MANUAL_COMBOS[@]}"; do
            echo "$combo"
        done
    fi

    # Iterate through combinations
    if [ "$USE_RANDOM_COMBOS" = true ]; then
        # Read from RANDOM_COMBOS (only if random selection)
        while read -r X_START Y_START PATCH_SIZE; do
            # Define case-specific paths
            NPY_FILE_PATH="${BASE_NPY_PATH}/${EXPERIMENT_ID}/${CASE_NAME}/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
            WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
            OVERLAY_SAVE_DIR="${BASE_OVERLAY_SAVE_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
            OVERLAY_SAVE_PATH="${OVERLAY_SAVE_DIR}/wsi_semanticseg_sidebyside_image_${X_START}_${Y_START}_${PATCH_SIZE}_transposedmask.png"

            # Validate paths
            if [ ! -f "$NPY_FILE_PATH" ]; then
                echo "File $NPY_FILE_PATH not found. Skipping." >&2
                continue
            fi
            if [ ! -f "$WSI_PATH" ]; then
                echo "File $WSI_PATH not found. Skipping." >&2
                continue
            fi

            # Create directories if not exist
            mkdir -p "$(dirname "$OVERLAY_SAVE_PATH")"

            # Print chosen values
            echo "Processing with values: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE"

            # Check if variables are not empty and are integers
            if [[ -z "$X_START" || -z "$Y_START" || -z "$PATCH_SIZE" ]]; then
                echo "One or more parameters are empty. Skipping this combination." >&2
                continue
            fi

            if ! [[ "$X_START" =~ ^-?[0-9]+$ && "$Y_START" =~ ^-?[0-9]+$ && "$PATCH_SIZE" =~ ^-?[0-9]+$ ]]; then
                echo "One or more parameters are not valid integers. Skipping this combination." >&2
                continue
            fi

            # Run the Python script
            echo "Running Python script with parameters:"
            echo "--task segmentation_overlay"
            echo "--npy_file_path $NPY_FILE_PATH"
            echo "--wsi_path $WSI_PATH"
            echo "--x_start $X_START"
            echo "--y_start $Y_START"
            echo "--patch_size $PATCH_SIZE"
            echo "--overlay_save_path $OVERLAY_SAVE_PATH"
            # echo "--show_side_by_side"
            echo "--transpose_segmask"

            python "$BASE_SCRIPT_PATH" \
                --task segmentation_overlay \
                --npy_file_path "$NPY_FILE_PATH" \
                --wsi_path "$WSI_PATH" \
                --x_start "$X_START" \
                --y_start "$Y_START" \
                --patch_size "$PATCH_SIZE" \
                --overlay_save_path "$OVERLAY_SAVE_PATH" \
                # --show_side_by_side \
                --transpose_segmask

            if [ $? -ne 0 ]; then
                echo "Python script failed for combination: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE" >&2
                continue
            fi

            echo "Completed processing for combination: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE"
            CURRENT_TASK=$((CURRENT_TASK + 1))
            progress_bar "$TOTAL_TASKS" "$CURRENT_TASK"
        done <<< "$RANDOM_COMBOS"
    else
        # Use manual combinations
        for combo in "${MANUAL_COMBOS[@]}"; do
            read -r X_START Y_START PATCH_SIZE <<< "$combo"

            # Define case-specific paths
            NPY_FILE_PATH="${BASE_NPY_PATH}/${EXPERIMENT_ID}/${CASE_NAME}/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
            WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
            OVERLAY_SAVE_DIR="${BASE_OVERLAY_SAVE_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
            OVERLAY_SAVE_PATH="${OVERLAY_SAVE_DIR}/wsi_semanticseg_sidebyside_image_${X_START}_${Y_START}_${PATCH_SIZE}_transposedmask.png"

            # Validate paths
            if [ ! -f "$NPY_FILE_PATH" ]; then
                echo "File $NPY_FILE_PATH not found. Skipping." >&2
                continue
            fi
            if [ ! -f "$WSI_PATH" ]; then
                echo "File $WSI_PATH not found. Skipping." >&2
                continue
            fi

            # Create directories if not exist
            mkdir -p "$(dirname "$OVERLAY_SAVE_PATH")"

            # Print chosen values
            echo "Processing with values: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE"

            # Check if variables are not empty and are integers
            if [[ -z "$X_START" || -z "$Y_START" || -z "$PATCH_SIZE" ]]; then
                echo "One or more parameters are empty. Skipping this combination." >&2
                continue
            fi

            if ! [[ "$X_START" =~ ^-?[0-9]+$ && "$Y_START" =~ ^-?[0-9]+$ && "$PATCH_SIZE" =~ ^-?[0-9]+$ ]]; then
                echo "One or more parameters are not valid integers. Skipping this combination." >&2
                continue
            fi

            # Run the Python script
            echo "Running Python script with parameters:"
            echo "--task segmentation_overlay"
            echo "--npy_file_path $NPY_FILE_PATH"
            echo "--wsi_path $WSI_PATH"
            echo "--x_start $X_START"
            echo "--y_start $Y_START"
            echo "--patch_size $PATCH_SIZE"
            echo "--overlay_save_path $OVERLAY_SAVE_PATH"
            echo "--show_side_by_side"
            echo "--transpose_segmask"

            python "$BASE_SCRIPT_PATH" \
                --task segmentation_overlay \
                --npy_file_path "$NPY_FILE_PATH" \
                --wsi_path "$WSI_PATH" \
                --x_start "$X_START" \
                --y_start "$Y_START" \
                --patch_size "$PATCH_SIZE" \
                --overlay_save_path "$OVERLAY_SAVE_PATH" \
                --show_side_by_side \
                --transpose_segmask

            if [ $? -ne 0 ]; then
                echo "Python script failed for combination: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE" >&2
                continue
            fi

            echo "Completed processing for combination: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE"
            CURRENT_TASK=$((CURRENT_TASK + 1))
            progress_bar "$TOTAL_TASKS" "$CURRENT_TASK"
        done
    fi
done

printf "\nAll experiments completed.\n"

# ============================================================================
# END OF SCRIPT
# ============================================================================