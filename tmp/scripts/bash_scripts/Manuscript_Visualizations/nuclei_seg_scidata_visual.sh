#!/bin/bash
# ============================================================================
# Manuscript Method Visualization Generation Script with Progress Bar:
# Convert Polygon Annotations to Binary Masks for Multiple Patches
# Script: nuclei_seg_scidata_visual.sh
# Python Script used: polygon_to_masks.py
# Author: Yujing Zou
# Date: Dec, 2024
# Description: Automates the conversion of polygon annotations to binary masks
#              for selected patches either randomly or manually chosen.
# ============================================================================

# =============================================================================
# Load Necessary Modules or Activate Conda Environment
# =============================================================================

# Uncomment the following line if you need to activate a specific conda environment
# conda activate segment2

# =============================================================================
# Define Variables for Paths and Parameters
# =============================================================================

# Path to the Python script
BASE_SCRIPT_PATH="/home/yujing/dockhome/Multimodality/Segment/tmp/scripts/polygon_to_masks.py"

# Parent directory containing CSV files organized by case names
CSV_PARENT_DIR="/Data/Yujing/Segment/tmp/unzipped_cesc_polygon"  # Corrected based on your clarification

# Parent directory where output masks will be saved
BASE_OUTPUT_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/nuclei_binary_mask"

# Parent directory where Whole Slide Images (WSI) are stored
BASE_WSI_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_svs"

# List of experiment identifiers
EXPERIMENTS=(
    "69037019-9df5-493e-8067-d4078d78e518"
    # "6edad00e-0e5b-42bc-a09d-ea81b1011c20"
)

# Case name (must match the directory structure under CSV_PARENT_DIR and BASE_WSI_PATH)
CASE_NAME="TCGA-MA-AA3X-01Z-00-DX1.44657CDB-53F1-4DED-AE54-2251118565EA.svs"
# CASE_NAME="TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs"

# Number of random combinations to select (if in random mode)
NUM_RANDOM_COMBOS=5

# Toggle Variable
USE_RANDOM_COMBOS=false #true  # Set to 'false' for manual selection

# Define manual combinations (only used if USE_RANDOM_COMBOS=false)
# Format: "X_START Y_START PATCH_SIZE"
MANUAL_COMBOS=(
    "28001 68001 4000" #3949 nuclei in total for this patch
    "32001 56001 4000" #4440 nuclei in total for this patch
    "104001 44001 4000" #4333 nuclei in total for this patch
    # Add more combinations as needed
)

# =============================================================================
# Function Definitions
# =============================================================================

# Function to extract random combinations of X_START, Y_START, and PATCH_SIZE from .csv filenames
get_random_combos() {
    local experiment_id=$1
    local case_name=$2
    local csv_dir="${CSV_PARENT_DIR}/${case_name}/cesc_polygon/${case_name}"
    
    echo "Searching for CSV files in directory: $csv_dir" >&2  # Debug: Print the directory being searched
    
    if [ -d "$csv_dir" ]; then
        COMBOS=$(find "$csv_dir" -name "*.csv" | while read -r csv_file; do
            filename=$(basename "$csv_file")
            echo "Parsing filename: $filename" >&2  # Redirect to stderr for debugging
            # Extract X_START, Y_START, PATCH_SIZE
            # Assuming filename format: X_START_Y_START_PATCH_SIZE_<additional_info>-features.csv
            IFS='_' read -r X Y PATCH rest <<< "$filename"
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

# =============================================================================
# Initialize Total Tasks Based on Selection Mode
# =============================================================================

if [ "$USE_RANDOM_COMBOS" = true ]; then
    TOTAL_TASKS=$((NUM_RANDOM_COMBOS * ${#EXPERIMENTS[@]}))
else
    TOTAL_TASKS=$(( ${#MANUAL_COMBOS[@]} * ${#EXPERIMENTS[@]} ))
fi

CURRENT_TASK=0

# =============================================================================
# Main Processing Loop
# =============================================================================

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
            # Assuming WSI files are stored as [EXPERIMENT_ID]/[CASE_NAME]
            # Do not append '.svs' since CASE_NAME already includes it
            WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
    
            # Construct the expected CSV filename pattern
            # Updated to match filenames ending with '-features.csv'
            CSV_FILENAME="${X_START}_${Y_START}_${PATCH_SIZE}_*-features.csv"
    
            # Define the search directory for CSV
            CSV_SEARCH_DIR="${CSV_PARENT_DIR}/${CASE_NAME}/cesc_polygon/${CASE_NAME}"
    
            # Debug: Print the search directory and expected CSV filename
            echo "Looking for CSV file in: $CSV_SEARCH_DIR" >&2
            echo "Expected CSV filename pattern: $CSV_FILENAME" >&2
    
            # Find the exact CSV file matching the combination
            MATCHING_CSV=$(find "$CSV_SEARCH_DIR" -type f -name "$CSV_FILENAME")
    
            # Debug: Print the MATCHING_CSV path
            echo "MATCHING_CSV for combination (X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE): $MATCHING_CSV" >&2
    
            if [ -z "$MATCHING_CSV" ]; then
                echo "No matching CSV file found for combination: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE. Skipping." >&2
                continue
            fi
    
            # Construct the output mask path
            OUTPUT_MASK_PATH="${BASE_OUTPUT_PATH}/${EXPERIMENT_ID}/visualizations/patch_${X_START}_${Y_START}_${PATCH_SIZE}/${X_START}_${Y_START}_${PATCH_SIZE}_mask.png"
    
            # Validate WSI_PATH
            if [ ! -f "$WSI_PATH" ]; then
                echo "WSI file $WSI_PATH not found. Skipping." >&2
                continue
            fi
    
            # Create directories if not exist
            mkdir -p "$(dirname "$OUTPUT_MASK_PATH")"
    
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
            echo "--task polygon_to_mask"
            echo "--csv_path $MATCHING_CSV"
            echo "--output_path $OUTPUT_MASK_PATH"
            echo "--display"
            echo "--patch_size $PATCH_SIZE"
    
            python "$BASE_SCRIPT_PATH" \
                --task polygon_to_mask \
                --csv_path "$MATCHING_CSV" \
                --output_path "$OUTPUT_MASK_PATH" \
                --display \
                --patch_size "$PATCH_SIZE"
    
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
            # Assuming WSI files are stored as [EXPERIMENT_ID]/[CASE_NAME]
            # Do not append '.svs' since CASE_NAME already includes it
            WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
    
            # Construct the expected CSV filename pattern
            # Updated to match filenames ending with '-features.csv'
            CSV_FILENAME="${X_START}_${Y_START}_${PATCH_SIZE}_*-features.csv"
    
            # Define the search directory for CSV
            CSV_SEARCH_DIR="${CSV_PARENT_DIR}/${CASE_NAME}/cesc_polygon/${CASE_NAME}"
    
            # Debug: Print the search directory and expected CSV filename
            echo "Looking for CSV file in: $CSV_SEARCH_DIR" >&2
            echo "Expected CSV filename pattern: $CSV_FILENAME" >&2
    
            # Find the exact CSV file matching the combination
            MATCHING_CSV=$(find "$CSV_SEARCH_DIR" -type f -name "$CSV_FILENAME")
    
            # Debug: Print the MATCHING_CSV path
            echo "MATCHING_CSV for combination (X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE): $MATCHING_CSV" >&2
    
            if [ -z "$MATCHING_CSV" ]; then
                echo "No matching CSV file found for combination: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE. Skipping." >&2
                continue
            fi
    
            # Construct the output mask path
            OUTPUT_MASK_PATH="${BASE_OUTPUT_PATH}/${EXPERIMENT_ID}/visualizations/patch_${X_START}_${Y_START}_${PATCH_SIZE}/${X_START}_${Y_START}_${PATCH_SIZE}_mask.png"
    
            # Validate WSI_PATH
            if [ ! -f "$WSI_PATH" ]; then
                echo "WSI file $WSI_PATH not found. Skipping." >&2
                continue
            fi
    
            # Create directories if not exist
            mkdir -p "$(dirname "$OUTPUT_MASK_PATH")"
    
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
            echo "--task polygon_to_mask"
            echo "--csv_path $MATCHING_CSV"
            echo "--output_path $OUTPUT_MASK_PATH"
            echo "--display"
            echo "--patch_size $PATCH_SIZE"
    
            python "$BASE_SCRIPT_PATH" \
                --task polygon_to_mask \
                --csv_path "$MATCHING_CSV" \
                --output_path "$OUTPUT_MASK_PATH" \
                --display \
                --patch_size "$PATCH_SIZE"
    
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
