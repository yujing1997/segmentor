#!/bin/bash
# Manuscript method figure

# Load necessary modules
module load StdEnv/2023
module load python/3.10.13
source ~/envs/semanticseg310/bin/activate

# Define variables for paths and parameters
BASE_SCRIPT_PATH="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/visualize_semantic_segmentation2.py"
BASE_NPY_PATH="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc"
BASE_WSI_PATH="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc"
CSV_FOLDER_PATH="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/unzipped_cesc_polygon"
BASE_OVERLAY_SAVE_PATH="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/manuscript_visualizations/tcga_cesc"

# Experiment parameters
EXPERIMENTS=(
    "2ca5c47d-120b-4f08-90e9-a9a345393bf1"
)

NUM_RANDOM_COMBOS=5  # Number of random combinations to pick
USE_MANUAL_VALUES=false #true  # Set to true to use manual values, false to auto-extract

# Manual values for X_START, Y_START, and PATCH_SIZE
MANUAL_X_START=68001
MANUAL_Y_START=28001
MANUAL_PATCH_SIZE=4000

# Function to extract random combinations of X_START, Y_START, and PATCH_SIZE from CSV filenames
get_random_combos() {
    local experiment_id=$1
    find "$CSV_FOLDER_PATH/${CASE_NAME}/cesc_polygon" -name "*.csv" | while read -r csv_file; do
        basename "$csv_file" | awk -F'_' '{print $1, $2, $3}'
    done | shuf -n "$NUM_RANDOM_COMBOS"
}

for EXPERIMENT_ID in "${EXPERIMENTS[@]}"
do
    # Define case-specific paths
    CASE_NAME="TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs"
    NPY_FILE_PATH="${BASE_NPY_PATH}/${EXPERIMENT_ID}/${CASE_NAME}/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
    WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
    OVERLAY_SAVE_FOLDER="${BASE_OVERLAY_SAVE_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"

    if [ "$USE_MANUAL_VALUES" = true ]; then
        echo "Using manual values for X_START, Y_START, and PATCH_SIZE"
        X_START="$MANUAL_X_START"
        Y_START="$MANUAL_Y_START"
        PATCH_SIZE="$MANUAL_PATCH_SIZE"

        OVERLAY_SAVE_PATH="${OVERLAY_SAVE_FOLDER}/wsi_semanticseg_sidebyside_image_${X_START}_${Y_START}_${PATCH_SIZE}.png"

        # Create directories if not exist
        mkdir -p "$OVERLAY_SAVE_FOLDER"

        # Run the Python script
        python "$BASE_SCRIPT_PATH" \
            --task segmentation_overlay \
            --npy_file_path "$NPY_FILE_PATH" \
            --wsi_path "$WSI_PATH" \
            --x_start "$X_START" \
            --y_start "$Y_START" \
            --patch_size "$PATCH_SIZE" \
            --overlay_save_path "$OVERLAY_SAVE_PATH" \
            --show_side_by_side

        echo "Completed processing for manual values: $X_START, $Y_START, $PATCH_SIZE"
    else
        echo "Picking $NUM_RANDOM_COMBOS random combinations for $EXPERIMENT_ID"
        RANDOM_COMBOS=$(get_random_combos "$EXPERIMENT_ID")
        while read -r X_START Y_START PATCH_SIZE; do
            OVERLAY_SAVE_PATH="${OVERLAY_SAVE_FOLDER}/wsi_semanticseg_sidebyside_image_${X_START}_${Y_START}_${PATCH_SIZE}.png"

            # Create directories if not exist
            mkdir -p "$OVERLAY_SAVE_FOLDER"

            # Run the Python script
            python "$BASE_SCRIPT_PATH" \
                --task segmentation_overlay \
                --npy_file_path "$NPY_FILE_PATH" \
                --wsi_path "$WSI_PATH" \
                --x_start "$X_START" \
                --y_start "$Y_START" \
                --patch_size "$PATCH_SIZE" \
                --overlay_save_path "$OVERLAY_SAVE_PATH" \
                --show_side_by_side

            echo "Completed processing for random combination: $X_START, $Y_START, $PATCH_SIZE"
        done <<< "$RANDOM_COMBOS"
    fi
done

echo "All experiments completed."



# Narval version 
# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/visualize_semantic_segmentation2.py \
#     --task segmentation_overlay \
#     --npy_file_path /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/2ca5c47d-120b-4f08-90e9-a9a345393bf1/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --wsi_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/2ca5c47d-120b-4f08-90e9-a9a345393bf1/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs \
#     --x_start 68001 \
#     --y_start 28001 \
#     --patch_size 4000 \
#     --overlay_save_path /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/manuscript_visualizations/tcga_cesc/2ca5c47d-120b-4f08-90e9-a9a345393bf1/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs/wsi_semanticseg_sidebyside_image_68001_28001_4000.png \
#     --show_side_by_side

# Choose this one from cesc_svs 
# 2ca5c47d-120b-4f08-90e9-a9a345393bf1
# TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs

# semantic mask:
# /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/2ca5c47d-120b-4f08-90e9-a9a345393bf1/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy
# wsi path:
# /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/2ca5c47d-120b-4f08-90e9-a9a345393bf1/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs
# cesc_polygon parent folder for this case: 
# /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/unzipped_cesc_polygon/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs/cesc_polygon/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs
# save path:
# /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/manuscript_visualizations/tcga_cesc/2ca5c47d-120b-4f08-90e9-a9a345393bf1/TCGA-C5-A1M6-01Z-00-DX1.13F7405D-AD0E-4A1C-9DF4-00DC90756D28.svs
