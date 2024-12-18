#!/bin/bash

# Define variables for paths and parameters
BASE_SCRIPT_PATH="/home/yujing/dockhome/Multimodality/Segment/tmp/scripts/bash_scripts/Manuscript_Visualizations/WSI_visual.py"
BASE_WSI_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
BASE_OUTPUT_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/full_wsi_mask_patch"

# Experiment parameters
EXPERIMENTS=(
    "69037019-9df5-493e-8067-d4078d78e518"
    # Add more experiments here if needed
)

CASE_NAME="TCGA-MA-AA3X-01Z-00-DX1.44657CDB-53F1-4DED-AE54-2251118565EA.svs"
POWER=1.25

# Define manual combinations of X_START, Y_START, PATCH_SIZE
MANUAL_COMBOS=(
    "28001 68001 4000"
    "32001 56001 4000"
    "104001 44001 4000"
    # Add more combinations as needed
)

# Loop through experiments
for EXPERIMENT_ID in "${EXPERIMENTS[@]}"
do
    # Construct the WSI_PATH and OUTPUT_PREFIX correctly
    WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
    OUTPUT_PREFIX="${BASE_OUTPUT_PATH}/${EXPERIMENT_ID}/${CASE_NAME%.svs}"

    echo "Running visualization for experiment: $EXPERIMENT_ID"
    echo "WSI Path: $WSI_PATH"

    # Check if WSI_PATH exists
    if [ ! -f "$WSI_PATH" ]; then
        echo "Error: WSI file not found at $WSI_PATH. Skipping experiment."
        continue
    fi

    # Convert combos into individual arguments
    COMBO_ARGS=()
    for COMBO in "${MANUAL_COMBOS[@]}"; do
        COMBO_ARGS+=("$COMBO")
    done

    # Run the visualization script
    python "$BASE_SCRIPT_PATH" \
        --wsi_path "$WSI_PATH" \
        --output_prefix "$OUTPUT_PREFIX" \
        --power "$POWER" \
        --combos "${COMBO_ARGS[@]}"

done

printf "\nAll experiments completed.\n"


# #!/bin/bash

# # Define variables for paths and parameters
# BASE_SCRIPT_PATH="/home/yujing/dockhome/Multimodality/Segment/tmp/scripts/bash_scripts/Manuscript_Visualizations/WSI_visual.py"
# BASE_WSI_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
# BASE_OUTPUT_PATH="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/full_wsi_mask_patch"

# # Experiment parameters
# EXPERIMENTS=(
#     "69037019-9df5-493e-8067-d4078d78e518"
#     # Add more experiments here if needed
# )

# CASE_NAME="TCGA-MA-AA3X-01Z-00-DX1.44657CDB-53F1-4DED-AE54-2251118565EA.svs"
# POWER=1.25

# # Define manual combinations of X_START, Y_START, PATCH_SIZE
# MANUAL_COMBOS=(
#     "28001 68001 4000"
#     "32001 56001 4000"
#     "104001 44001 4000"
#     # Add more combinations as needed
# )

# # Progress bar function
# progress_bar() {
#     local total=$1
#     local current=$2
#     local width=50
#     local percent=$((current * 100 / total))
#     local filled=$((width * current / total))
#     local empty=$((width - filled))
#     printf "\r[%-${width}s] %d%%" "$(printf "#%.0s" $(seq 1 $filled))" $percent
# }

# # Loop through experiments
# for EXPERIMENT_ID in "${EXPERIMENTS[@]}"
# do
#     # Construct the WSI_PATH and OUTPUT_PREFIX correctly
#     WSI_PATH="${BASE_WSI_PATH}/${EXPERIMENT_ID}/${CASE_NAME}"
#     OUTPUT_PREFIX="${BASE_OUTPUT_PATH}/${EXPERIMENT_ID}/${CASE_NAME%.svs}"

#     echo "Running visualization for experiment: $EXPERIMENT_ID"
#     echo "WSI Path: $WSI_PATH"

#     # Check if WSI_PATH exists
#     if [ ! -f "$WSI_PATH" ]; then
#         echo "Error: WSI file not found at $WSI_PATH. Skipping experiment."
#         continue
#     fi

#     TOTAL_COMBOS=${#MANUAL_COMBOS[@]}
#     CURRENT_COMBO=0

#     # Loop through manual combinations
#     for COMBO in "${MANUAL_COMBOS[@]}"
#     do
#         read -r X_START Y_START PATCH_SIZE <<< "$COMBO"
#         echo "Processing with values: X_START=$X_START, Y_START=$Y_START, PATCH_SIZE=$PATCH_SIZE"

#         # Run the visualization script
#         python "$BASE_SCRIPT_PATH" \
#             --wsi_path "$WSI_PATH" \
#             --output_prefix "$OUTPUT_PREFIX" \
#             --x_start "$X_START" \
#             --y_start "$Y_START" \
#             --patch_size "$PATCH_SIZE" \
#             --power "$POWER"

#         # Update progress bar
#         CURRENT_COMBO=$((CURRENT_COMBO + 1))
#         progress_bar $TOTAL_COMBOS $CURRENT_COMBO
#     done
#     printf "\n"
# done

# printf "\nAll experiments completed.\n"
