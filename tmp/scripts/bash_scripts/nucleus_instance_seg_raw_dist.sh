#!/bin/bash

# Script: nucleus_instance_seg_raw_dist_batch.sh
# Description:
#   This script reads a sample sheet, constructs the necessary paths for each sample,
#   and runs the nucleus_instance_seg_raw_dist.py script for each sample.
#   It processes samples in batches to utilize available CPU resources efficiently.

# ==============================
# Define paths and arguments
# ==============================

# Path to the sample sheet
SAMPLE_SHEET="/home/yujing/dockhome/Multimodality/Segment/tmp/manifest/run_4_instanceseg_2024_11_27_DxOnly_Proton.tsv"

# Parent directories
PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
PARENT_PRED_INST_DIR="/Data/Yujing/Segment/tmp/tcga_svs_instance_seg"
OUTPUT_DIR="/Data/Yujing/Segment/tmp/Instance_Segmentation/nuclei_instance_classify_results"

# Path to the nucleus_instance_seg_raw_dist.py script
SCRIPT_PATH="/home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nucleus_instance_seg_raw_dist.py"

# Number of workers for parallel processing (adjust based on your CPU)
NUM_WORKERS=8

# Logging
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nucleus_instance_seg_raw_dist_log_$(date +%Y%m%d_%H%M%S).log"

# Start of the script log
echo "Starting nucleus instance segmentation raw distribution processing at $(date)" | tee -a "$LOG_FILE"
echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
echo "Parent pred_inst directory: $PARENT_PRED_INST_DIR" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# ==============================
# Read the sample sheet
# ==============================

# Read the sample sheet into an array, skipping the header row
mapfile -t lines < <(tail -n +2 "$SAMPLE_SHEET")

# Batch processing parameters
BATCH_SIZE=10  # Number of samples to process in each batch

# ==============================
# Function to process a single sample
# ==============================

process_sample() {
    local row="$1"
    local line="$2"

    # Extract fields from the line
    IFS=$'\t' read -r FILE_ID FILE_NAME DATA_CATEGORY DATA_TYPE PROJECT_ID CASE_ID SAMPLE_ID SAMPLE_TYPE <<< "$line"

    echo "Processing row $row: $FILE_NAME" | tee -a "$LOG_FILE"

    # Construct paths
    WSI_PATH="$PARENT_WSI_DIR/$FILE_ID/$FILE_NAME"
    PRED_INST_PATH="$PARENT_PRED_INST_DIR/$FILE_ID/$FILE_NAME/wsi_instance_segmentation_results/0.dat"

    # Verify that the WSI file exists
    if [ ! -f "$WSI_PATH" ]; then
        echo "Error: WSI file $WSI_PATH does not exist. Skipping sample." | tee -a "$LOG_FILE"
        return 1
    fi

    # Verify that the pred_inst file exists
    if [ ! -f "$PRED_INST_PATH" ]; then
        echo "Error: pred_inst file $PRED_INST_PATH does not exist. Skipping sample." | tee -a "$LOG_FILE"
        return 1
    fi

    # Construct the output directory (including subdirectories)
    # OUTPUT_SUBDIR="$OUTPUT_DIR/$FILE_ID/$FILE_NAME"
    OUTPUT_SUBDIR="$OUTPUT_DIR/$FILE_NAME"
    mkdir -p "$OUTPUT_SUBDIR"

    # Run the nucleus_instance_seg_raw_dist.py script
    start_time=$(date +%s)
    python -u "$SCRIPT_PATH" \
        --task process_wsi \
        --wsi_path "$WSI_PATH" \
        --pred_inst_path "$PRED_INST_PATH" \
        --output_dir "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1
    end_time=$(date +%s)

    echo "Completed processing for $FILE_NAME in $((end_time - start_time)) seconds." | tee -a "$LOG_FILE"
}

# ==============================
# Process samples in batches
# ==============================

total_samples=${#lines[@]}
echo "Total samples to process: $total_samples" | tee -a "$LOG_FILE"

for ((i = 0; i < total_samples; i += BATCH_SIZE)); do
    batch=("${lines[@]:i:BATCH_SIZE}")

    echo "Processing batch starting at index $i" | tee -a "$LOG_FILE"

    # Process each sample in the batch in parallel
    for ((j = 0; j < ${#batch[@]}; j++)); do
        line="${batch[j]}"
        process_sample $((i + j + 2)) "$line" &
    done

    # Wait for all processes in the current batch to complete
    wait
done

echo "All samples processed at $(date)" | tee -a "$LOG_FILE"
