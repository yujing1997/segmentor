#!/bin/bash

# Bash script to perform nucleus instance segmentation on multiple WSIs using nucleus_instance_seg.py

# Configuration
TASK="predict_wsi"
# SAMPLE_SHEET="/home/yujing/dockhome/Multimodality/Segment/tmp/manifest/run_4_instanceseg_2024_11_27_DxOnly.tsv"  # Update this path
SAMPLE_SHEET="/home/yujing/dockhome/Multimodality/Segment/tmp/manifest/run_4_instanceseg_2024_11_27_DxOnly_Proton.tsv"  # Update this path
PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
PARENT_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_svs_instance_seg"       # Update this path
ON_GPU="--on_gpu"                         # Include this flag to use GPU, remove if not needed
LOG_FILE="/Data/Yujing/Segment/tmp/tcga_svs_instance_seg/logs/instance_segmentation_log_$(date +%Y%m%d_%H%M%S).log"  # Update this path
NUM_ITERATIONS=0  # Set to 0 to process all entries, or specify a number to limit

# Ensure the save directory exists
mkdir -p "$PARENT_SAVE_DIR"

# Start of the script log
echo "Starting Nucleus Instance Segmentation at $(date)" | tee -a "$LOG_FILE"
echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
echo "Parent save directory: $PARENT_SAVE_DIR" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Read the sample sheet into an array, skipping the header row
mapfile -t lines < <(tail -n +2 "$SAMPLE_SHEET")

# Get the number of rows in the sample sheet
num_rows=${#lines[@]}
echo "Number of rows in sample sheet: $num_rows" | tee -a "$LOG_FILE"

# Determine the number of iterations
if [[ $NUM_ITERATIONS -eq 0 ]]; then
    iterations=$num_rows
else
    iterations=$NUM_ITERATIONS
fi
echo "Number of iterations: $iterations" | tee -a "$LOG_FILE"

# Loop over each line in the sample sheet using a range
for ((i=0; i<iterations; i++)); do
    line=${lines[$i]}
    # Extract the File ID and File Name from each line (using tab as delimiter)
    file_id=$(echo "$line" | cut -f1)
    file_name=$(echo "$line" | cut -f2)
    
    echo "Processing WSI: $file_name with File ID: $file_id" | tee -a "$LOG_FILE"
    start_time=$(date +%s)

    # Construct full paths
    wsi_path="$PARENT_WSI_DIR/$file_id/$file_name"
    save_dir="$PARENT_SAVE_DIR/$file_id/$file_name/wsi_instance_segmentation_results"

    # Check if WSI exists
    if [[ ! -f "$wsi_path" ]]; then
        echo "WSI file does not exist: $wsi_path. Skipping." | tee -a "$LOG_FILE"
        echo "-------------------------------------------" | tee -a "$LOG_FILE"
        continue
    fi

    # Run the Python script for each WSI
    python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nucleus_instance_seg.py \
        --task "$TASK" \
        --wsi_path "$wsi_path" \
        --save_dir "$save_dir" \
        $ON_GPU 2>&1 | tee -a "$LOG_FILE"

    # Log the time taken for each WSI
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Completed $file_name in $duration seconds at $(date)" | tee -a "$LOG_FILE"
    echo "-------------------------------------------" | tee -a "$LOG_FILE"

done

echo "Instance segmentation completed for all WSIs at $(date)" | tee -a "$LOG_FILE"
