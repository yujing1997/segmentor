
#!/bin/bash

# THIS BASH SCRIPT PATHS IS FOR THE PROTON GPUS (LOCAL)
# Modified from run_semantic_seg.sh for loop such that each row of the sample sheets are not repeated! 

# Define paths and arguments
TASK="predict_wsi"
SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_Proton_filemap.tsv"
PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
# PARENT_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask"
PARENT_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask_qa"
ON_GPU="--on_gpu"
LOG_FILE="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/output_logs/segmentation_log_$(date +%Y%m%d_%H%M%S).log"

# Start of the script log
echo "Starting WSI segmentation at $(date)" | tee -a "$LOG_FILE"
echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
echo "Parent save directory: $PARENT_SAVE_DIR" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Read the sample sheet into an array, skipping the header row
mapfile -t lines < <(tail -n +2 "$SAMPLE_SHEET")

# Loop over each line in the sample sheet
for line in "${lines[@]}"; do
    # Extract the File ID and File Name from each line (using tab as delimiter)
    file_id=$(echo "$line" | cut -f1)
    file_name=$(echo "$line" | cut -f2)
    
    echo "Processing WSI: $file_name with File ID: $file_id" | tee -a "$LOG_FILE"
    start_time=$(date +%s)

    # Run the Python script for each WSI
    python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation3.py \
        --task "$TASK" \
        --sample_sheet "$SAMPLE_SHEET" \
        --parent_wsi_dir "$PARENT_WSI_DIR" \
        --parent_save_dir "$PARENT_SAVE_DIR" \
        $ON_GPU 2>&1 | tee -a "$LOG_FILE"

    # Log the time taken for each WSI
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Completed $file_name in $duration seconds at $(date)" | tee -a "$LOG_FILE"
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
done

echo "Segmentation completed for all WSIs at $(date)" | tee -a "$LOG_FILE"