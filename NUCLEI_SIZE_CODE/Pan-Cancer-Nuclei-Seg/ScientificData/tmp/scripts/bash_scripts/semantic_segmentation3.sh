#!/bin/bash

# THIS BASH SCRIPT PATHS IS FOR PROTON LOCAL

# Define paths and arguments
TASK="predict_wsi"
SAMPLE_SHEET="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/run_partition/run_Narval_YZ_filemap.tsv"
PARENT_WSI_DIR="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc"
PARENT_SAVE_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc"
ON_GPU="--on_gpu"
LOG_FILE="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/output_logs/segmentation_log_$(date +%Y%m%d_%H%M%S).log"

# Start of the script log
echo "Starting WSI segmentation at $(date)" | tee -a "$LOG_FILE"
echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
echo "Parent save directory: $PARENT_SAVE_DIR" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Loop through each line in the sample sheet to extract File ID and File Name for logging
while IFS=$'\t' read -r file_id file_name _; do
    echo "Processing WSI: $file_name with File ID: $file_id" | tee -a "$LOG_FILE"
    start_time=$(date +%s)

    # Run the Python script for each WSI
    python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation3.py \
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

done < <(tail -n +2 "$SAMPLE_SHEET")  # Skip header row

echo "Segmentation completed for all WSIs at $(date)" | tee -a "$LOG_FILE"