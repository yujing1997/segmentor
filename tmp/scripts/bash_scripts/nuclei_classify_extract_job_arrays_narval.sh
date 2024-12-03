#!/bin/bash

#SBATCH --job-name=nuclei_classify
#SBATCH --output=/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results/logs/nuclei_classify_log_%A_%a.out
#SBATCH --error=/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results/logs/nuclei_classify_log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=10:00:00
#SBATCH --array=1-192 # 230

# This bash script was submitted as an job array on Narval where each SLURM_ARRAY_TASK_ID is mapped to a row from the SAMPLE_SHEET 

# Load necessary modules
module load StdEnv/2023
module load python/3.10.13
source ~/envs/semanticseg310/bin/activate

# Define paths and arguments
SAMPLE_SHEET="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/manifest_files/gdc_sample_sheet.2024-11-11.tsv"
PARENT_WSI_DIR="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc"
PARENT_POLYGON_TAR_GZ_DIR="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon"
UNZIPPED_POLYGON_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/unzipped_cesc_polygon"
PRIMARY_SAVE_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc"
ALTERNATE_SAVE_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks2/tcga_cesc"
PARENT_SAVE_DIR_NUCLEI_CLASSIFY="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results"

LOG_DIR="$PARENT_SAVE_DIR_NUCLEI_CLASSIFY/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nuclei_classify_log_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-1}_$(date +%Y%m%d_%H%M%S).log"

# Function to process a single row
process_row() {
    local row=$1
    echo "Processing row: $row" | tee -a "$LOG_FILE"

    # Read the specified row from the sample sheet
    local line=$(sed -n "${row}p" "$SAMPLE_SHEET")
    echo "Read line from SAMPLE_SHEET: $line" | tee -a "$LOG_FILE"

    # Skip header line
    if [ "$row" -eq 1 ]; then
        echo "Skipping header line." | tee -a "$LOG_FILE"
        return 0
    fi

    # Extract fields from the line
    IFS=$'\t' read -r FILE_ID FILE_NAME DATA_CATEGORY DATA_TYPE PROJECT_ID CASE_ID SAMPLE_ID SAMPLE_TYPE <<< "$line"

    # Paths
    WSI_PATH="$PARENT_WSI_DIR/$FILE_ID/$FILE_NAME"
    POLYGON_TAR_GZ_FILE="$PARENT_POLYGON_TAR_GZ_DIR/${FILE_NAME}.tar.gz/${FILE_NAME}.tar.gz"
    POLYGON_DIR="$UNZIPPED_POLYGON_DIR/$FILE_NAME/cesc_polygon/$FILE_NAME"

    echo "Computed WSI_PATH: $WSI_PATH" | tee -a "$LOG_FILE"
    echo "Computed POLYGON_TAR_GZ_FILE: $POLYGON_TAR_GZ_FILE" | tee -a "$LOG_FILE"
    echo "Computed POLYGON_DIR: $POLYGON_DIR" | tee -a "$LOG_FILE"

    # Verify WSI file existence
    if [ ! -f "$WSI_PATH" ]; then
        echo "Error: WSI file $WSI_PATH does not exist. Skipping row $row." | tee -a "$LOG_FILE"
        return 1
    fi

    # Check if polygon directory exists and is non-empty; if not, proceed to extract
    if [ ! -d "$POLYGON_DIR" ] || [ -z "$(ls -A "$POLYGON_DIR" 2>/dev/null)" ]; then
        echo "Extracting polygon data for $FILE_NAME." | tee -a "$LOG_FILE"
        mkdir -p "$UNZIPPED_POLYGON_DIR/$FILE_NAME"
        if [ -f "$POLYGON_TAR_GZ_FILE" ]; then
            echo "Unzipping polygon data from $POLYGON_TAR_GZ_FILE to $UNZIPPED_POLYGON_DIR/$FILE_NAME" | tee -a "$LOG_FILE"
            tar -xzvf "$POLYGON_TAR_GZ_FILE" -C "$UNZIPPED_POLYGON_DIR/$FILE_NAME" | tee -a "$LOG_FILE"
            echo "Extraction completed." | tee -a "$LOG_FILE"
        else
            echo "Polygon tar.gz file $POLYGON_TAR_GZ_FILE does not exist. Skipping row $row." | tee -a "$LOG_FILE"
            return 1
        fi
    else
        echo "Polygon directory $POLYGON_DIR already exists and is not empty. Skipping extraction." | tee -a "$LOG_FILE"
    fi

    echo "Proceeding to determine segmentation output path." | tee -a "$LOG_FILE"

    # Determine the segmentation output path
    SEGMENTATION_OUTPUT_PATH="$PRIMARY_SAVE_DIR/$FILE_ID/$FILE_NAME/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
    if [ ! -f "$SEGMENTATION_OUTPUT_PATH" ]; then
        SEGMENTATION_OUTPUT_PATH="$ALTERNATE_SAVE_DIR/$FILE_ID/$FILE_NAME/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
    fi

    echo "Computed SEGMENTATION_OUTPUT_PATH: $SEGMENTATION_OUTPUT_PATH" | tee -a "$LOG_FILE"

    if [ ! -f "$SEGMENTATION_OUTPUT_PATH" ]; then
        echo "Segmentation output not found for $FILE_NAME. Skipping row $row." | tee -a "$LOG_FILE"
        return 1
    fi

    OUTPUT_DIR="$PARENT_SAVE_DIR_NUCLEI_CLASSIFY/$FILE_NAME"
    mkdir -p "$OUTPUT_DIR"
    echo "Created OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$LOG_FILE"

    echo "Proceeding to run nuclei classification Python script." | tee -a "$LOG_FILE"

    # Start processing and log the details
    echo "Starting nuclei classification for File ID: $FILE_ID, File Name: $FILE_NAME at $(date)" | tee -a "$LOG_FILE"
    echo "WSI Path: $WSI_PATH" | tee -a "$LOG_FILE"
    echo "Polygon Directory: $POLYGON_DIR" | tee -a "$LOG_FILE"
    echo "Segmentation Output Path: $SEGMENTATION_OUTPUT_PATH" | tee -a "$LOG_FILE"
    echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"

    # Run the Python script with unbuffered output
    start_time=$(date +%s)
    python -u /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/nuclei_classify_wsi.py \
        --task process_wsi \
        --wsi_path "$WSI_PATH" \
        --csv_dir "$POLYGON_DIR" \
        --segmentation_output_path "$SEGMENTATION_OUTPUT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --transpose_segmask \
        --parallel \
        --num_workers 16 2>&1 | tee -a "$LOG_FILE"
    echo "Python script execution completed." | tee -a "$LOG_FILE"

    # Log the completion time
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Completed nuclei classification for WSI: $FILE_NAME in $duration seconds at $(date)" | tee -a "$LOG_FILE"
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
}

# Process the specific row assigned to this job array instance
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" | tee -a "$LOG_FILE"
    process_row $((SLURM_ARRAY_TASK_ID + 1))
else
    echo "SLURM_ARRAY_TASK_ID is not set. Exiting script." | tee -a "$LOG_FILE"
fi

echo "Processing complete." | tee -a "$LOG_FILE"



#!/bin/bash
# BELOW WORKS! JUST NOT ACCOUNTING FOR the class name and cannot parallel process multiple csvs at once

#SBATCH --job-name=nuclei_classify
#SBATCH --output=/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results/logs/nuclei_classify_log_%A_%a.out
#SBATCH --error=/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results/logs/nuclei_classify_log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --array=1-100

# # Load necessary modules
# module load StdEnv/2023
# module load python/3.10.13
# source ~/envs/semanticseg310/bin/activate

# # Define paths and arguments
# SAMPLE_SHEET="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/manifest_files/gdc_sample_sheet.2024-11-11.tsv"
# PARENT_WSI_DIR="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc"
# PARENT_POLYGON_TAR_GZ_DIR="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon"
# UNZIPPED_POLYGON_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/unzipped_cesc_polygon"
# PRIMARY_SAVE_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc"
# ALTERNATE_SAVE_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks2/tcga_cesc"
# PARENT_SAVE_DIR_NUCLEI_CLASSIFY="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results"

# LOG_DIR="$PARENT_SAVE_DIR_NUCLEI_CLASSIFY/logs"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/nuclei_classify_log_${SLURM_ARRAY_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-1}_$(date +%Y%m%d_%H%M%S).log"

# # Function to process a single row
# process_row() {
#     local row=$1
#     echo "Processing row: $row" | tee -a "$LOG_FILE"

#     # Read the specified row from the sample sheet
#     local line=$(sed -n "${row}p" "$SAMPLE_SHEET")
#     echo "Read line from SAMPLE_SHEET: $line" | tee -a "$LOG_FILE"

#     # Skip header line
#     if [ "$row" -eq 1 ]; then
#         echo "Skipping header line." | tee -a "$LOG_FILE"
#         return 0
#     fi

#     # Extract fields from the line
#     IFS=$'\t' read -r FILE_ID FILE_NAME DATA_CATEGORY DATA_TYPE PROJECT_ID CASE_ID SAMPLE_ID SAMPLE_TYPE <<< "$line"

#     # Paths
#     WSI_PATH="$PARENT_WSI_DIR/$FILE_ID/$FILE_NAME"
#     POLYGON_TAR_GZ_FILE="$PARENT_POLYGON_TAR_GZ_DIR/${FILE_NAME}.tar.gz/${FILE_NAME}.tar.gz"
#     POLYGON_DIR="$UNZIPPED_POLYGON_DIR/$FILE_NAME/cesc_polygon/$FILE_NAME"

#     echo "Computed WSI_PATH: $WSI_PATH" | tee -a "$LOG_FILE"
#     echo "Computed POLYGON_TAR_GZ_FILE: $POLYGON_TAR_GZ_FILE" | tee -a "$LOG_FILE"
#     echo "Computed POLYGON_DIR: $POLYGON_DIR" | tee -a "$LOG_FILE"

#     # Verify WSI file existence
#     if [ ! -f "$WSI_PATH" ]; then
#         echo "Error: WSI file $WSI_PATH does not exist. Skipping row $row." | tee -a "$LOG_FILE"
#         return 1
#     fi

#     # Check if polygon directory exists and is non-empty; if not, proceed to extract
#     if [ ! -d "$POLYGON_DIR" ] || [ -z "$(ls -A "$POLYGON_DIR" 2>/dev/null)" ]; then
#         echo "Extracting polygon data for $FILE_NAME." | tee -a "$LOG_FILE"
#         mkdir -p "$UNZIPPED_POLYGON_DIR/$FILE_NAME"
#         if [ -f "$POLYGON_TAR_GZ_FILE" ]; then
#             echo "Unzipping polygon data from $POLYGON_TAR_GZ_FILE to $UNZIPPED_POLYGON_DIR/$FILE_NAME" | tee -a "$LOG_FILE"
#             tar -xzvf "$POLYGON_TAR_GZ_FILE" -C "$UNZIPPED_POLYGON_DIR/$FILE_NAME" | tee -a "$LOG_FILE"
#             echo "Extraction completed." | tee -a "$LOG_FILE"
#         else
#             echo "Polygon tar.gz file $POLYGON_TAR_GZ_FILE does not exist. Skipping row $row." | tee -a "$LOG_FILE"
#             return 1
#         fi
#     else
#         echo "Polygon directory $POLYGON_DIR already exists and is not empty. Skipping extraction." | tee -a "$LOG_FILE"
#     fi

#     echo "Proceeding to determine segmentation output path." | tee -a "$LOG_FILE"

#     # Determine the segmentation output path
#     SEGMENTATION_OUTPUT_PATH="$PRIMARY_SAVE_DIR/$FILE_ID/$FILE_NAME/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
#     if [ ! -f "$SEGMENTATION_OUTPUT_PATH" ]; then
#         SEGMENTATION_OUTPUT_PATH="$ALTERNATE_SAVE_DIR/$FILE_ID/$FILE_NAME/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy"
#     fi

#     echo "Computed SEGMENTATION_OUTPUT_PATH: $SEGMENTATION_OUTPUT_PATH" | tee -a "$LOG_FILE"

#     if [ ! -f "$SEGMENTATION_OUTPUT_PATH" ]; then
#         echo "Segmentation output not found for $FILE_NAME. Skipping row $row." | tee -a "$LOG_FILE"
#         return 1
#     fi

#     OUTPUT_DIR="$PARENT_SAVE_DIR_NUCLEI_CLASSIFY/$FILE_NAME"
#     mkdir -p "$OUTPUT_DIR"
#     echo "Created OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$LOG_FILE"

#     echo "Proceeding to run nuclei classification Python script." | tee -a "$LOG_FILE"

#     # Start processing and log the details
#     echo "Starting nuclei classification for File ID: $FILE_ID, File Name: $FILE_NAME at $(date)" | tee -a "$LOG_FILE"
#     echo "WSI Path: $WSI_PATH" | tee -a "$LOG_FILE"
#     echo "Polygon Directory: $POLYGON_DIR" | tee -a "$LOG_FILE"
#     echo "Segmentation Output Path: $SEGMENTATION_OUTPUT_PATH" | tee -a "$LOG_FILE"
#     echo "Output Directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"

#     # Run the Python script with unbuffered output and without parallel processing for debugging
#     start_time=$(date +%s)
#     python -u /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/nuclei_classify_wsi.py \
#         --task process_wsi \
#         --wsi_path "$WSI_PATH" \
#         --csv_dir "$POLYGON_DIR" \
#         --segmentation_output_path "$SEGMENTATION_OUTPUT_PATH" \
#         --output_dir "$OUTPUT_DIR" \
#         --transpose_segmask 2>&1 | tee -a "$LOG_FILE" \
#         # --parallel 
#     echo "Python script execution completed." | tee -a "$LOG_FILE"

#     # Log the completion time
#     end_time=$(date +%s)
#     duration=$((end_time - start_time))
#     echo "Completed nuclei classification for WSI: $FILE_NAME in $duration seconds at $(date)" | tee -a "$LOG_FILE"
#     echo "-------------------------------------------" | tee -a "$LOG_FILE"
# }

# # Process the specific row assigned to this job array instance
# if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
#     echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID" | tee -a "$LOG_FILE"
#     process_row $((SLURM_ARRAY_TASK_ID + 1))
# else
#     echo "SLURM_ARRAY_TASK_ID is not set. Exiting script." | tee -a "$LOG_FILE"
# fi

# echo "Processing complete." | tee -a "$LOG_FILE"
