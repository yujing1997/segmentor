#!/bin/bash

# For-loop version of nuclei_classify_extract_job_arrays_narval.sh with parallel processing
# Processes each row of the SAMPLE_SHEET in batches to maximize CPU usage.

# conda activate segment2

# Define paths and arguments
SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_to_be_rerun.tsv"
PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
PARENT_POLYGON_TAR_GZ_DIR="/Data/Yujing/Segment/tmp/cesc_polygon"
UNZIPPED_POLYGON_DIR="/Data/Yujing/Segment/tmp/unzipped_cesc_polygon"
PRIMARY_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask"
ALTERNATE_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask_qa"
PARENT_SAVE_DIR_NUCLEI_CLASSIFY="/Data/Yujing/Segment/tmp/nuclei_classify_results"

LOG_DIR="$PARENT_SAVE_DIR_NUCLEI_CLASSIFY/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nuclei_classify_log_$(date +%Y%m%d_%H%M%S).log"

# Set batch size for parallel processing
BATCH_SIZE=10  # Adjust based on available resources

# Start of the script log
echo "Starting nuclei classification at $(date)" | tee -a "$LOG_FILE"
echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
echo "Parent save directory: $PARENT_SAVE_DIR_NUCLEI_CLASSIFY" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Read the sample sheet into an array, skipping the header row
mapfile -t lines < <(tail -n +2 "$SAMPLE_SHEET")

# Function to process a single row
process_row() {
    local row="$1"
    local line="$2"

    echo "Processing row: $row" | tee -a "$LOG_FILE"
    echo "Read line from SAMPLE_SHEET: $line" | tee -a "$LOG_FILE"

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

    start_time=$(date +%s)
    python -u /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify_wsi.py \
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

# Loop through each line of the sample sheet in batches
row=1
for ((i = 0; i < ${#lines[@]}; i += BATCH_SIZE)); do
    # Launch BATCH_SIZE rows in parallel
    for ((j = 0; j < BATCH_SIZE && (i + j) < ${#lines[@]}; j++)); do
        line="${lines[i + j]}"
        ((row++))
        process_row "$row" "$line" &
    done
    wait  # Wait for all background processes in the batch to complete
done

echo "Processing complete for all rows at $(date)" | tee -a "$LOG_FILE"

# #!/bin/bash

# # For-loop version of nuclei_classify_extract_job_arrays_narval.sh
# # Runs locally without SLURM job arrays but processes each row of the SAMPLE_SHEET.

# # Define paths and arguments
# SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_to_be_rerun.tsv"
# PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
# PARENT_POLYGON_TAR_GZ_DIR="/Data/Yujing/Segment/tmp/cesc_polygon"
# UNZIPPED_POLYGON_DIR="/Data/Yujing/Segment/tmp/unzipped_cesc_polygon"
# PRIMARY_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask"
# ALTERNATE_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask_qa"
# PARENT_SAVE_DIR_NUCLEI_CLASSIFY="/Data/Yujing/Segment/tmp/nuclei_classify_results"

# LOG_DIR="$PARENT_SAVE_DIR_NUCLEI_CLASSIFY/logs"
# mkdir -p "$LOG_DIR"
# LOG_FILE="$LOG_DIR/nuclei_classify_log_$(date +%Y%m%d_%H%M%S).log"

# # Start of the script log
# echo "Starting nuclei classification at $(date)" | tee -a "$LOG_FILE"
# echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
# echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
# echo "Parent save directory: $PARENT_SAVE_DIR_NUCLEI_CLASSIFY" | tee -a "$LOG_FILE"
# echo "-------------------------------------------" | tee -a "$LOG_FILE"

# # Read the sample sheet into an array, skipping the header row
# mapfile -t lines < <(tail -n +2 "$SAMPLE_SHEET")

# # Function to process a single row
# process_row() {
#     local row="$1"
#     local line="$2"

#     echo "Processing row: $row" | tee -a "$LOG_FILE"
#     echo "Read line from SAMPLE_SHEET: $line" | tee -a "$LOG_FILE"

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

#     start_time=$(date +%s)
#     python -u /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify_wsi.py \
#         --task process_wsi \
#         --wsi_path "$WSI_PATH" \
#         --csv_dir "$POLYGON_DIR" \
#         --segmentation_output_path "$SEGMENTATION_OUTPUT_PATH" \
#         --output_dir "$OUTPUT_DIR" \
#         --transpose_segmask \
#         --parallel \
#         --num_workers 16 2>&1 | tee -a "$LOG_FILE"
#     echo "Python script execution completed." | tee -a "$LOG_FILE"

#     # Log the completion time
#     end_time=$(date +%s)
#     duration=$((end_time - start_time))
#     echo "Completed nuclei classification for WSI: $FILE_NAME in $duration seconds at $(date)" | tee -a "$LOG_FILE"
#     echo "-------------------------------------------" | tee -a "$LOG_FILE"
# }

# # Loop through each line of the sample sheet
# row=1
# for line in "${lines[@]}"; do
#     ((row++))
#     process_row "$row" "$line"
# done

# echo "Processing complete for all rows at $(date)" | tee -a "$LOG_FILE"
