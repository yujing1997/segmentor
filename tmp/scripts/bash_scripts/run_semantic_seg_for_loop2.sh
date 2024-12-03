#!/bin/bash
TASK="predict_wsi"
# SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_Narval_YZ_Proton_filemap.tsv"
SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_to_be_rerun.tsv"
PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
PARENT_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask_qa"
# PARENT_SAVE_DIR="/media/yujing/Seagate/Segment/tcga_cesc_semantic_mask_qa"
ON_GPU="--on_gpu"
LOG_FILE="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/output_logs/segmentation_log_$(date +%Y%m%d_%H%M%S).log"
NUM_ITERATIONS=0  # Set to 0 to iterate over all rows, or set to a specific number to limit iterations

# conda activate segment2

# This bash script runs some subsets of WSI given a SAMPLE_SHEET, you'd have to make sure the wsi_path actually exists though from the paths build from the file id and file name of the SAMPLE_SHEET
# Here I did something quick and dirty where I manually separated the rows whose wsi_paths exists and those who didn't and need to be rerun (since the tcga-cesc was still downloading...)


# Start of the script log
echo "Starting WSI segmentation at $(date)" | tee -a "$LOG_FILE"
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



# # This version below works, but starts looping from the first row of the SAMPLE_SHEET again instead of exiting the for loop

# #!/bin/bash
# TASK="predict_wsi"
# # SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_Narval_YZ_Proton_filemap.tsv"
# SAMPLE_SHEET="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_to_be_rerun.tsv"
# PARENT_WSI_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_svs"
# # PARENT_SAVE_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask_qa"
# PARENT_SAVE_DIR="/media/yujing/Seagate/Segment/tcga_cesc_semantic_mask_qa"
# ON_GPU="--on_gpu"
# LOG_FILE="/Data/Yujing/Segment/tmp/tcga_cesc_manifest/output_logs/segmentation_log_$(date +%Y%m%d_%H%M%S).log"

# # Start of the script log
# echo "Starting WSI segmentation at $(date)" | tee -a "$LOG_FILE"
# echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
# echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
# echo "Parent save directory: $PARENT_SAVE_DIR" | tee -a "$LOG_FILE"
# echo "-------------------------------------------" | tee -a "$LOG_FILE"

# # Read the sample sheet into an array, skipping the header row
# mapfile -t lines < <(tail -n +2 "$SAMPLE_SHEET")

# # Loop over each line in the sample sheet
# for line in "${lines[@]}"; do
#     # Extract the File ID and File Name from each line (using tab as delimiter)
#     file_id=$(echo "$line" | cut -f1)
#     file_name=$(echo "$line" | cut -f2)
    
#     echo "Processing WSI: $file_name with File ID: $file_id" | tee -a "$LOG_FILE"
#     start_time=$(date +%s)

#     # Run the Python script for each WSI
#     python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation3.py \
#         --task "$TASK" \
#         --sample_sheet "$SAMPLE_SHEET" \
#         --parent_wsi_dir "$PARENT_WSI_DIR" \
#         --parent_save_dir "$PARENT_SAVE_DIR" \
#         $ON_GPU 2>&1 | tee -a "$LOG_FILE"

#     # Log the time taken for each WSI
#     end_time=$(date +%s)
#     duration=$((end_time - start_time))
#     echo "Completed $file_name in $duration seconds at $(date)" | tee -a "$LOG_FILE"
#     echo "-------------------------------------------" | tee -a "$LOG_FILE"
# done

# echo "Segmentation completed for all WSIs at $(date)" | tee -a "$LOG_FILE"