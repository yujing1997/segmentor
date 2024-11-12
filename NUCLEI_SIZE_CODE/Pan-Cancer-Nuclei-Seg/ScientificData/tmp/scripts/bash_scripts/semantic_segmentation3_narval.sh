#!/bin/bash
#SBATCH --job-name=wsi_segmentation
#SBATCH --output=/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/output_logs/segmentation_log_%A_%a.out
#SBATCH --error=/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/output_logs/segmentation_log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=05:00:00

# Calculate the number of array tasks based on the sample sheet row count (excluding header)
N_TASKS=$(( $(wc -l < /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/run_partition/run_Narval_YZ_filemap.tsv) - 1 ))

# Request the exact number of array tasks with a 20-task concurrency limit
#SBATCH --array=1-${N_TASKS}%20

# Load necessary modules
module load StdEnv/2023
module load python/3.10.13  # Adjust Python version if needed
source ~/envs/semanticseg310/bin/activate

# Define paths and arguments
TASK="predict_wsi"
SAMPLE_SHEET="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/run_partition/run_Narval_YZ_filemap.tsv"
PARENT_WSI_DIR="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc"
PARENT_SAVE_DIR="/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc"
ON_GPU="--on_gpu"
LOG_FILE="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/output_logs/segmentation_log_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$(date +%Y%m%d_%H%M%S).log"

# Extract the specific line from SAMPLE_SHEET for the current array job
IFS=$'\t' read -r file_id file_name _ < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SAMPLE_SHEET")

# Define paths for the specific WSI file
WSI_PATH="$PARENT_WSI_DIR/$file_id/$file_name"
SAVE_DIR="$PARENT_SAVE_DIR/$file_id/$file_name/wsi_segmentation_results2_0.2277mpp_40x"

# Check if WSI file exists
if [ ! -f "$WSI_PATH" ]; then
    echo "Error: WSI file $WSI_PATH does not exist. Skipping." | tee -a "$LOG_FILE"
    exit 1
fi

# Ensure SAVE_DIR exists
mkdir -p "$SAVE_DIR"

# Start logging
echo "Starting WSI segmentation for File ID: $file_id, File Name: $file_name at $(date)" | tee -a "$LOG_FILE"
echo "Sample sheet: $SAMPLE_SHEET" | tee -a "$LOG_FILE"
echo "Parent WSI directory: $PARENT_WSI_DIR" | tee -a "$LOG_FILE"
echo "Parent save directory: $PARENT_SAVE_DIR" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

# Run the Python script for the current WSI
start_time=$(date +%s)

python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation3.py \
    --task "$TASK" \
    --wsi_path "$WSI_PATH" \
    --save_dir "$SAVE_DIR" \
    $ON_GPU 2>&1 | tee -a "$LOG_FILE"

# Log completion time for this WSI
end_time=$(date +%s)
duration=$((end_time - start_time))
echo "Completed WSI: $file_name in $duration seconds at $(date)" | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"
