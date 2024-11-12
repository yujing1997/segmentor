#!/bin/bash
#SBATCH --account=def-senger
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --time=0:30:00
#SBATCH --output=/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/output_logs/segmentation_log_%j.log

module load StdEnv/2023
module load python/3.10.13  # Adjust version if needed
source ~/envs/semanticseg310/bin/activate

python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation.py \
    --task predict_wsi \
    --wsi_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Patient-Specific_Microdosimetry/DATABASE_GYN/TCGA_CESC/histopathology/svs_slides/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/TCGA-C5-A905-01Z-00-DX1.CD4B818F-C70D-40C9-9968-45DF22F1B149.svs \
    --save_dir /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/semantic_mask/tcga_cesc/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/wsi_segmentation_results2_0.2277mpp_40x \
    --mpp 0.2277 \
    # --on_gpu

# Create a timestamped log file
LOG_FILE="/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/output_logs/segmentation_log_$(date +'%Y%m%d_%H%M%S').log"

echo "Segmentation job completed. Logs saved to $LOG_FILE"
