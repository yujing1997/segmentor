"""
Semantic segmentation via TiaToolbox

Proton GPU 

Example usage: can dynamically loop through file names from a directory
python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation.py \
    --task predict_wsi \
    --sample_sheet /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/manifest_files/gdc_sample_sheet.2024-11-11.tsv \
    --parent_wsi_dir /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc \
    --parent_save_dir /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc \
    --mpp 0.2277 \
    --on_gpu
- previous version: directly passing through the specific wsi_path and save_dir
    - see semantic_segmentation2.py

Yujing Zou
"""
"""
Semantic segmentation via TiaToolbox

Proton GPU 

Example usage:
python semantic_segmentation3.py \
    --task predict_wsi \
    --wsi_path /path/to/wsi/file.svs \
    --save_dir /path/to/save/directory \
    --on_gpu
"""

import os
import shutil
import logging
import warnings
import argparse
import time
import pandas as pd
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox import logger

# Suppress less critical warnings
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

class WSISemanticSegmentor:
    def __init__(self, wsi_path, save_dir, on_gpu=True):
        self.wsi_path = wsi_path
        self.save_dir = save_dir
        self.on_gpu = on_gpu
        self.mpp = self._get_mpp_from_wsi()
        self.segmentor = self._initialize_segmentor()
        self.ioconfig = self._create_ioconfig()

        # Set tiatoolbox logger to log only errors
        logger.setLevel(logging.ERROR)

    def _get_mpp_from_wsi(self):
        # Use WSIReader to obtain mpp value from the WSI file
        wsi_reader = WSIReader.open(self.wsi_path)
        mpp = wsi_reader.info.mpp[0]  # Assuming square pixels, take the first element for mpp
        print(f"Using mpp value from WSI: {mpp}")
        return mpp

    def _initialize_segmentor(self):
        return SemanticSegmentor(
            pretrained_model="fcn_resnet50_unet-bcss",
            num_loader_workers=48,
            batch_size=32,
            auto_generate_mask=True,
            verbose=False
        )

    def _create_ioconfig(self):
        return IOSegmentorConfig(
            input_resolutions=[{"units": "mpp", "resolution": self.mpp}],
            output_resolutions=[{"units": "mpp", "resolution": self.mpp}],
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            save_resolution={"units": "mpp", "resolution": self.mpp},
        )

    def predict_wsi(self):
        print(f"Starting prediction on WSI: {self.wsi_path}")
        
        # Perform the WSI prediction
        wsi_output = self.segmentor.predict(
            imgs=[self.wsi_path],
            save_dir=self.save_dir,
            mode="wsi",
            ioconfig=self.ioconfig,
            on_gpu=self.on_gpu,
            crash_on_exception=True,
        )
        print(f"Prediction completed. Results saved to {self.save_dir}")
        return wsi_output

def main(args):
    start_time = time.time()

    # Verify that wsi_path exists
    if not os.path.exists(args.wsi_path):
        raise FileNotFoundError(f"WSI file not found at {args.wsi_path}")

    # Remove previous results if they exist
    if os.path.exists(args.save_dir):
        print(f"Removing existing save_dir: {args.save_dir}")
        shutil.rmtree(args.save_dir)

    # Initialize and run the segmentor
    segmentor = WSISemanticSegmentor(
        wsi_path=args.wsi_path,
        save_dir=args.save_dir,
        on_gpu=args.on_gpu
    )
    segmentor.predict_wsi()

    # Log execution time
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Semantic Segmentation Script")
    parser.add_argument("--task", type=str, required=True, choices=["predict_wsi"], help="Task to execute")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save segmentation results")
    parser.add_argument("--on_gpu", action="store_true", help="Flag to use GPU for prediction")

    args = parser.parse_args()
    main(args)


# USE EXAMPLES

# Single WSI File Example

# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation3.py \
#     --task predict_wsi \
#     --wsi_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Patient-Specific_Microdosimetry/DATABASE_GYN/TCGA_CESC/histopathology/svs_slides/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/TCGA-C5-A905-01Z-00-DX1.CD4B818F-C70D-40C9-9968-45DF22F1B149.svs \
#     --save_dir /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/semantic_mask/tcga_cesc/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/wsi_segmentation_results2_0.2277mpp_40x \
#     --on_gpu

# Batch Processing with Sample Sheet Example
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation3.py \
#     --task predict_wsi \
#     --sample_sheet /Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_Proton_filemap.tsv \
#     --parent_wsi_dir /Data/Yujing/Segment/tmp/tcga_cesc_svs \
#     --parent_save_dir /media/yujing/One Touch3/Segment/semantic_mask/tcga_cesc \
#     --on_gpu

# RUN ./scripts/bash_scripts/semantic_segmentation3.sh
