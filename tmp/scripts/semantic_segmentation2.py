"""
Semantic segmentation via TiaToolbox

Example usage: can path directly the specific wsi_path and save_dir
# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation.py \
#     --task predict_wsi \
#     --wsi_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Patient-Specific_Microdosimetry/DATABASE_GYN/TCGA_CESC/histopathology/svs_slides/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/TCGA-C5-A905-01Z-00-DX1.CD4B818F-C70D-40C9-9968-45DF22F1B149.svs \
#     --save_dir /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/semantic_mask/tcga_cesc/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/wsi_segmentation_results2_0.2277mpp_40x \
#     --mpp 0.2277 \
#     --on_gpu

- Cannot dynamically loop through file names from a directory 
- Modifications to do that: see ./semantic_segmentation3.py


Proton GPU 

Yujing Zou
"""
import os
import shutil
# import loggingpython
import logging
import warnings
import matplotlib as mpl
import numpy as np
import torch
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox import logger
import argparse
import time

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

# Set logging level to ERROR to suppress less critical messages
logging.basicConfig(level=logging.ERROR)


class WSISemanticSegmentor:
    def __init__(self, wsi_path, save_dir, mpp, on_gpu=True):
        self.wsi_path = wsi_path
        self.save_dir = save_dir
        self.mpp = mpp
        self.on_gpu = on_gpu
        self.segmentor = self._initialize_segmentor()
        self.ioconfig = self._create_ioconfig()

        # if logging.getLogger().hasHandlers():
        #     logging.getLogger().handlers.clear()
        # Set tiatoolbox logger to only log errors to avoid super verbose output
        logger.setLevel(logging.ERROR)

    def _initialize_segmentor(self):
        return SemanticSegmentor(
            pretrained_model="fcn_resnet50_unet-bcss",
            num_loader_workers=64, #64,
            batch_size=64, #32 ,2 
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
    # Start the timer
    start_time = time.time()
    
    # Check if save_dir exists and remove it if necessary
    if os.path.exists(args.save_dir):
        print(f"Removing existing save_dir: {args.save_dir}")
        shutil.rmtree(args.save_dir)
    
    # Create WSI segmentor instance
    segmentor = WSISemanticSegmentor(
        wsi_path=args.wsi_path,
        save_dir=args.save_dir,
        mpp=args.mpp,
        on_gpu=args.on_gpu
    )

    # Perform task based on specified argument
    if args.task == "predict_wsi":
        segmentor.predict_wsi()
    else:
        print(f"Unknown task: {args.task}")

    # End the timer
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Semantic Segmentation Script")
    parser.add_argument("--task", type=str, required=True, choices=["predict_wsi"], help="Task to execute")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save segmentation results")
    parser.add_argument("--mpp", type=float, default=0.2277, help="Microns per pixel resolution for the WSI")
    parser.add_argument("--on_gpu", action="store_true", help="Flag to use GPU for prediction")

    args = parser.parse_args()
    main(args)
    
# USE EXAMPLES 
# semantic segmentation of a WSI file from a TCGA-CESC case

# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/semantic_segmentation.py \
#     --task predict_wsi \
#     --wsi_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Patient-Specific_Microdosimetry/DATABASE_GYN/TCGA_CESC/histopathology/svs_slides/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/TCGA-C5-A905-01Z-00-DX1.CD4B818F-C70D-40C9-9968-45DF22F1B149.svs \
#     --save_dir /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/semantic_mask/tcga_cesc/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/wsi_segmentation_results2_0.2277mpp_40x \
#     --mpp 0.2277 \
#     --on_gpu

# uppress all output except the progress bar
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation2.py \
#     --task predict_wsi \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --save_dir /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x \
#     --mpp 0.2277 \
#     --on_gpu > /dev/null 2>&1

# /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/blca_polygon/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs.tar.gz
