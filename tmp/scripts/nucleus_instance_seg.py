"""
Nucleus Instance Segmentation via TiaToolbox

Author: Yujing Zou

Virtual environment setup:
    conda activate segment4
        Here, using segment3 virtual environment since the SimpleITK package had stuck in the segment2 installation 
Virtual environment for all other scripts other than this:
    conda activate segment2


Example usage:
    Batch processing:
    python nucleus_instance_seg.py \
        --task predict_wsi \
        --sample_sheet /path/to/sample_sheet.tsv \
        --parent_wsi_dir /path/to/wsi_dir \
        --parent_save_dir /path/to/save_dir \
        --on_gpu

    Single WSI processing:
    python nucleus_instance_seg.py \
        --task predict_wsi \
        --wsi_path /path/to/wsi.svs \
        --save_dir /path/to/save_dir \
        --on_gpu
        
"""

import os
import shutil
import logging
import warnings
import pandas as pd
import matplotlib as mpl
from tiatoolbox.models import NucleusInstanceSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox import logger
import argparse
import time
import joblib

mpl.rcParams["figure.dpi"] = 600
mpl.rcParams["figure.facecolor"] = "white"
warnings.filterwarnings("ignore")

# Set logging level to ERROR to suppress less critical messages
logging.basicConfig(level=logging.ERROR)
logger.setLevel(logging.ERROR)


class WSInstanceSegmentor:
    def __init__(self, wsi_path, save_dir, on_gpu=True):
        self.wsi_path = wsi_path
        self.save_dir = save_dir
        self.on_gpu = on_gpu
        self.mpp = self._get_mpp_from_wsi()
        self.segmentor = self._initialize_segmentor()

    def _get_mpp_from_wsi(self):
        # Use WSIReader to obtain mpp value from the WSI file
        try:
            wsi_reader = WSIReader.open(self.wsi_path)
            mpp = wsi_reader.info.mpp[0]  # Assuming square pixels, take the first element for mpp
            print(f"Using mpp value from WSI: {mpp}")
            return mpp
        except Exception as e:
            print(f"Error reading WSI: {e}")
            raise

    def _initialize_segmentor(self):
        try:
            segmentor = NucleusInstanceSegmentor(
                batch_size= 64, #32, #4,  # Adjust based on your GPU memory
                num_loader_workers=96,  # Adjust based on your CPU cores
                num_postproc_workers=96,  # Adjust based on your CPU cores
                pretrained_model="hovernet_fast-pannuke",  # You can choose different models
                auto_generate_mask=True,
                verbose= False, #True,
                # device="cuda" if self.on_gpu else "cpu",
            )
            return segmentor
        except Exception as e:
            print(f"Error initializing NucleusInstanceSegmentor: {e}")
            raise

    def predict_wsi(self):
        print(f"Starting instance segmentation on WSI: {self.wsi_path}")
        
        try:
            # Perform the WSI prediction
            wsi_output = self.segmentor.predict(
                imgs=[self.wsi_path],
                save_dir=self.save_dir,
                mode="wsi",
                on_gpu=self.on_gpu,
                crash_on_exception=True,
            )
            print(f"Instance segmentation completed. Results saved to {self.save_dir}")
            return wsi_output
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise


def process_sample_sheet(args):
    try:
        df = pd.read_csv(args.sample_sheet, sep="\t")
    except Exception as e:
        print(f"Error reading sample sheet: {e}")
        return

    for _, row in df.iterrows():
        file_id = row["File ID"]
        file_name = row["File Name"]
        wsi_path = os.path.join(args.parent_wsi_dir, file_id, file_name)
        save_dir = os.path.join(
            args.parent_save_dir, file_id, file_name, "wsi_instance_segmentation_results"
        )
        
        if os.path.exists(save_dir):
            print(f"Removing existing save_dir: {save_dir}")
            shutil.rmtree(save_dir)
        
        if not os.path.exists(wsi_path):
            print(f"WSI path does not exist: {wsi_path}. Skipping.")
            continue
        
        try:
            segmentor = WSInstanceSegmentor(
                wsi_path=wsi_path,
                save_dir=save_dir,
                on_gpu=args.on_gpu
            )
            segmentor.predict_wsi()
        except Exception as e:
            print(f"Failed to process WSI {wsi_path}: {e}")
            continue


def main(args):
    start_time = time.time()
    
    if args.sample_sheet:
        process_sample_sheet(args)
    else:
        if not args.wsi_path or not args.save_dir:
            print("For single WSI processing, --wsi_path and --save_dir must be provided.")
            return
        
        if os.path.exists(args.save_dir):
            print(f"Removing existing save_dir: {args.save_dir}")
            shutil.rmtree(args.save_dir)
        
        try:
            segmentor = WSInstanceSegmentor(
                wsi_path=args.wsi_path,
                save_dir=args.save_dir,
                on_gpu=args.on_gpu
            )
            segmentor.predict_wsi()
        except Exception as e:
            print(f"Failed to process WSI {args.wsi_path}: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Nucleus Instance Segmentation Script")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["predict_wsi"],
        help="Task to execute",
    )
    parser.add_argument(
        "--wsi_path",
        type=str,
        help="Path to the WSI file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save segmentation results",
    )
    parser.add_argument(
        "--sample_sheet",
        type=str,
        help="Path to the sample sheet TSV file",
    )
    parser.add_argument(
        "--parent_wsi_dir",
        type=str,
        help="Parent directory for WSI files",
    )
    parser.add_argument(
        "--parent_save_dir",
        type=str,
        help="Parent directory for saving segmentation results",
    )
    parser.add_argument(
        "--on_gpu",
        action="store_true",
        help="Flag to use GPU for prediction",
    )
    
    args = parser.parse_args()
    main(args)
    
# USE CASE EXAMPLE 
# input svs path: /Data/Yujing/Segment/tmp/tcga_cesc_svs/a63b6131-6667-4a0a-88f3-e5ff1131175b/TCGA-C5-A1BNTCGA-C5-A1BN-01Z-00-DX1.75ED1BD9-C458-42D3-8A98-8FE41BB9CC2E.svs

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nucleus_instance_seg.py \
#     --task predict_wsi \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/a63b6131-6667-4a0a-88f3-e5ff1131175b/TCGA-C5-A1BN-01Z-00-DX1.75ED1BD9-C458-42D3-8A98-8FE41BB9CC2E.svs \
#     --save_dir /Data/Yujing/Segment/tmp/tcga_svs_instance_seg/TCGA-C5-A1BN \
#     --on_gpu

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nucleus_instance_seg.py \
#     --task predict_wsi \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/75c0e41c-5d67-4d64-aac5-451a168a281d/TCGA-C5-A1MF-01Z-00-DX1.8C787217-8C11-4296-A955-45DA0B17C2BA.svs \
#     --save_dir /Data/Yujing/Segment/tmp/tcga_svs_instance_seg/75c0e41c-5d67-4d64-aac5-451a168a281d/TCGA-C5-A1MF-01Z-00-DX1.8C787217-8C11-4296-A955-45DA0B17C2BA.svs/wsi_instance_segmentation_results \
#     --on_gpu
