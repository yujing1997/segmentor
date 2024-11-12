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


import os
import shutil
import logging
import warnings
import pandas as pd
import matplotlib as mpl
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig, SemanticSegmentor
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
    def __init__(self, wsi_path, save_dir, on_gpu=True):
        self.wsi_path = wsi_path
        self.save_dir = save_dir
        self.on_gpu = on_gpu
        self.mpp = self._get_mpp_from_wsi()
        self.segmentor = self._initialize_segmentor()
        self.ioconfig = self._create_ioconfig()

        # Set tiatoolbox logger to only log errors to avoid super verbose output
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


def process_sample_sheet(args):
    df = pd.read_csv(args.sample_sheet, sep="\t")
    for _, row in df.iterrows():
        file_id = row["File ID"]
        file_name = row["File Name"]
        wsi_path = os.path.join(args.parent_wsi_dir, file_id, file_name)
        save_dir = os.path.join(args.parent_save_dir, file_id, file_name, "wsi_segmentation_results2_0.2277mpp_40x")
        
        if os.path.exists(save_dir):
            print(f"Removing existing save_dir: {save_dir}")
            shutil.rmtree(save_dir)
        
        segmentor = WSISemanticSegmentor(
            wsi_path=wsi_path,
            save_dir=save_dir,
            on_gpu=args.on_gpu
        )
        segmentor.predict_wsi()


def main(args):
    start_time = time.time()
    
    if args.sample_sheet:
        process_sample_sheet(args)
    else:
        if os.path.exists(args.save_dir):
            print(f"Removing existing save_dir: {args.save_dir}")
            shutil.rmtree(args.save_dir)
        
        segmentor = WSISemanticSegmentor(
            wsi_path=args.wsi_path,
            save_dir=args.save_dir,
            on_gpu=args.on_gpu
        )
        segmentor.predict_wsi()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Semantic Segmentation Script")
    parser.add_argument("--task", type=str, required=True, choices=["predict_wsi"], help="Task to execute")
    parser.add_argument("--wsi_path", type=str, help="Path to the WSI file")
    parser.add_argument("--save_dir", type=str, help="Directory to save segmentation results")
    parser.add_argument("--sample_sheet", type=str, help="Path to the sample sheet TSV file")
    parser.add_argument("--parent_wsi_dir", type=str, help="Parent directory for WSI files")
    parser.add_argument("--parent_save_dir", type=str, help="Parent directory for saving segmentation results")
    parser.add_argument("--on_gpu", action="store_true", help="Flag to use GPU for prediction")

    args = parser.parse_args()
    main(args)
    

# USE EXAMPLES

# Single WSI File Example

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation3.py \
#     --task predict_wsi \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs \
#     --save_dir /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/wsi_segmentation_results2_0.2277mpp_40x \
#     --on_gpu

# Batch Processing with Sample Sheet Example
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation3.py \
#     --task predict_wsi \
#     --sample_sheet /Data/Yujing/Segment/tmp/tcga_cesc_manifest/run_partition/run_Proton_filemap.tsv \
#     --parent_wsi_dir /Data/Yujing/Segment/tmp/tcga_cesc_svs \
#     --parent_save_dir /media/yujing/One Touch3/Segment/semantic_mask/tcga_cesc \
#     --on_gpu

# HOW TO MOUNT LOCAL E DRIVE TO BE RECOGNIZED BY DOCKER
