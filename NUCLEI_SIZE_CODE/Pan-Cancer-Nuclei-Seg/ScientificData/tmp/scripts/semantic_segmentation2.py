"""
Semantic segmentation via TiaToolbox

Proton GPU 

Yujing Zou
"""


import os 
import shutil
# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm

from tiatoolbox import logger
from tiatoolbox.models.architecture.unet import UNetModel
from tiatoolbox.models.engine.semantic_segmentor import (
    IOSegmentorConfig,
    SemanticSegmentor,
)
from tiatoolbox.utils.misc import download_data, imread
from tiatoolbox.utils.visualization import overlay_prediction_mask
from tiatoolbox.wsicore.wsireader import WSIReader

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
warnings.filterwarnings("ignore")

ON_GPU = True

wsi_path = '/home/yujingz/scratch/NUCLEI_SIZE_CODE/Patient-Specific_Microdosimetry/DATABASE_GYN/TCGA_CESC/histopathology/svs_slides/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/TCGA-C5-A905-01Z-00-DX1.CD4B818F-C70D-40C9-9968-45DF22F1B149.svs'

# mpp = micrometers per pixel 
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=16, # Proton GPU has 48 cores, default is 4 
    batch_size=32, # defaut 4, when Proton on full capacity, 32 or 64 is still fine
    auto_generate_mask=True,  # Enable tissue mask generation
)

# save_dir = "/home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results_0.2277mpp_40x"
# segmenation output map below saved to ./wsi_segmentation_results
# bcc_wsi_ioconfig = IOSegmentorConfig(
#     input_resolutions=[{"units": "mpp", "resolution": 0.5}],  # Adjust as needed for the model
#     output_resolutions=[{"units": "mpp", "resolution": 0.5}], # Adjust as needed for the model
#     patch_input_shape=[1024, 1024], # This might require adjustments for GPU memory
#     patch_output_shape=[512, 512], # This might require adjustments based on the model
#     stride_shape=[512, 512], # Adjust for desired overlap
#     save_resolution={"units": "mpp", "resolution": 2},  # Adjust for desired output resolution
# )

# Update IOSegmentorConfig with 4000 x 4000 patch size at 40x (0.25 mpp)
# OUTPUT WAS TOO LARGE FOR MEMORY TO BE SAVED 
# WENT BACK TO NARVAL FOR 40X RESOLUTION INFERENCE 

# Trying below at 40 x resolution mpp = 0.2277 for this specific WSI 

mpp = 0.2277
bcc_wsi_ioconfig = IOSegmentorConfig(
    input_resolutions=[{"units": "mpp", "resolution": mpp}],   # Set to 0.25 mpp for 40x, coded in the csv results file 
    output_resolutions=[{"units": "mpp", "resolution": mpp}],  # Match output to input
    patch_input_shape=[1024,1024],     # Set input patch size to 4000 x 4000
    patch_output_shape=[512,512],    # Model output is half of input dimensions
    stride_shape=[512,512],          # Set stride to match output size to avoid gaps
    save_resolution={"units": "mpp", "resolution": mpp}  # Save at 40x for accurate overlay
)

# save_dir = "/home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x"
# Save data on the ./Data/Yujing directory (has 4TB of space)

# ALREADY SAVED!!
# save_dir = "/Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x"

save_dir = "/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/semantic_mask/tcga_cesc/1b85f8c4-d60e-4fe9-bb4f-fc1875ef5ca3/wsi_segmentation_results2_0.2277mpp_40x"


if os.path.exists(save_dir):
    shutil.rmtree(save_dir)  # Remove the directory and all its contents


# WSI prediction 
wsi_output = bcc_segmentor.predict(
    imgs=[wsi_path],
    # masks=None,
    save_dir=save_dir,  # Choose a directory to save the results
    mode="wsi",
    ioconfig=bcc_wsi_ioconfig,
    on_gpu=ON_GPU,
    crash_on_exception=True,
)