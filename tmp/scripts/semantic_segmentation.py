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

wsi_path = '/home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs'

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
save_dir = "/Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x"

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

# ./visualize_semantic_segmentation.py visualizes the segmentation outpus 
# WSI outputs 5 channels of probability maps for each class: 
    # label_dict = {"Tumour": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}
# 0. Generate segmentation mask with the highest probability per pixel
# 1. Must map pixel by pixel of semantic segmentation output to the Pan-Cancer-Nuclei-Seg .csv 4k by 4k files 
# 2. Merge semantic segmentation output to the Pan-Cancer-Nuclei-Seg .csv 4k by 4k files
    # Once a pixel class is obtained, within a QAed segmented nuclei (number of pixels), majority vote for classification of nuclei class
    # PixelInAreas was already reported in the Pan-Cancer-Nuclei-Seg .csv files
    # Output for each patch, PixelInAreas vector for each class
    # Output for each WSI: concatenation of PixelInAreas vectors for each class ffrom each patch 
