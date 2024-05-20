# Clear logger to use tiatoolbox.logger
import logging
import warnings
import os
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
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import imread
from tiatoolbox.wsicore.wsireader import WSIReader
import tiatoolbox
from tiatoolbox.models.architecture import get_pretrained_model
import cv2
import os, glob

# Clear existing handlers if any
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.size": 5})

print("Imported tiatoolbox modules")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and move it to the GPU if available
# Assuming get_pretrained_model returns a tuple where the first element is the model
model, _ = get_pretrained_model("hovernet_fast-pannuke")
model = model.to(device)

# Create a SemanticSegmentor
segmentor = SemanticSegmentor(model=model)

# Load an example .svs image
svs_path = os.path.join('/usb', 'DATA', 'Head and Neck_Sultanem', 'Organized_PT_Subfolders2', 'output_box1', 'SS-16-11371', '1012088.svs')
wsi = WSIReader.open(svs_path)
print(f"WSI object created: {wsi}")

# Define the region to process (e.g., the whole slide or a specific region)
# region = wsi.bounds(level=0)
# print(f"Region to process: {region}")

# Segment the tumor region
results = segmentor.predict(wsi, patch_input_shape=(256, 256), resolution=0.5)

# The 'results' contains the segmentation mask. Convert it to binary if necessary
binary_mask = results['masks'][0] > 0.5  # Assuming the model returns a probability map

# Save the binary mask
binary_mask_root = 'DATA/temp/binary_mask'
binary_mask_path = f"{binary_mask_root}/binary_mask.png"
cv2.imwrite(binary_mask_path, (binary_mask * 255).astype(np.uint8))

print(f"Binary mask saved to: {binary_mask_path}")
