"""
Test scripts before the real one: "nucleus_instance_seg_raw_dist.py"

Obtain the raw distributions of nuclei types after nucleus_instance_seg.py which uses the hovernet of TiaToolBox
The outputs of the raw distributions from here should match with that of the semantic seg matching with the cesc_polygon workflow
before undergoing the histogram correction process to obtain the final distribution mean and sd
before passing to CoxPH model for survival analysis


Author: Yujing Zou
Dec, 2024 
"""

# Clear logger to use tiatoolbox.logger
import logging
import warnings

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tiatoolbox import logger
from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
from tiatoolbox.utils.misc import download_data, imread

# We need this function to visualize the nuclear predictions
from tiatoolbox.utils.visualization import (
    overlay_prediction_contours,
)
from tiatoolbox.wsicore.wsireader import WSIReader

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
plt.rcParams.update({"font.size": 5})

wsi_file_name = '/Data/Yujing/Segment/tmp/tcga_cesc_svs/a63b6131-6667-4a0a-88f3-e5ff1131175b/TCGA-C5-A1BN-01Z-00-DX1.75ED1BD9-C458-42D3-8A98-8FE41BB9CC2E.svs'
seg_filepath = '/Data/Yujing/Segment/tmp/tcga_svs_instance_seg/a63b6131-6667-4a0a-88f3-e5ff1131175b/TCGA-C5-A1BN-01Z-00-DX1.75ED1BD9-C458-42D3-8A98-8FE41BB9CC2E.svs/wsi_instance_segmentation_results/0.dat'
wsi_pred = joblib.load(seg_filepath)
logger.info("Number of detected nuclei: %d", len(wsi_pred))

# Extracting the nucleus IDs and select a random nucleus
rng = np.random.default_rng()
nuc_id_list = list(wsi_pred.keys())
selected_nuc_id = nuc_id_list[
    rng.integers(0, len(wsi_pred))
]  # randomly select a nucleus

logger.info("Nucleus prediction structure for nucleus ID: %s", selected_nuc_id)
sample_nuc = wsi_pred[selected_nuc_id]
sample_nuc_keys = list(sample_nuc)
logger.info(
    "Keys in the output dictionary: [%s, %s, %s, %s, %s]",
    sample_nuc_keys[0],
    sample_nuc_keys[1],
    sample_nuc_keys[2],
    sample_nuc_keys[3],
    sample_nuc_keys[4],
)
logger.info(
    "Bounding box: (%d, %d, %d, %d)",
    sample_nuc["box"][0],
    sample_nuc["box"][1],
    sample_nuc["box"][2],
    sample_nuc["box"][3],
)
logger.info(
    "Centroid: (%d, %d)",
    sample_nuc["centroid"][0],
    sample_nuc["centroid"][1],
)

# [WSI overview extraction]
# Reading the WSI
wsi = WSIReader.open(wsi_file_name)
logger.info(
    "WSI original dimensions: (%d, %d)",
    wsi.info.slide_dimensions[0],
    wsi.info.slide_dimensions[1],
)

# Reading the whole slide in the highest resolution as a plane image
wsi_overview = wsi.slide_thumbnail(resolution=0.25, units="mpp")
logger.info(
    "WSI overview dimensions: (%d, %d, %d)",
    wsi_overview.shape[0],
    wsi_overview.shape[1],
    wsi_overview.shape[2],
)

color_dict = {
    0: ("neoplastic epithelial", (255, 0, 0)),
    1: ("Inflammatory", (255, 255, 0)),
    2: ("Connective", (0, 255, 0)),
    3: ("Dead", (0, 0, 0)),
    4: ("non-neoplastic epithelial", (0, 0, 255)),
}

# Create the overlay image
overlaid_predictions = overlay_prediction_contours(
    canvas=wsi_overview,
    inst_dict=wsi_pred,
    draw_dot=False,
    type_colours=color_dict,
    line_thickness=4,
)

# showing processed results alongside the original images
fig = (
    plt.figure(),
    plt.imshow(wsi_overview),
    plt.axis("off"),
    plt.title("Whole Slide Image"),
)
fig = (
    plt.figure(),
    plt.imshow(overlaid_predictions),
    plt.axis("off"),
    plt.title("Instance Segmentation Overlaid"),
)

