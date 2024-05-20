"""

"""

"""
Using TIA Toolbox's semantic segmentation model to obtain 
a tumor binary mask of an input svs. histopathology whole slide image.

First try on an existing Head and Neck slide image, then process for TCGA-CESC
datasets. They don't have the HPV status but can compare with the other clinical 
variables. 

This is a tester script following this: https://tia-toolbox.readthedocs.io/en/latest/_notebooks/jnb/06-semantic-segmentation.html

Author: Yujing Zou 

"""

"""Import modules required to run the Jupyter notebook."""

"""
In HIPT_Embedding_Env virtual environment, had to do the following pip install
pip install --no-index urllib3
pip install --no-index charset_normalizer
pip install --no-index idna
pip install --no-index certifi
pip install --no-index importlib-metadata


"""

# Clear logger to use tiatoolbox.logger
import logging
import warnings
import os 

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

# These file name are used for
img_file_name = "sample_tile.jpg"
wsi_file_name = "sample_wsi.svs"
mini_wsi_file_name = "mini_wsi.svs"
model_file_name = "tissue_mask_model.pth"

logger.info("Download has started. Please wait...")

# Downloading sample image tile
if not os.path.exists(img_file_name):
    download_data(
        "https://tiatoolbox.dcs.warwick.ac.uk/sample_imgs/breast_tissue.jpg",
        img_file_name,
    )

# Downloading sample whole-slide image
if not os.path.exists(wsi_file_name):
    download_data(
        "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/wsi4_12k_12k.svs",
        wsi_file_name,
    )

# Downloading mini whole-slide image
if not os.path.exists(mini_wsi_file_name):
    download_data(
        "https://tiatoolbox.dcs.warwick.ac.uk/sample_wsis/CMU-1.ndpi",
        mini_wsi_file_name,
    )

# Download external model
if not os.path.exists(model_file_name):
    download_data(
        "https://tiatoolbox.dcs.warwick.ac.uk//models/seg/fcn-tissue_mask.pth",
        model_file_name,
    )

logger.info("Download is complete.")


# Tile prediction
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
)

output = bcc_segmentor.predict(
    [img_file_name],
    save_dir="sample_tile_results/",
    mode="tile",
    resolution=1.0,
    units="baseline",
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    on_gpu=ON_GPU,
    crash_on_exception=True,
)



logger.info(
    "Prediction method output is: %s, %s",
    output[0][0],
    output[0][1],
)
tile_prediction_raw = np.load(
    output[0][1] + ".raw.0.npy",
)  # Loading the first prediction [0] based on the output address [1]
logger.info(
    "Raw prediction dimensions: (%d, %d, %d)",
    tile_prediction_raw.shape[0],
    tile_prediction_raw.shape[1],
    tile_prediction_raw.shape[2],
)

# Simple processing of the raw prediction to generate semantic segmentation task
tile_prediction = np.argmax(
    tile_prediction_raw,
    axis=-1,
)  # select the class with highest probability
logger.info(
    "Processed prediction dimensions: (%d, %d)",
    tile_prediction.shape[0],
    tile_prediction.shape[1],
)

# showing the predicted semantic segmentation
tile = imread(img_file_name)
logger.info(
    "Input image dimensions: (%d, %d, %d)",
    tile.shape[0],
    tile.shape[1],
    tile.shape[2],
)

fig = plt.figure()
label_names_dict = {
    0: "Tumour",
    1: "Stroma",
    2: "Inflamatory",
    3: "Necrosis",
    4: "Others",
}
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    plt.imshow(tile_prediction_raw[:, :, i]), plt.xlabel(
        label_names_dict[i],
    ), ax.axes.xaxis.set_ticks([]), ax.axes.yaxis.set_ticks([])
fig.suptitle("Row prediction maps for different classes", y=0.65)

# save the fig
fig.savefig("row_prediction_maps.png")

# showing processed results
fig2 = plt.figure()
ax1 = plt.subplot(1, 2, 1), plt.imshow(tile), plt.axis("off")
ax2 = plt.subplot(1, 2, 2), plt.imshow(tile_prediction), plt.axis("off")
fig2.suptitle("Processed prediction map", y=0.82)

# save the fig 
fig2.savefig("processed_prediction_map.png")


# inference on whole slide image
bcc_segmentor = SemanticSegmentor(
    pretrained_model="fcn_resnet50_unet-bcss",
    num_loader_workers=4,
    batch_size=4,
    auto_generate_mask=False,
)

bcc_wsi_ioconfig = IOSegmentorConfig(
    input_resolutions=[{"units": "mpp", "resolution": 0.25}],
    output_resolutions=[{"units": "mpp", "resolution": 0.25}],
    patch_input_shape=[1024, 1024],
    patch_output_shape=[512, 512],
    stride_shape=[512, 512],
    save_resolution={"units": "mpp", "resolution": 2},
)

# WSI prediction
wsi_output = bcc_segmentor.predict(
    imgs=[wsi_file_name],
    masks=None,
    save_dir="sample_wsi_results/",
    mode="wsi",
    ioconfig=bcc_wsi_ioconfig,
    on_gpu=ON_GPU,
    crash_on_exception=True,
)

logger.info(
    "Prediction method output is: %s, %s",
    wsi_output[0][0],
    wsi_output[0][1],
)
wsi_prediction_raw = np.load(
    wsi_output[0][1] + ".raw.0.npy",
)  # Loading the first prediction [0] based on the output address [1]
logger.info(
    "Raw prediction dimensions: (%d, %d, %d)",
    wsi_prediction_raw.shape[0],
    wsi_prediction_raw.shape[1],
    wsi_prediction_raw.shape[2],
)

# Simple processing of the raw prediction to generate semantic segmentation task
wsi_prediction = np.argmax(
    wsi_prediction_raw,
    axis=-1,
)  # select the class with highest probability
logger.info(
    "Processed prediction dimensions: (%d, %d)",
    wsi_prediction.shape[0],
    wsi_prediction.shape[1],
)

# [WSI overview extraction]
# Now reading the WSI to extract it's overview
wsi = WSIReader.open(wsi_file_name)
logger.info(
    "WSI original dimensions: (%d, %d)",
    wsi.info.slide_dimensions[0],
    wsi.info.slide_dimensions[1],
)

# using the prediction save_resolution to create the wsi overview at the same resolution
overview_info = bcc_wsi_ioconfig.save_resolution

# extracting slide overview using `slide_thumbnail` method
wsi_overview = wsi.slide_thumbnail(
    resolution=overview_info["resolution"],
    units=overview_info["units"],
)
logger.info(
    "WSI overview dimensions: (%d, %d)",
    wsi_overview.shape[0],
    wsi_overview.shape[1],
)
plt.figure(), plt.imshow(wsi_overview)
plt.axis("off")

# [Overlay map creation]
# creating label-color dictionary to be fed into `overlay_prediction_mask` function
# to help generating color legend
label_dict = {"Tumour": 0, "Stroma": 1, "Inflamatory": 2, "Necrosis": 3, "Others": 4}
label_color_dict = {}
colors = cm.get_cmap("Set1").colors
for class_name, label in label_dict.items():
    label_color_dict[label] = (class_name, 255 * np.array(colors[label]))
# Creat overlay map using the `overlay_prediction_mask` helper function
overlay = overlay_prediction_mask(
    wsi_overview,
    wsi_prediction,
    alpha=0.5,
    label_info=label_color_dict,
    return_ax=True,
)

