"""
TiaToolbox Patch Extractor 

"""

from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.misc import imread
from tiatoolbox.utils.misc import read_locations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode

print("Imported tiatoolbox patch_extractor modules")

import requests

img_file_name = "sample_img.png"
csv_file_name = "sample_coordinates.csv"

# Downloading sample image from MoNuSeg
r = requests.get(
    "https://tiatoolbox.dcs.warwick.ac.uk/testdata/patchextraction/TCGA-HE-7130-01Z-00-DX1.png"
)
with open(img_file_name, "wb") as f:
    f.write(r.content)

# Downloading points list
r = requests.get(
    "https://tiatoolbox.dcs.warwick.ac.uk/testdata/patchextraction/sample_patch_extraction.csv"
)
with open(csv_file_name, "wb") as f:
    f.write(r.content)

print("Download is complete.")

input_img = imread(img_file_name)
centroids_list = read_locations(csv_file_name)

print("Image size: {}".format(input_img.shape))
print("This image has {} point annotations".format(centroids_list.shape[0]))
print("First few lines of dataframe:\n", centroids_list.head())

input_img = imread(img_file_name)
plt.imshow(input_img)
plt.axis("off")
plt.show()

# overlay nuclei centroids on image and plot
plt.imshow(input_img)
plt.scatter(np.array(centroids_list)[:, 0], np.array(centroids_list)[:, 1], s=1)
plt.axis("off")
plt.show()

# save the image 
plt.imshow(input_img)
plt.scatter(np.array(centroids_list)[:, 0], np.array(centroids_list)[:, 1], s=1)
plt.axis("off")
plt.savefig("sample_img_with_centroids.png")
plt.show()
