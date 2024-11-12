"""
Converting the polygon annotation to binary masks for each 4k by 4k patches of the WSI 
- Hou et.al. (Scientific Data) provided: https://github.com/SBU-BMI/quip_cnn_segmentation/blob/master/segmentation-of-nuclei/READMD.md#extracting-segmentation-mask-from-output-folder
  - But docker image was too big, haven't been successful in running the code; wrote our own code here 

Source: Pan-Cancer-Nuclei-Seg Code 

Author: Yujing Zou
"""

import os
import argparse
import pandas as pd
import numpy as np
from skimage.measure import label, regionprops
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from matplotlib.patches import Patch 
from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader
import torch




# class NucleiSegmentationMask:
#     def __init__(self, csv_path, tile_size=(4000, 4000)):
#         """
#         Initialize the NucleiSegmentationMask with a CSV file path and tile size.
#         """
#         self.csv_path = csv_path
#         self.tile_size = tile_size
#         self.offset = self.extract_offset_from_filename(csv_path)  # Dynamically set the offset
#         self.data = None
#         self.mask = np.zeros(tile_size, dtype=np.uint8)

#     def extract_offset_from_filename(self, filepath):
#         """
#         Extract the offset (x, y) from the filename.
#         Assumes the filename format includes the offset as the first two numbers separated by underscores.
#         """
#         filename = os.path.basename(filepath)
#         parts = filename.split("_")
#         x_offset = int(parts[0])
#         y_offset = int(parts[1])
#         print(f"Extracted offset: (x={x_offset}, y={y_offset})")
#         return (x_offset, y_offset)

#     def load_data(self):
#         """
#         Load CSV data into a DataFrame.
#         """
#         self.data = pd.read_csv(self.csv_path)
#         print("Data loaded successfully.")

#     def parse_polygon(self, polygon_str):
#         """
#         Parse a polygon string into a list of (x, y) coordinates.
#         Adjust coordinates based on the offset.
#         """
#         points = polygon_str.strip("[]").split(":")
#         # Adjust each coordinate by subtracting the offset
#         return [(int(float(points[i])) - self.offset[0], int(float(points[i+1])) - self.offset[1]) for i in range(0, len(points), 2)]

#     def create_mask(self):
#         """
#         Create a binary mask from the polygons in the CSV data.
#         """
#         if self.data is None:
#             raise ValueError("Data not loaded. Please call load_data() first.")
        
#         for _, row in self.data.iterrows():
#             polygon_str = row['Polygon']
#             vertices = self.parse_polygon(polygon_str)
            
#             # Only draw polygons that are within the bounds of the mask
#             if all(0 <= x < self.tile_size[0] and 0 <= y < self.tile_size[1] for x, y in vertices):
#                 vertices_np = np.array([vertices], dtype=np.int32)
#                 cv2.fillPoly(self.mask, vertices_np, color=1)  # Fill with 1 to create binary mask
#         print("Mask created successfully.")

#     def save_mask(self, output_path):
#         """
#         Save the mask as a binary image.
#         """
#         Image.fromarray(self.mask * 255).save(output_path)
#         print(f"Binary segmentation mask saved at {output_path}")

#     def display_mask(self):
#         """
#         Display the mask using matplotlib.
#         """
#         plt.imshow(self.mask, cmap='gray')
#         plt.title("Nuclei Segmentation Binary Mask")
#         plt.axis('off')
#         plt.show()

class NucleiSegmentationMask:
    def __init__(self, csv_path, tile_size=(4000, 4000)):
        """
        Initialize the NucleiSegmentationMask with a CSV file path and tile size.
        """
        self.csv_path = csv_path
        self.tile_size = tile_size
        self.offset = self.extract_offset_from_filename(csv_path)  # Dynamically set the offset
        self.data = None
        self.mask = np.zeros(tile_size, dtype=np.uint8)

    def extract_offset_from_filename(self, filepath):
        """
        Extract the offset (x, y) from the filename.
        Assumes the filename format includes the offset as the first two numbers separated by underscores.
        """
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        x_offset = int(parts[0])
        y_offset = int(parts[1])
        print(f"Extracted offset: (x={x_offset}, y={y_offset})")
        return (x_offset, y_offset)

    def load_data(self):
        """
        Load CSV data into a DataFrame.
        """
        self.data = pd.read_csv(self.csv_path)
        print("Data loaded successfully.")

    def parse_polygon(self, polygon_str):
        """
        Parse a polygon string into a list of (x, y) coordinates.
        Adjust coordinates based on the offset.
        """
        points = polygon_str.strip("[]").split(":")
        # Adjust each coordinate by subtracting the offset
        return [(int(float(points[i])) - self.offset[0], int(float(points[i+1])) - self.offset[1]) for i in range(0, len(points), 2)]

    def create_mask(self):
        """
        Create a binary mask from the polygons in the CSV data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        for _, row in self.data.iterrows():
            polygon_str = row['Polygon']
            vertices = self.parse_polygon(polygon_str)
            
            # Only draw polygons that are within the bounds of the mask
            if all(0 <= x < self.tile_size[0] and 0 <= y < self.tile_size[1] for x, y in vertices):
                vertices_np = np.array([vertices], dtype=np.int32)
                cv2.fillPoly(self.mask, vertices_np, color=1)  # Fill with 1 to create binary mask
        print("Mask created successfully.")

    def save_mask(self, output_path):
        """
        Save the mask as a binary image.
        """
        Image.fromarray(self.mask * 255).save(output_path)
        print(f"Binary segmentation mask saved at {output_path}")

    def display_mask(self):
        """
        Display the mask using matplotlib.
        """
        plt.imshow(self.mask, cmap='gray')
        plt.title("Nuclei Segmentation Binary Mask")
        plt.axis('off')
        plt.show()



class QA_NucleiMaskAreaAnalysis:
    def __init__(self, csv_path, mask_path, output_path="area_comparison_histogram.png"):
        """
        Initialize with the paths to the original CSV file, binary mask image, and output file path.
        """
        self.csv_path = csv_path
        self.mask_path = mask_path
        self.output_path = output_path
        self.offset = self.extract_offset_from_filename(csv_path)
        self.original_data = None
        self.binary_mask = None
        self.areas_from_mask = []
        self.areas_from_csv = []

    def extract_offset_from_filename(self, filepath):
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        x_offset = int(parts[0])
        y_offset = int(parts[1])
        print(f"Extracted offset: (x={x_offset}, y={y_offset})")
        return (x_offset, y_offset)

    def load_data(self):
        self.original_data = pd.read_csv(self.csv_path)
        self.binary_mask = np.array(Image.open(self.mask_path).convert('L')) // 255
        print("Data loaded successfully.")

    def parse_polygon(self, polygon_str):
        points = polygon_str.strip("[]").split(":")
        return [(int(float(points[i])) - self.offset[0], int(float(points[i+1])) - self.offset[1]) for i in range(0, len(points), 2)]

    def calculate_areas_from_mask_using_polygons(self):
        for _, row in self.original_data.iterrows():
            polygon_str = row['Polygon']
            vertices = self.parse_polygon(polygon_str)
            vertices_np = np.array([vertices], dtype=np.int32)
            nucleus_mask = np.zeros_like(self.binary_mask, dtype=np.uint8)
            cv2.fillPoly(nucleus_mask, [vertices_np], 1)
            area_in_mask = np.sum(self.binary_mask * nucleus_mask)
            self.areas_from_mask.append(area_in_mask)
        print("Areas from binary mask calculated successfully.")

    def calculate_areas_from_csv(self):
        self.areas_from_csv = self.original_data['AreaInPixels'].tolist()
        print("Areas from CSV extracted successfully.")

    def plot_overlapping_histogram(self):
        mean_area_mask = np.mean(self.areas_from_mask)
        mean_area_csv = np.mean(self.areas_from_csv)
        plt.figure(figsize=(12, 8))
        plt.hist(self.areas_from_mask, bins=100, range=(0, 5000), alpha=0.5, label=f"Binary Mask (Mean Area: {mean_area_mask:.2f})", color='blue')
        plt.hist(self.areas_from_csv, bins=100, range=(0, 5000), alpha=0.5, label=f"CSV Data (Mean Area: {mean_area_csv:.2f})", color='green')
        plt.xlabel("Area in Pixels", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Comparison of AreaInPixels Distributions", fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.show()
        print(f"Histogram saved at {self.output_path}")
        print(f"Mean Area from Binary Mask: {mean_area_mask:.2f}")
        print(f"Mean Area from CSV: {mean_area_csv:.2f}")
        print(f"Difference in Mean Area: {abs(mean_area_mask - mean_area_csv):.2f} pixels")

    def run_analysis(self):
        self.load_data()
        self.calculate_areas_from_mask_using_polygons()
        self.calculate_areas_from_csv()
        self.plot_overlapping_histogram()



def main(args):
    if args.task == "polygon_to_mask":
        mask_generator = NucleiSegmentationMask(args.csv_path, tile_size=(args.tile_size, args.tile_size))
        mask_generator.load_data()
        mask_generator.create_mask()
        mask_generator.save_mask(args.output_path)
        if args.display:
            mask_generator.display_mask()

    elif args.task == "qa_area_analysis":
        area_analysis = QA_NucleiMaskAreaAnalysis(args.csv_path, args.mask_path, args.output_path)
        area_analysis.run_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nuclei Segmentation and QA Analysis")
    parser.add_argument("--task", type=str, required=True, choices=["polygon_to_mask", "qa_area_analysis"], help="Task to perform")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with polygon annotations")
    parser.add_argument("--output_path", type=str, help="Path to save the output mask or histogram image")
    parser.add_argument("--tile_size", type=int, default=4000, help="Tile size for mask generation (default: 4000)")
    parser.add_argument("--display", action="store_true", help="Display the generated mask if set")
    parser.add_argument("--mask_path", type=str, help="Path to the binary mask image (required for qa_area_analysis)")

    args = parser.parse_args()
    main(args)

# USE CASES

# 1. Convert polygon annotations to binary mask for a single patch
# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/polygon_to_masks.py \
#     --task polygon_to_mask \
#     --csv_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/blca_polygon/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-features.csv \
#     --output_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/output_masks/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-mask.png \
#     --display \
#     --tile_size 4000

# python nuclei_segmentation.py --task polygon_to_mask --csv_path path/to/file.csv --output_path path/to/save/mask.png --display --tile_size 4000


# 2. Perform QA analysis comparing AreaInPixels from CSV and binary mask for a single patch

# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/polygon_to_masks.py \
#     --task qa_area_analysis \
#     --csv_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/blca_polygon/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-features.csv \
#     --mask_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/output_masks/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-mask.png \
#     --output_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/QA_mask_to_area/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-area_histograms_comp.png

# Save only the binary mask 
# python /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/polygon_to_masks.py \
#     --csv_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/blca_polygon/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-features.csv \
#     --task save_mask \
#     --save_mask_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/output_masks/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-mask.png 



# python polygon_to_mask.py --csv_path path/to/polygon.csv --task save_mask --save_mask_path path/to/save_mask.png


# 3. To save and 

# python polygon_to_mask.py \
#     --csv_path path/to/polygon.csv \
#     --wsi_path path/to/wsi.svs \
#     --task plot_side_by_side \
#     --show_overlay \
#     --save_plot_path path/to/save_plot.png
