# BELOW WORKE! JUST NOT WRITING THE ACTUAL CLASS FIELD BUT THE INTEGER and is only handling one csv at a time

#!/usr/bin/env python3
"""
Visualize WSI inference results and classify nuclei based on the segmentation mask.

Author: Yujing Zou
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader
from skimage.io import imread, imsave
import multiprocessing as mp
from multiprocessing import Pool

import torch
# import cupy as cp
from collections import defaultdict
import pandas as pd
import cv2
from tqdm import tqdm

# Clear previous handlers for the logger
if logger.hasHandlers():
    logger.handlers.clear()

class NucleiSegmentationMask:
    def __init__(self, csv_path):
        """
        Initialize the NucleiSegmentationMask with a CSV file path.
        """
        self.csv_path = csv_path
        self.x_start, self.y_start, self.patch_size = self.extract_offset_from_filename(csv_path)
        self.offset = (self.x_start, self.y_start)
        self.data = None
        self.mask = np.zeros(self.patch_size, dtype=np.uint8)

    def extract_offset_from_filename(self, filepath):
        """
        Extract the offset (x, y) and patch size from the filename.
        Assumes the filename format includes the offset and patch size as numbers separated by underscores.
        """
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        x_start = int(parts[0])
        y_start = int(parts[1])
        patch_width = int(parts[2])
        patch_height = int(parts[3])
        return x_start, y_start, (patch_height, patch_width)  # Note the order for mask dimensions (rows, cols)

    def load_data(self):
        """
        Load CSV data into a DataFrame with optimized data types.
        """
        dtypes = {
            'Polygon': 'object',
            'AreaInPixels': 'int32'
        }
        self.data = pd.read_csv(self.csv_path, dtype=dtypes)

    def parse_polygon(self, polygon_str):
        """
        Parse a polygon string into a list of (x, y) coordinates.
        Adjust coordinates based on the offset.
        """
        polygons = polygon_str.strip('[]').split('],[')
        all_vertices = []
        for polygon in polygons:
            points = polygon.strip('[]').split(':')
            vertices = [(int(float(points[i])) - self.offset[0], int(float(points[i + 1])) - self.offset[1])
                        for i in range(0, len(points), 2)]
            all_vertices.append(vertices)
        return all_vertices

    def create_mask(self):
        """
        Create a binary mask from the polygons in the CSV data.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        for _, row in self.data.iterrows():
            polygon_str = row['Polygon']
            polygons = self.parse_polygon(polygon_str)
            for vertices in polygons:
                if all(0 <= x < self.patch_size[1] and 0 <= y < self.patch_size[0] for x, y in vertices):
                    vertices_np = np.array([vertices], dtype=np.int32)
                    cv2.fillPoly(self.mask, vertices_np, color=1)

    def calculate_area_in_square_microns(self, area_in_pixels, mpp):
        """
        Calculate the area in square microns given area in pixels and microns per pixel.
        """
        area_in_square_microns = area_in_pixels * (mpp ** 2)
        return area_in_square_microns

    @staticmethod
    def area_to_radius(area):
        return np.sqrt(area / np.pi)

    def classify_nuclei(self, segmentation_mask, mpp):
        """
        Classifies nuclei based on the majority class within each contour.
        """
        classified_nuclei = []
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nuclei_num_per_patch = len(contours)
        print(f"Number of nuclei in patch: {nuclei_num_per_patch}")

        for contour in contours:
            contour_mask = np.zeros_like(self.mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            contour_region = segmentation_mask[contour_mask == 1]
            if len(contour_region) == 0:
                continue
            unique, counts = np.unique(contour_region, return_counts=True)
            # print(f"Unique classes in contour: {unique}")
            # print(f"Counts of unique classes: {counts}")
            majority_class = unique[np.argmax(counts)]
            area_in_pixels = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            area_in_square_microns = self.calculate_area_in_square_microns(area_in_pixels, mpp)
            equivalent_radius_microns = self.area_to_radius(area_in_square_microns)
            classified_nuclei.append({
                'majority_class': int(majority_class),
                'AreaInPixels': area_in_pixels,
                'perimeter': perimeter,
                'AreaInSquareMicrons': area_in_square_microns,
                'RadiusInMicrons': equivalent_radius_microns,
                'mpp': mpp
            })
        return classified_nuclei, nuclei_num_per_patch

class NpyImagePlotter:
    def __init__(self, file_path, label_dict=None, x_start=0, y_start=0, patch_size=4000, transpose_segmask=True):
        self.file_path = file_path
        self.x_start = x_start
        self.y_start = y_start
        self.patch_size = patch_size
        self.transpose_segmask = transpose_segmask
        self.data = None
        self.segmentation_mask = None
        self.label_dict = label_dict if label_dict is not None else {}

    def load_data(self):
        try:
            self.data = np.load(self.file_path, mmap_mode='r')
            print("Data loaded with memory mapping.")

            if self.transpose_segmask:
                self.data = np.transpose(self.data, (1, 0, 2))  # Swap the first two axes
                print("Data transposed to match WSI dimensions.")

        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def read_mpp_wsi(self, wsi_path):
        logger.info(f"Loading WSI: {wsi_path}")
        self.reader = WSIReader.open(wsi_path)
        # Access the mpp value of the WSI
        mpp = self.reader.info.mpp  # This returns an array [mpp_x, mpp_y]
        print(f"Micrometers per pixel: {mpp}")

        # Ensure mpp is a scalar value by taking the mean
        mpp_mean = np.mean(mpp)
        print(f"Mean micrometers per pixel: {mpp_mean}")
        return mpp_mean

    def extract_patch(self):
        """Extract a patch from the loaded data based on x_start, y_start, and patch_size."""
        if self.data is None:
            self.load_data()

        if self.x_start is None or self.y_start is None:
            raise ValueError("x_start and y_start must be specified.")

        patch = self.data[
            self.y_start:self.y_start + self.patch_size[0],
            self.x_start:self.x_start + self.patch_size[1],
            :
        ]

        # Transpose the patch if the transpose_segmask option is enabled
        if self.transpose_segmask:
            patch = patch.transpose((1, 0, 2))  # Swap x and y axes for the patch
            print(f"Transposed patch with shape: {patch.shape}")
        else:
            print(f"Extracted patch with shape: {patch.shape}")

        return patch

    def generate_segmentation_mask(self):
        """Generate the segmentation mask by selecting the class with the highest probability for each pixel."""
        if self.data is None:
            self.load_data()
        # Check if data is loaded properly
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Generate mask for the specific patch
        patch = self.extract_patch()
        self.segmentation_mask = np.argmax(patch, axis=-1)
        print(f"PATCH: Segmentation mask shape: {self.segmentation_mask.shape}")

        # Rotate the mask by 90 degrees counterclockwise if transpose_segmask is set
        if self.transpose_segmask:
            self.segmentation_mask = np.rot90(self.segmentation_mask, k=4)  # 90 degrees counterclockwise
            print("Segmentation mask rotated by 90 degrees counterclockwise.")

        print("Segmentation mask shape:", self.segmentation_mask.shape)
        return self.segmentation_mask

class SegmentationVisualizer:
    def __init__(self, wsi_path, csv_dir, segmentation_output_path, label_dict=None, transpose_segmask=False):
        """
        Initialize the SegmentationVisualizer with required paths and parameters.

        Args:
            wsi_path (str): Path to the WSI file for H&E extraction.
            csv_dir (str): Path to the directory containing CSV files for multiple patches.
            segmentation_output_path (str): Path to the semantic segmentation mask .npy file.
            label_dict (dict): Dictionary mapping semantic segmentation classes to indices.
            transpose_segmask (bool): Whether to transpose the segmentation mask.
        """
        self.wsi_path = wsi_path
        self.csv_dir = csv_dir
        self.segmentation_output_path = segmentation_output_path
        self.label_dict = label_dict or {"Tumor": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}
        self.transpose_segmask = transpose_segmask
        self.all_classified_nuclei = []


    def process_all_patches(self, parallel=False, num_workers=8):
        """
        Process all patches with an option for parallel execution and a progress bar.

        Args:
            parallel (bool): If True, enable parallel processing.
            num_workers (int): Number of parallel workers if parallel is enabled.
        """
        csv_files = [os.path.join(self.csv_dir, f) for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
        print(f"Found {len(csv_files)} CSV files to process.", flush=True)

        if not csv_files:
            print(f"No CSV files found in directory: {self.csv_dir}", flush=True)
            return []

        if parallel:
            print(f"Processing CSV files in parallel with {num_workers} workers...", flush=True)
            results = []
            with Pool(num_workers) as pool:
                with tqdm(total=len(csv_files), desc="Processing CSV Files") as pbar:
                    for csv_file in csv_files:
                        # Increment the progress bar for each completed task
                        result = pool.apply_async(self.process_single_patch, args=(csv_file,), callback=lambda _: pbar.update(1))
                        results.append(result)

                    # Close the pool and wait for the tasks to complete
                    pool.close()
                    pool.join()
                    
                # Retrieve results after all tasks are completed
                results = [result.get() for result in results]
        else:
            print("Processing CSV files sequentially...", flush=True)
            results = [self.process_single_patch(csv_file) for csv_file in tqdm(csv_files, desc="Processing CSV Files")]

        # Flatten the list of lists
        self.all_classified_nuclei = [nucleus for sublist in results for nucleus in sublist]
        print(f"Total nuclei classified: {len(self.all_classified_nuclei)}", flush=True)
        return self.all_classified_nuclei

    def process_single_patch(self, csv_file):
        try:
            print(f"Processing patch: {csv_file}", flush=True)
            # Initialize NucleiSegmentationMask for this patch
            nuclei_mask = NucleiSegmentationMask(csv_file)
            # print(f"Initialized NucleiSegmentationMask for {csv_file}", flush=True)
        
            # Load data for nuclei mask
            nuclei_mask.load_data()
            # print(f"Loaded nuclei data for {csv_file}", flush=True)
            nuclei_mask.create_mask()
            # print(f"Created nuclei mask for {csv_file}", flush=True)
        
            # Use x_start, y_start, and patch_size from the nuclei_mask
            x_start = nuclei_mask.x_start
            y_start = nuclei_mask.y_start
            patch_size = nuclei_mask.patch_size  # (height, width)
        
            print(f"x_start: {x_start}, y_start: {y_start}, patch_size: {patch_size}", flush=True)
        
            # Initialize NpyImagePlotter for this patch
            semantic_plotter = NpyImagePlotter(
                file_path=self.segmentation_output_path,
                label_dict=self.label_dict,
                x_start=x_start,
                y_start=y_start,
                patch_size=patch_size,  # (height, width)
                transpose_segmask=self.transpose_segmask
            )
            semantic_plotter.load_data()
            print(f"Loaded segmentation data for {csv_file}", flush=True)
        
            # Read mpp
            mpp = semantic_plotter.read_mpp_wsi(self.wsi_path)
            print(f"Read mpp: {mpp} for {csv_file}", flush=True)
        
            # Generate segmentation mask for this patch
            segmentation_patch = semantic_plotter.generate_segmentation_mask()
            print(f"Generated segmentation mask for {csv_file}", flush=True)
        
            # Classify nuclei
            classified_nuclei, nuclei_num_per_patch = nuclei_mask.classify_nuclei(
                segmentation_patch, mpp
            )
            print(f"Classified {nuclei_num_per_patch} nuclei in patch {csv_file}", flush=True)
            # print(f"Classified nuclei: {classified_nuclei}")
            return classified_nuclei
        except Exception as e:
            print(f"Error processing patch {csv_file}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return []



    def save_aggregated_results(self, output_dir):
        """
        Save aggregated results per class in JSON files.

        Args:
            output_dir (str): Directory to save the output results.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Aggregate data per class
        class_aggregated_data = defaultdict(list)
        for nucleus in self.all_classified_nuclei:
            class_label = nucleus['majority_class']
            class_aggregated_data[class_label].append(nucleus)

        # Save per-class data with attributes as keys and lists of values
        for class_label, nuclei_data in class_aggregated_data.items():
            aggregated_attributes = defaultdict(list)
            for nucleus in nuclei_data:
                for key, value in nucleus.items():
                    if key != 'majority_class':
                        aggregated_attributes[key].append(value)
            output_file = os.path.join(output_dir, f'class_{int(class_label)}_nuclei.json')
            print(f"Saving data for class {int(class_label)} with {len(nuclei_data)} nuclei to {output_file}")
            with open(output_file, 'w') as f:
                json.dump(aggregated_attributes, f)
            print(f"Saved data for class {int(class_label)} with {len(nuclei_data)} nuclei to {output_file}")

        # Save overall statistics
        total_nuclei = len(self.all_classified_nuclei)
        nuclei_per_class = {str(int(class_label)): len(nuclei_data) for class_label, nuclei_data in class_aggregated_data.items()}
        stats = {
            'total_nuclei': total_nuclei,
            'nuclei_per_class': nuclei_per_class
        }
        stats_file = os.path.join(output_dir, 'nuclei_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Saved nuclei statistics to {stats_file}")

def main(args):
    print(f"Starting main function with task: {args.task}", flush=True)
    if args.task == "process_wsi":
        # Ensure output directory exists
        if args.output_dir is None:
            print("Please specify an output directory using --output_dir", flush=True)
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)

        print("Initializing SegmentationVisualizer...", flush=True)
        visualizer = SegmentationVisualizer(
            wsi_path=args.wsi_path,
            csv_dir=args.csv_dir,
            segmentation_output_path=args.segmentation_output_path,
            label_dict=None,  # Use default label_dict
            transpose_segmask=args.transpose_segmask
        )

        print(f"CSV Directory: {args.csv_dir}", flush=True)

        # Process all patches
        print("Calling process_all_patches...", flush=True)
        all_classified_nuclei = visualizer.process_all_patches(parallel=args.parallel, num_workers=args.num_workers)
        # Save the aggregated results
        visualizer.save_aggregated_results(args.output_dir)
    else:
        print(f"Unknown task: {args.task}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Visualization Script")
    parser.add_argument("--task", type=str, required=True, choices=["process_wsi"],
                        help="Task to execute")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file")
    parser.add_argument("--csv_dir", type=str, help="Path to the directory containing CSV files for multiple patches")
    parser.add_argument("--segmentation_output_path", type=str, required=True, help="Path to the .npy file for semantic segmentation")
    parser.add_argument("--output_dir", type=str, help="Directory to save the aggregated results")
    parser.add_argument("--transpose_segmask", action="store_true", help="Transpose segmentation mask if needed")
    parser.add_argument("--parallel", action="store_true", help="Process patches in parallel")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for parallel processing")

    args = parser.parse_args()
    main(args)



# # BELOW WORKE! JUST NOT WRITING THE ACTUAL CLASS FIELD BUT THE INTEGER and is only handling one csv at a time

# #!/usr/bin/env python3
# """
# Visualize WSI inference results and classify nuclei based on the segmentation mask.

# Author: Yujing Zou
# """

# import os
# import sys
# import json
# import argparse
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# from tiatoolbox import logger
# from tiatoolbox.wsicore.wsireader import WSIReader
# from skimage.io import imread, imsave
# import multiprocessing as mp
# from multiprocessing import Pool

# import torch
# # import cupy as cp
# from collections import defaultdict
# import pandas as pd
# import cv2
# from tqdm import tqdm

# # Clear previous handlers for the logger
# if logger.hasHandlers():
#     logger.handlers.clear()

# class NucleiSegmentationMask:
#     def __init__(self, csv_path):
#         """
#         Initialize the NucleiSegmentationMask with a CSV file path.
#         """
#         self.csv_path = csv_path
#         self.x_start, self.y_start, self.patch_size = self.extract_offset_from_filename(csv_path)
#         self.offset = (self.x_start, self.y_start)
#         self.data = None
#         self.mask = np.zeros(self.patch_size, dtype=np.uint8)

#     def extract_offset_from_filename(self, filepath):
#         """
#         Extract the offset (x, y) and patch size from the filename.
#         Assumes the filename format includes the offset and patch size as numbers separated by underscores.
#         """
#         filename = os.path.basename(filepath)
#         parts = filename.split("_")
#         x_start = int(parts[0])
#         y_start = int(parts[1])
#         patch_width = int(parts[2])
#         patch_height = int(parts[3])
#         return x_start, y_start, (patch_height, patch_width)  # Note the order for mask dimensions (rows, cols)

#     def load_data(self):
#         """
#         Load CSV data into a DataFrame with optimized data types.
#         """
#         dtypes = {
#             'Polygon': 'object',
#             'AreaInPixels': 'int32'
#         }
#         self.data = pd.read_csv(self.csv_path, dtype=dtypes)

#     def parse_polygon(self, polygon_str):
#         """
#         Parse a polygon string into a list of (x, y) coordinates.
#         Adjust coordinates based on the offset.
#         """
#         polygons = polygon_str.strip('[]').split('],[')
#         all_vertices = []
#         for polygon in polygons:
#             points = polygon.strip('[]').split(':')
#             vertices = [(int(float(points[i])) - self.offset[0], int(float(points[i + 1])) - self.offset[1])
#                         for i in range(0, len(points), 2)]
#             all_vertices.append(vertices)
#         return all_vertices

#     def create_mask(self):
#         """
#         Create a binary mask from the polygons in the CSV data.
#         """
#         if self.data is None:
#             raise ValueError("Data not loaded. Please call load_data() first.")

#         for _, row in self.data.iterrows():
#             polygon_str = row['Polygon']
#             polygons = self.parse_polygon(polygon_str)
#             for vertices in polygons:
#                 if all(0 <= x < self.patch_size[1] and 0 <= y < self.patch_size[0] for x, y in vertices):
#                     vertices_np = np.array([vertices], dtype=np.int32)
#                     cv2.fillPoly(self.mask, vertices_np, color=1)

#     def calculate_area_in_square_microns(self, area_in_pixels, mpp):
#         """
#         Calculate the area in square microns given area in pixels and microns per pixel.
#         """
#         area_in_square_microns = area_in_pixels * (mpp ** 2)
#         return area_in_square_microns

#     @staticmethod
#     def area_to_radius(area):
#         return np.sqrt(area / np.pi)

#     def classify_nuclei(self, segmentation_mask, mpp):
#         """
#         Classifies nuclei based on the majority class within each contour.
#         """
#         classified_nuclei = []
#         contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         nuclei_num_per_patch = len(contours)
#         for contour in contours:
#             contour_mask = np.zeros_like(self.mask, dtype=np.uint8)
#             cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
#             contour_region = segmentation_mask[contour_mask == 1]
#             if len(contour_region) == 0:
#                 continue
#             unique, counts = np.unique(contour_region, return_counts=True)
#             majority_class = unique[np.argmax(counts)]
#             area_in_pixels = cv2.contourArea(contour)
#             perimeter = cv2.arcLength(contour, True)
#             area_in_square_microns = self.calculate_area_in_square_microns(area_in_pixels, mpp)
#             equivalent_radius_microns = self.area_to_radius(area_in_square_microns)
#             classified_nuclei.append({
#                 'majority_class': int(majority_class),
#                 'AreaInPixels': area_in_pixels,
#                 'perimeter': perimeter,
#                 'AreaInSquareMicrons': area_in_square_microns,
#                 'RadiusInMicrons': equivalent_radius_microns,
#                 'mpp': mpp
#             })
#         return classified_nuclei, nuclei_num_per_patch

# class NpyImagePlotter:
#     def __init__(self, file_path, label_dict=None, x_start=0, y_start=0, patch_size=4000, transpose_segmask=True):
#         self.file_path = file_path
#         self.x_start = x_start
#         self.y_start = y_start
#         self.patch_size = patch_size
#         self.transpose_segmask = transpose_segmask
#         self.data = None
#         self.segmentation_mask = None
#         self.label_dict = label_dict if label_dict is not None else {}

#     def load_data(self):
#         try:
#             self.data = np.load(self.file_path, mmap_mode='r')
#             print("Data loaded with memory mapping.")

#             if self.transpose_segmask:
#                 self.data = np.transpose(self.data, (1, 0, 2))  # Swap the first two axes
#                 print("Data transposed to match WSI dimensions.")

#         except FileNotFoundError:
#             print(f"File not found: {self.file_path}")
#         except Exception as e:
#             print(f"An error occurred: {e}")

#     def read_mpp_wsi(self, wsi_path):
#         logger.info(f"Loading WSI: {wsi_path}")
#         self.reader = WSIReader.open(wsi_path)
#         # Access the mpp value of the WSI
#         mpp = self.reader.info.mpp  # This returns an array [mpp_x, mpp_y]
#         print(f"Micrometers per pixel: {mpp}")

#         # Ensure mpp is a scalar value by taking the mean
#         mpp_mean = np.mean(mpp)
#         print(f"Mean micrometers per pixel: {mpp_mean}")
#         return mpp_mean

#     def extract_patch(self):
#         """Extract a patch from the loaded data based on x_start, y_start, and patch_size."""
#         if self.data is None:
#             self.load_data()

#         if self.x_start is None or self.y_start is None:
#             raise ValueError("x_start and y_start must be specified.")

#         patch = self.data[
#             self.y_start:self.y_start + self.patch_size[0],
#             self.x_start:self.x_start + self.patch_size[1],
#             :
#         ]

#         # Transpose the patch if the transpose_segmask option is enabled
#         if self.transpose_segmask:
#             patch = patch.transpose((1, 0, 2))  # Swap x and y axes for the patch
#             print(f"Transposed patch with shape: {patch.shape}")
#         else:
#             print(f"Extracted patch with shape: {patch.shape}")

#         return patch

#     def generate_segmentation_mask(self):
#         """Generate the segmentation mask by selecting the class with the highest probability for each pixel."""
#         if self.data is None:
#             self.load_data()
#         # Check if data is loaded properly
#         if self.data is None:
#             raise ValueError("Data not loaded. Call load_data() first.")

#         # Generate mask for the specific patch
#         patch = self.extract_patch()
#         self.segmentation_mask = np.argmax(patch, axis=-1)
#         print(f"PATCH: Segmentation mask shape: {self.segmentation_mask.shape}")

#         # Rotate the mask by 90 degrees counterclockwise if transpose_segmask is set
#         if self.transpose_segmask:
#             self.segmentation_mask = np.rot90(self.segmentation_mask, k=4)  # 90 degrees counterclockwise
#             print("Segmentation mask rotated by 90 degrees counterclockwise.")

#         print("Segmentation mask shape:", self.segmentation_mask.shape)
#         return self.segmentation_mask

# class SegmentationVisualizer:
#     def __init__(self, wsi_path, csv_dir, segmentation_output_path, label_dict=None, transpose_segmask=False):
#         """
#         Initialize the SegmentationVisualizer with required paths and parameters.

#         Args:
#             wsi_path (str): Path to the WSI file for H&E extraction.
#             csv_dir (str): Path to the directory containing CSV files for multiple patches.
#             segmentation_output_path (str): Path to the semantic segmentation mask .npy file.
#             label_dict (dict): Dictionary mapping semantic segmentation classes to indices.
#             transpose_segmask (bool): Whether to transpose the segmentation mask.
#         """
#         self.wsi_path = wsi_path
#         self.csv_dir = csv_dir
#         self.segmentation_output_path = segmentation_output_path
#         self.label_dict = label_dict or {"Tumor": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}
#         self.transpose_segmask = transpose_segmask
#         self.all_classified_nuclei = []

#     def process_all_patches(self, parallel=False):
#         csv_files = [os.path.join(self.csv_dir, f) for f in os.listdir(self.csv_dir) if f.endswith('.csv')]
#         print(f"Found {len(csv_files)} CSV files to process.", flush=True)
#         if not csv_files:
#             print(f"No CSV files found in directory: {self.csv_dir}", flush=True)
#         results = []
#         for csv_file in tqdm(csv_files):
#             result = self.process_single_patch(csv_file)
#             results.append(result)
#         # Flatten the list of lists
#         self.all_classified_nuclei = [nucleus for sublist in results for nucleus in sublist]
#         print(f"Total nuclei classified: {len(self.all_classified_nuclei)}", flush=True)
#         return self.all_classified_nuclei


#     def process_single_patch(self, csv_file):
#         try:
#             print(f"Processing patch: {csv_file}", flush=True)
#             # Initialize NucleiSegmentationMask for this patch
#             nuclei_mask = NucleiSegmentationMask(csv_file)
#             # print(f"Initialized NucleiSegmentationMask for {csv_file}", flush=True)
        
#             # Load data for nuclei mask
#             nuclei_mask.load_data()
#             # print(f"Loaded nuclei data for {csv_file}", flush=True)
#             nuclei_mask.create_mask()
#             # print(f"Created nuclei mask for {csv_file}", flush=True)
        
#             # Use x_start, y_start, and patch_size from the nuclei_mask
#             x_start = nuclei_mask.x_start
#             y_start = nuclei_mask.y_start
#             patch_size = nuclei_mask.patch_size  # (height, width)
        
#             print(f"x_start: {x_start}, y_start: {y_start}, patch_size: {patch_size}", flush=True)
        
#             # Initialize NpyImagePlotter for this patch
#             semantic_plotter = NpyImagePlotter(
#                 file_path=self.segmentation_output_path,
#                 label_dict=self.label_dict,
#                 x_start=x_start,
#                 y_start=y_start,
#                 patch_size=patch_size,  # (height, width)
#                 transpose_segmask=self.transpose_segmask
#             )
#             semantic_plotter.load_data()
#             print(f"Loaded segmentation data for {csv_file}", flush=True)
        
#             # Read mpp
#             mpp = semantic_plotter.read_mpp_wsi(self.wsi_path)
#             print(f"Read mpp: {mpp} for {csv_file}", flush=True)
        
#             # Generate segmentation mask for this patch
#             segmentation_patch = semantic_plotter.generate_segmentation_mask()
#             print(f"Generated segmentation mask for {csv_file}", flush=True)
        
#             # Classify nuclei
#             classified_nuclei, nuclei_num_per_patch = nuclei_mask.classify_nuclei(
#                 segmentation_patch, mpp
#             )
#             print(f"Classified {nuclei_num_per_patch} nuclei in patch {csv_file}", flush=True)
#             return classified_nuclei
#         except Exception as e:
#             print(f"Error processing patch {csv_file}: {e}", flush=True)
#             import traceback
#             traceback.print_exc()
#             return []



#     def save_aggregated_results(self, output_dir):
#         """
#         Save aggregated results per class in JSON files.

#         Args:
#             output_dir (str): Directory to save the output results.
#         """
#         os.makedirs(output_dir, exist_ok=True)
#         # Aggregate data per class
#         class_aggregated_data = defaultdict(list)
#         for nucleus in self.all_classified_nuclei:
#             class_label = nucleus['majority_class']
#             class_aggregated_data[class_label].append(nucleus)

#         # Save per-class data with attributes as keys and lists of values
#         for class_label, nuclei_data in class_aggregated_data.items():
#             aggregated_attributes = defaultdict(list)
#             for nucleus in nuclei_data:
#                 for key, value in nucleus.items():
#                     if key != 'majority_class':
#                         aggregated_attributes[key].append(value)
#             output_file = os.path.join(output_dir, f'class_{int(class_label)}_nuclei.json')
#             print(f"Saving data for class {int(class_label)} with {len(nuclei_data)} nuclei to {output_file}")
#             with open(output_file, 'w') as f:
#                 json.dump(aggregated_attributes, f)
#             print(f"Saved data for class {int(class_label)} with {len(nuclei_data)} nuclei to {output_file}")

#         # Save overall statistics
#         total_nuclei = len(self.all_classified_nuclei)
#         nuclei_per_class = {str(int(class_label)): len(nuclei_data) for class_label, nuclei_data in class_aggregated_data.items()}
#         stats = {
#             'total_nuclei': total_nuclei,
#             'nuclei_per_class': nuclei_per_class
#         }
#         stats_file = os.path.join(output_dir, 'nuclei_stats.json')
#         with open(stats_file, 'w') as f:
#             json.dump(stats, f)
#         print(f"Saved nuclei statistics to {stats_file}")

# def main(args):
#     print(f"Starting main function with task: {args.task}", flush=True)
#     if args.task == "process_wsi":
#         # Ensure output directory exists
#         if args.output_dir is None:
#             print("Please specify an output directory using --output_dir", flush=True)
#             sys.exit(1)
#         os.makedirs(args.output_dir, exist_ok=True)

#         print("Initializing SegmentationVisualizer...", flush=True)
#         visualizer = SegmentationVisualizer(
#             wsi_path=args.wsi_path,
#             csv_dir=args.csv_dir,
#             segmentation_output_path=args.segmentation_output_path,
#             label_dict=None,  # Use default label_dict
#             transpose_segmask=args.transpose_segmask
#         )

#         print(f"CSV Directory: {args.csv_dir}", flush=True)

#         # Process all patches
#         print("Calling process_all_patches...", flush=True)
#         all_classified_nuclei = visualizer.process_all_patches(parallel=args.parallel)

#         # Save the aggregated results
#         print("Saving aggregated results...", flush=True)
#         visualizer.save_aggregated_results(args.output_dir)
#     else:
#         print(f"Unknown task: {args.task}", flush=True)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Segmentation Visualization Script")
#     parser.add_argument("--task", type=str, required=True, choices=["process_wsi"],
#                         help="Task to execute")
#     parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file")
#     parser.add_argument("--csv_dir", type=str, help="Path to the directory containing CSV files for multiple patches")
#     parser.add_argument("--segmentation_output_path", type=str, required=True, help="Path to the .npy file for semantic segmentation")
#     parser.add_argument("--output_dir", type=str, help="Directory to save the aggregated results")
#     parser.add_argument("--transpose_segmask", action="store_true", help="Transpose segmentation mask if needed")
#     parser.add_argument("--parallel", action="store_true", help="Process patches in parallel")
#     args = parser.parse_args()
#     main(args)
