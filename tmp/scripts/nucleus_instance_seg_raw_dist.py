#!/usr/bin/env python3
"""
Obtain the raw distributions of nuclei types after nucleus_instance_seg.py which uses HoverNet of TiaToolBox.
The outputs of the raw distributions from here should match with that of the semantic seg matching with the cesc_polygon workflow
before undergoing the histogram correction process to obtain the final distribution mean and sd
before passing to CoxPH model for survival analysis.

Author: Yujing Zou
Dec, 2024 
"""

# =============================
# Import Necessary Libraries
# =============================

import os
import sys
import json
import argparse
import logging
import warnings
from collections import defaultdict

import numpy as np
import cv2
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader

# =============================
# Configure Logger
# =============================

# Clear previous handlers for the logger to avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# Set logger level (optional: adjust as needed)
logger.setLevel(logging.INFO)

# =============================
# Color Dictionary for Visualization
# =============================

color_dict = {
    0: ("neoplastic epithelial", (255, 0, 0)),
    1: ("Inflammatory", (255, 255, 0)),
    2: ("Connective", (0, 255, 0)),
    3: ("Dead", (0, 0, 0)),
    4: ("non-neoplastic epithelial", (0, 0, 255)),
}

# =============================
# Utility Functions
# =============================

def load_pred_inst(dat_path):
    """
    Load the pred_inst dictionary from a .dat file using joblib.

    Args:
        dat_path (str): Path to the .dat file.

    Returns:
        dict: Loaded pred_inst dictionary.
    """
    if not os.path.exists(dat_path):
        logger.error(f"The pred_inst file does not exist: {dat_path}")
        sys.exit(1)
    try:
        wsi_pred = joblib.load(dat_path)
        logger.info("Number of detected nuclei: %d", len(wsi_pred))
        return wsi_pred
    except Exception as e:
        logger.error(f"Failed to load pred_inst from {dat_path}: {e}")
        sys.exit(1)

# =============================
# Classes
# =============================

class SegmentationVisualizer:
    def __init__(self, wsi_path, pred_inst=None, label_dict=None, color_dict=None):
        """
        Initialize the SegmentationVisualizer with required paths and parameters.

        Args:
            wsi_path (str): Path to the WSI file for H&E extraction.
            pred_inst (dict, optional): Dictionary containing instance information for the entire WSI.
            label_dict (dict, optional): Dictionary mapping semantic segmentation classes to indices.
            color_dict (dict, optional): Dictionary mapping class indices to (name, color).
        """
        self.wsi_path = wsi_path
        self.pred_inst = pred_inst  # Dictionary of nuclei
        self.label_dict = label_dict or {
            "neoplastic epithelial": 0,
            "Inflammatory": 1,
            "Connective": 2,
            "Dead": 3,
            "non-neoplastic epithelial": 4
        }
        self.color_dict = color_dict or {}
        self.all_classified_nuclei = []

    def read_mpp_wsi(self):
        """
        Read micrometers per pixel (mpp) directly from the WSI.

        Returns:
            float: Mean micrometers per pixel.
        """
        logger.info(f"Loading WSI: {self.wsi_path}")
        try:
            reader = WSIReader.open(self.wsi_path)
            mpp = reader.info.mpp  # Typically [mpp_x, mpp_y]
            logger.info(f"Micrometers per pixel: {mpp}")
            # Ensure mpp is a scalar value by taking the mean
            mpp_mean = np.mean(mpp)
            logger.info(f"Mean micrometers per pixel: {mpp_mean}")
            return mpp_mean
        except Exception as e:
            logger.error(f"Failed to read mpp from WSI: {e}")
            sys.exit(1)

    def process_pred_inst(self, mpp):
        """
        Process the pred_inst dictionary and classify nuclei.

        Args:
            mpp (float): Micrometers per pixel.

        Returns:
            list: List of classified nuclei.
        """
        if self.pred_inst is None:
            logger.error("pred_inst dictionary is not provided.")
            sys.exit(1)

        logger.info("Classifying nuclei from pred_inst dictionary...")
        classified_nuclei = []
        for instance_id, instance_data in self.pred_inst.items():
            contour = instance_data.get('contour', [])
            # Convert contour to a numpy array if it's not already
            if isinstance(contour, list):
                contour = np.array(contour, dtype=np.float32)
            elif not isinstance(contour, np.ndarray):
                logger.warning(f"Nucleus ID {instance_id} has invalid contour format. Skipping.")
                continue

            # Check if the contour array is empty
            if not contour.size:
                logger.warning(f"Nucleus ID {instance_id} has no contour data. Skipping.")
                continue  # Skip if no contour is provided

            # Convert contour to integer type for cv2
            try:
                contour_np = contour.astype(np.int32).reshape((-1, 1, 2))
            except ValueError:
                logger.warning(f"Invalid contour shape for nucleus ID {instance_id}. Skipping.")
                continue

            # Calculate area in pixels
            area_in_pixels = cv2.contourArea(contour_np)

            # Calculate perimeter
            perimeter = cv2.arcLength(contour_np, True)

            # Calculate area in square microns
            area_in_square_microns = area_in_pixels * (mpp ** 2)

            # Calculate equivalent radius
            equivalent_radius_microns = np.sqrt(area_in_square_microns / np.pi)

            # Get majority class (type) and probability
            majority_class = instance_data.get('type', -1)  # Default to -1 if not found
            prob = instance_data.get('prob', None)

            # Append to classified_nuclei
            nucleus = {
                'AreaInPixels': area_in_pixels,
                'perimeter': perimeter,
                'AreaInSquareMicrons': area_in_square_microns,
                'RadiusInMicrons': equivalent_radius_microns,
                'mpp': mpp,
                'majority_class': int(majority_class)
            }

            if prob is not None:
                nucleus['prob'] = prob

            classified_nuclei.append(nucleus)

        logger.info(f"Total nuclei classified from pred_inst: {len(classified_nuclei)}")
        self.all_classified_nuclei = classified_nuclei
        return classified_nuclei

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
            logger.info(f"Saving data for class {int(class_label)} with {len(nuclei_data)} nuclei to {output_file}")
            try:
                with open(output_file, 'w') as f:
                    json.dump(aggregated_attributes, f, indent=4)
                logger.info(f"Saved data for class {int(class_label)} with {len(nuclei_data)} nuclei to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save JSON file {output_file}: {e}")

        # Save overall statistics
        total_nuclei = len(self.all_classified_nuclei)
        nuclei_per_class = {str(int(class_label)): len(nuclei_data) for class_label, nuclei_data in class_aggregated_data.items()}
        stats = {
            'total_nuclei': total_nuclei,
            'nuclei_per_class': nuclei_per_class
        }
        stats_file = os.path.join(output_dir, 'nuclei_stats.json')
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            logger.info(f"Saved nuclei statistics to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics JSON file {stats_file}: {e}")

# =============================
# Main Function
# =============================

def main(args):
    """
    Main function to execute the script based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logger.info(f"Starting main function with task: {args.task}")

    if args.task == "process_wsi":
        # Ensure output directory exists
        if args.output_dir is None:
            logger.error("Please specify an output directory using --output_dir")
            sys.exit(1)
        os.makedirs(args.output_dir, exist_ok=True)

        # Determine input method: pred_inst_path
        input_method = None
        if args.pred_inst_path:
            input_method = "pred_inst"
        else:
            logger.error("No input data provided. Please specify --pred_inst_path.")
            sys.exit(1)

        if input_method == "pred_inst":
            # Load pred_inst dictionary
            pred_inst = load_pred_inst(args.pred_inst_path)

            # Initialize SegmentationVisualizer
            visualizer = SegmentationVisualizer(
                wsi_path=args.wsi_path,
                pred_inst=pred_inst,
                label_dict=None,  # Use default label_dict
                color_dict=color_dict  # Use the provided color_dict
            )

            # Intelligently obtain mpp directly from WSI
            mpp = visualizer.read_mpp_wsi()

            # Process the pred_inst dictionary
            logger.info("Processing pred_inst dictionary...")
            visualizer.process_pred_inst(mpp)

            # Adjust output directory based on wsi_path components
            logger.info("Adjusting output directory based on wsi_path...")

            # Extract the last two components from wsi_path
            wsi_path_components = os.path.normpath(args.wsi_path).split(os.sep)
            # subdir1 = wsi_path_components[-2]
            subdir2 = os.path.splitext(wsi_path_components[-1])[0]  # Remove file extension
            subdir2 = f"{subdir2}.svs"
            # Construct the new output directory path
            # output_subdir = os.path.join(args.output_dir, subdir1, subdir2)
            output_subdir = os.path.join(args.output_dir, subdir2)

            # Ensure the new output directory exists
            os.makedirs(output_subdir, exist_ok=True)

            logger.info(f"Output will be saved to: {output_subdir}")

            # Save the aggregated results
            logger.info("Saving aggregated results...")
            visualizer.save_aggregated_results(output_subdir)
            print(f"Aggregated results saved to: {output_subdir}")

        else:
            logger.error(f"Unsupported input method: {input_method}")
            sys.exit(1)

    else:
        logger.error(f"Unknown task: {args.task}")
        sys.exit(1)

# =============================
# Argument Parser
# =============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nuclei Classification and Statistics Generation Script")
    parser.add_argument("--task", type=str, required=True, choices=["process_wsi"],
                        help="Task to execute. Currently supported: process_wsi")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI (.svs) file")
    parser.add_argument("--pred_inst_path", type=str, required=True, help="Path to the pred_inst .dat file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the aggregated results")
    parser.add_argument("--color_dict_path", type=str, help="Path to a JSON file containing color_dict (optional)")
    # Removed --csv_dir, --segmentation_output_path, --mpp, --parallel, --num_workers as they are not needed for pred_inst input
    parser.add_argument("--transpose_segmask", action="store_true", help="Transpose segmentation mask if needed")

    args = parser.parse_args()

    # Optionally, load color_dict from a JSON file if provided
    if args.color_dict_path:
        if not os.path.exists(args.color_dict_path):
            logger.error(f"The color_dict file does not exist: {args.color_dict_path}")
            sys.exit(1)
        try:
            with open(args.color_dict_path, 'r') as f:
                loaded_color_dict = json.load(f)
            # Convert color tuples from lists to tuples if necessary
            for key in loaded_color_dict:
                loaded_color_dict[int(key)] = tuple(loaded_color_dict[key])
            color_dict = loaded_color_dict
            logger.info(f"Loaded color_dict from {args.color_dict_path}")
        except Exception as e:
            logger.error(f"Failed to load color_dict from {args.color_dict_path}: {e}")
            sys.exit(1)
    # Else, use the default color_dict defined above

    # Execute the main function
    main(args)

# =============================
# End of Script
# =============================

# USE EXAMPLE 
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nucleus_instance_seg_raw_dist.py \
#     --task process_wsi \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/a63b6131-6667-4a0a-88f3-e5ff1131175b/TCGA-C5-A1BN-01Z-00-DX1.75ED1BD9-C458-42D3-8A98-8FE41BB9CC2E.svs \
#     --pred_inst_path /Data/Yujing/Segment/tmp/tcga_svs_instance_seg/a63b6131-6667-4a0a-88f3-e5ff1131175b/TCGA-C5-A1BN-01Z-00-DX1.75ED1BD9-C458-42D3-8A98-8FE41BB9CC2E.svs/wsi_instance_segmentation_results/0.dat \
#     --output_dir /Data/Yujing/Segment/tmp/Instance_Segmentation/nuclei_instance_classify_results

# OPTIONAL 
    # --color_dict_path /path/to/custom_color_dict.json
