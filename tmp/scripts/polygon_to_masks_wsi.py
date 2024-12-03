#!/usr/bin/env python3
"""
Converting the polygon annotation to binary masks for each 4k by 4k patches of the WSI.

Author: Yujing Zou
"""

import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tiatoolbox.wsicore.wsireader import WSIReader
import torch
from tqdm import tqdm

class NucleiSegmentationMask:
    def __init__(self, csv_path, patch_size=(4000, 4000)):
        """
        Initialize the NucleiSegmentationMask with a CSV file path and tile size.
        """
        self.csv_path = csv_path
        self.patch_size = patch_size
        self.offset = self.extract_offset_from_filename(csv_path)  # Dynamically set the offset
        self.data = None
        self.mask = np.zeros(patch_size, dtype=np.uint8)

    def extract_offset_from_filename(self, filepath):
        """
        Extract the offset (x, y) from the filename.
        Assumes the filename format includes the offset as the first two numbers separated by underscores.
        """
        filename = os.path.basename(filepath)
        parts = filename.split("_")
        x_offset = int(parts[0])
        y_offset = int(parts[1])
        return (x_offset, y_offset)

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
        polygons = polygon_str.split(',')
        all_vertices = []
        for polygon in polygons:
            points = polygon.strip("[]").split(":")
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
                if all(0 <= x < self.patch_size[0] and 0 <= y < self.patch_size[1] for x, y in vertices):
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

    def classify_nuclei(self, segmentation_mask, color_map, mpp):
        """
        Classifies nuclei based on the majority class within each contour.
        """
        classified_nuclei = []
        contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nuclei_num_per_patch = len(contours)
        for contour in contours:
            contour_mask = np.zeros_like(self.mask, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 1, thickness=cv2.FILLED)
            contour_region = segmentation_mask[contour_mask == 1]
            if len(contour_region) == 0:
                continue
            unique, counts = np.unique(contour_region, return_counts=True)
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
