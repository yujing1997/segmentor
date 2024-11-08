"""
Visualize WSI inference results from semantic segmentations via TiaToolbox

Yujing Zou
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader
from skimage.io import imread, imsave
import multiprocessing as mp
import torch
import cupy as cp
from matplotlib.patches import Patch 

# Set matplotlib parameters for consistent visualization
plt.rcParams.update({"font.size": 5, "figure.dpi": 150, "figure.facecolor": "white"})

# Clear previous handlers for the logger
if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

class WSISegmentVisualizer:

    """
    Visualize various WSI segmentation outputs.
    and input svs WSI file
    
    """
    def __init__(self, wsi_path, output_path, expected_shape, label_dict, num_channels=5):
        self.wsi_path = wsi_path
        self.output_path = output_path
        self.num_channels = num_channels # Expected number of channels (e.g., 5 for 5 classes)
        self.label_dict = label_dict
        self.output = None
        self.expected_shape = expected_shape
        self.reader = None
        self.segmentation_mask = None
        self.logger = logging.getLogger(__name__)   

    def load_wsi(self, resolution, units="mpp"):
        logger.info(f"Loading WSI: {self.wsi_path}")
        self.reader = WSIReader.open(self.wsi_path)
        dimensions = self.reader.slide_dimensions(resolution, units)
        print("WSI Dimensions at", resolution, units, ":", dimensions)
        return dimensions
    
    def read_mpp_wsi(self):
        logger.info(f"Loading WSI: {self.wsi_path}")
        self.reader = WSIReader.open(self.wsi_path)
        # Access the mpp value of the WSI
        # mpp = self.reader.convert_resolution_units(40, "power", "mpp")
        mpp_40x = self.reader.convert_resolution_units(40, "mpp")
        print(f"Micrometers per pixel at 40x resolution: {mpp_40x}")

    @staticmethod
    def process_patch(patch_indices, output, patch_size):
        """Calculate min and max for a given patch."""
        start_idx = patch_indices * patch_size
        end_idx = start_idx + patch_size
        patch = output[start_idx:end_idx]  # Load only the patch
        return np.min(patch), np.max(patch)


    def load_output(self, chunk_size=4000):

        """
        Load the segmentation output file, ensuring it has the correct shape.
        Falls back to np.memmap if memory is insufficient and validates min/max values.

        Parameters:
            chunk_size (int): Number of rows to load at once when calculating min and max with memmap. 
                              Chose 4000 as default since that is the patch size of the Pan-Cancer-Nuclei-Seg

        Returns:
            output: The loaded or memory-mapped array.
            dtype: Data type of the array.
            min_val: Minimum value in the array.
            max_val: Maximum value in the array.

        """
        # Attempt full load to memory first
        try:
            self.output = np.load(self.output_path)  # Attempt full load
            min_val, max_val = np.min(self.output), np.max(self.output)
            print("Loaded full array into memory with shape:", self.output.shape)
            use_memmap = False
        except MemoryError:
            print("Memory insufficient; falling back to memory-mapped loading.")
            use_memmap = True
        
        if use_memmap:
            temp_output = np.load(self.output_path, mmap_mode="r")
            shape, dtype = temp_output.shape, temp_output.dtype

            # Check if reshaping is needed and save if necessary
            if shape != self.expected_shape:
                print(f"Warning: Expected shape {self.expected_shape} but found {shape}. Please verify.")
            
                if self._reshape_and_save_if_needed(shape, self.expected_shape, self.output_path):
                    # Load reshaped output
                    self.output_path = self.output_path.replace(".npy", "_reshaped.npy")
                    self.output = np.load(self.output_path)
                    shape = self.output.shape
                else:
                    print(f"Warning: Expected shape {self.expected_shape} but found {shape}. Please verify.")

            self.output = np.memmap(self.output_path, dtype=dtype, mode="r", shape=shape)
            print("Using memory-mapped array with shape:", self.output.shape)
            
            min_val, max_val = self._calculate_min_max_chunked(chunk_size)
        else:
            dtype = self.output.dtype

        print("Data type:", dtype)
        print("Min value:", min_val)
        print("Max value:", max_val)

        return self.output, dtype, min_val, max_val
    

    
    @staticmethod
    def _reshape_and_save_if_needed(current_shape, expected_shape, output_path):
        """
        Check if the current shape has swapped dimensions and, if so,
        reshape the array and save it with "_reshaped" added to the filename.

        Parameters:
            current_shape (tuple): Current shape of the array.
            expected_shape (tuple): Expected shape of the array.
            output_path (str): Path to the .npy file to reshape and save.

        Returns:
            bool: True if reshaping was performed and saved, False otherwise.
        """
        if current_shape[0] == expected_shape[1] and current_shape[1] == expected_shape[0] and current_shape[2] == expected_shape[2]:
            print("Swapped dimensions detected. Reshaping array to expected shape using memory mapping.")

            # Use memory mapping to load the array and avoid loading the entire file into memory
            data = np.memmap(output_path, dtype="float32", mode="r", shape=current_shape)
            new_output_path = output_path.replace(".npy", "_reshaped.npy")
    
            # Create a memory-mapped file for the reshaped array
            reshaped_data = np.memmap(new_output_path, dtype="float32", mode="w+", shape=expected_shape)

            # Copy data in chunks to avoid memory overload
            for i in range(current_shape[0]):  # Loop over the first dimension (width)
                reshaped_data[i, :, :] = data[:, i, :]  # Transpose by swapping axes

            # Flush to disk and close the memory-mapped file
            reshaped_data.flush()
            del reshaped_data
            print(f"Reshaped array saved to {new_output_path}")

            return True
        return False



    def _calculate_min_max_chunked(self, chunk_size):
        """Calculate min and max in chunks to ensure accurate values for memory-mapped data."""
        min_val, max_val = None, None
        for i in range(0, self.output.shape[0], chunk_size):
            # Load a chunk and compute min/max
            chunk = self.output[i:i + chunk_size]
            chunk_min, chunk_max = np.min(chunk), np.max(chunk)

            # Update global min/max based on chunk values
            min_val = chunk_min if min_val is None else min(min_val, chunk_min)
            max_val = chunk_max if max_val is None else max(max_val, chunk_max)

        # Ensure values are reasonable probabilities (0 to 1)
        if max_val > 1:
            print("Warning: Unexpected high max value; values should be within [0, 1].")
            max_val = min(max_val, 1)  # Cap to 1 if necessary

        return min_val, max_val
    
    def visualize_channel(self, channel_index):
        # Visualize a specific channel for the entire WSI segmentation output
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")
        plt.figure()
        plt.imshow(self.output[..., channel_index], cmap="twilight")
        plt.title(f"Channel {channel_index} - {self.get_label(channel_index)}")
        plt.colorbar()
        plt.show()
        

    # Visualize a patch of the segmentation output
    def visualize_channel_x_y_patch(self, channel_index, start_x=0, start_y=0, patch_size=4000):
        
        """
        Visualize a specific channel within a defined patch.
        
        Parameters:
            channel_index (int): Index of the channel to visualize.
            start_x (int): Starting x-coordinate of the patch.
            start_y (int): Starting y-coordinate of the patch.
            patch_size (int): Size of the patch to visualize (patch_size x patch_size).
        """
        # Check if output is loaded
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")

        # Define the patch boundaries
        end_x = start_x + patch_size
        end_y = start_y + patch_size

        # Ensure patch boundaries do not exceed the array dimensions
        if end_x > self.output.shape[1] or end_y > self.output.shape[0]:
            raise ValueError("Patch boundaries exceed output dimensions.")

        # Extract the specified patch for the chosen channel
        patch = self.output[start_y:end_y, start_x:end_x, channel_index]

        # Visualize the patch
        plt.figure(figsize=(6, 6))
        plt.imshow(patch, cmap="twilight")
        plt.title(f"Channel {channel_index} - {self.get_label(channel_index)} (Patch [{start_x}:{end_x}, {start_y}:{end_y}])")
        plt.colorbar()
        plt.show()
        
        # print shape of the patch
        print(f"Patch shape: {patch.shape}")
        
        return patch.shape

    def save_channel_images(self, save_dir):
        # Save each channel as an image
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")
        os.makedirs(save_dir, exist_ok=True)
        for i in range(self.output.shape[-1]):
            plt.imsave(f"{save_dir}/channel_{i}.png", self.output[..., i], cmap="twilight_shifted")
            print(f"Saved channel {i} as {save_dir}/channel_{i}.png")

    # save_channel_images for specific patches with x_start and y_start and patch_size parameters 
    def save_channel_images_x_y_patch(self, save_dir, start_x=0, start_y=0, patch_size=4000):
        
        """
        Save a specified patch of each channel as an image.

        Parameters:
            save_dir (str): Directory to save the channel images.
            start_x (int): X-coordinate for the start of the patch.
            start_y (int): Y-coordinate for the start of the patch.
            patch_size (int): Size of the square patch to extract and save.
        """
        # Ensure output is loaded
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Define the patch region
        end_x = start_x + patch_size
        end_y = start_y + patch_size

        # Loop over each channel, extract the patch, and save it
        for i in range(self.output.shape[-1]):
            # Extract patch for the current channel
            channel_patch = self.output[start_y:end_y, start_x:end_x, i]

            # Save the patch as an image
            plt.imsave(f"{save_dir}/channel_{i}_patch_{start_x}_{start_y}_{patch_size}.png", channel_patch, cmap="twilight_shifted")
            print(f"Saved channel {i} patch as {save_dir}/channel_{i}_patch_{start_x}_{start_y}_{patch_size}.png")
            

    def get_label(self, channel_index):
        # Retrieve the label name based on the channel index, which class it represents 
        return [label for label, idx in self.label_dict.items() if idx == channel_index][0]

    def plot_class_histogram(self,save_class_hist):
        # Plot histogram for each class distribution
        plt.figure(figsize=(15, 25))
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")
        class_counts = [np.sum(self.output[..., i]) for i in range(self.output.shape[-1])]
        plt.figure()
        plt.bar(self.label_dict.keys(), class_counts)
        plt.xlabel("Class", fontsize=28)
        plt.ylabel("Pixel Count", fontsize=28)
        plt.xticks(rotation=12,fontsize=15)
        plt.yticks(fontsize=15)
        plt.title("Class Distribution",fontsize=28)
        plt.show()
        # save histogram 
        plt.savefig(save_class_hist,dpi=600)
    
    def plot_channel_distribution(self,save_channel_prob_hist):
        """
        Plot and save the distribution of pixel intensities (probability maps) for each channel.
        
        Parameters:
            save_class_hist (str): Path to save the histogram image.
        """
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")
        
        # Initialize the plot
        plt.figure(figsize=(20, 15))
    
        # Plot the distribution for each channel/class
        for i in range(self.output.shape[-1]):
            channel_data = self.output[..., i].flatten()
            plt.hist(channel_data, bins=100, alpha=0.6, label=self.get_label(i), density=True)
        # Set plot labels and title
        plt.xlabel("Probability", fontsize=32)
        plt.ylabel("Density", fontsize=32)
        plt.title("Distribution of Probability Maps for Each Class", fontsize=30)
        plt.legend(title="Class", fontsize=28)
        plt.grid(False)
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        
        # Show plot
        plt.show()
        
        # Save the histogram
        plt.savefig(save_channel_prob_hist,dpi=600)
        print(f"Channel probability distribution histogram saved to {save_channel_prob_hist}")

    def generate_segmentation_mask(self):
        """Generate the segmentation mask by selecting the class with the highest probability for each pixel."""
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")
        
        self.segmentation_mask = np.argmax(self.output, axis=-1)
        self.logger.info("Generated segmentation mask with shape: %s", self.segmentation_mask.shape)
        
        # Log dimensions of each channel
        for i in range(self.output.shape[-1]):
            self.logger.info("Channel %d min: %.3f, max: %.3f", i, np.min(self.output[..., i]), np.max(self.output[..., i]))
        
        return self.segmentation_mask

    def save_segmentation_mask(self, save_path):
        """Save the generated segmentation mask to the specified path."""
        if self.segmentation_mask is None:
            raise ValueError("Segmentation mask not generated. Please call generate_segmentation_mask() first.")
        
        imsave(save_path, self.segmentation_mask.astype(np.uint8))
        self.logger.info("Segmentation mask saved to %s", save_path)

    def visualize_probability_maps_with_subplots(self, save_path=None):
        """Visualize and optionally save the raw probability maps for each class using subplots."""
        if self.output is None:
            raise ValueError("Output not loaded. Please call load_output() first.")
        
        num_classes = len(self.label_dict)
        fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 8))  # Adjust width to accommodate each class
        cmap = "viridis"
        
        for i, (label, index) in enumerate(self.label_dict.items()):
            ax = axes[i]
            im = ax.imshow(self.output[..., index], cmap=cmap)
            ax.set_title(label, fontsize=18)
            ax.axis("off")
        
        # Add a single color bar below all subplots
        cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
        cbar.set_label("Probability", fontsize=16)
        cbar.ax.tick_params(labelsize=14)
        
        fig.suptitle("Probability Maps for Each Class", fontsize=22, y=0.85)
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight",dpi=600)
            self.logger.info("Probability maps saved to %s", save_path)
        
        plt.show()

    def visualize_segmentation_results(self, save_path=None):
        """Visualize and optionally save the raw probability maps and the processed segmentation mask side-by-side."""
        if self.segmentation_mask is None:
            self.generate_segmentation_mask()

        tile = imread(self.wsi_path)
        self.logger.info("Input image dimensions: (%d, %d, %d)", *tile.shape)
        
        # Plot probability maps for each class
        fig1, axes1 = plt.subplots(1, len(self.label_dict), figsize=(15, 5))
        for i, (label, index) in enumerate(self.label_dict.items()):
            ax = axes1[i]
            ax.imshow(self.output[..., index], cmap="viridis")
            ax.set_title(label)
            ax.axis("off")
        fig1.suptitle("Probability Maps for Each Class", y=0.65)
        
        # Save if specified
        if save_path:
            probability_maps_path = save_path.replace(".png", "_probability_maps.png")
            plt.savefig(probability_maps_path,dpi=600)
            self.logger.info("Probability maps saved to %s", probability_maps_path)

        plt.show()

        # Plot original tile and segmentation mask
        fig2 = plt.figure(figsize=(10, 5))
        
        # Original Image
        ax1 = fig2.add_subplot(1, 2, 1)
        ax1.imshow(tile)
        ax1.set_title("Original Tile")
        ax1.axis("off")
        
        # Segmentation Mask
        ax2 = fig2.add_subplot(1, 2, 2)
        ax2.imshow(self.segmentation_mask, cmap="tab10")
        ax2.set_title("Segmentation Mask")
        ax2.axis("off")
        
        fig2.suptitle("Processed Prediction Map", y=0.82)

        # Save if specified
        if save_path:
            segmentation_results_path = save_path.replace(".png", "_segmentation_results.png")
            plt.savefig(segmentation_results_path,dpi=600)
            self.logger.info("Segmentation results saved to %s", segmentation_results_path)
        
        plt.show()

# Test visualization of a 2D slice from a output segmentation prob map .npy file
# Only uses CPUs

class NpyImagePlotter:

    def __init__(self, file_path, channels=None, x_start=0, y_start=0, patch_size=4000, label_dict=None, full_image=False,transpose_segmask=True):
        """
        Initialize the plotter with parameters.

        Args:
            file_path (str): Path to the .npy file.
            channels (list of int, optional): Channels to visualize. Only required for npy_plot. Default is None.
            x_start (int): Starting x-coordinate for the patch.
            y_start (int): Starting y-coordinate for the patch.
            patch_size (int): Size of the patch to extract.
        """
        self.file_path = file_path
        self.channels = channels if channels is not None else []
        self.x_start = x_start
        self.y_start = y_start
        self.patch_size = patch_size
        self.full_image = full_image
        self.transpose_segmask = transpose_segmask
        self.data = None
        self.segmentation_mask = None
        self.label_dict = label_dict if label_dict is not None else {}

    def load_data(self):
        """Load the .npy file data with memory mapping, transposing it if needed."""
        try:
            self.data = np.load(self.file_path, mmap_mode='r')
            print("Data loaded with memory mapping.")

            # Transpose the data if transpose_segmask is True
            if self.transpose_segmask:
                self.data = np.transpose(self.data, (1, 0, 2))  # Swap the first two axes
                print("Data transposed to match WSI dimensions.")

        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")


    def extract_patch(self):
        """Extract a patch from the loaded data based on x_start, y_start, and patch_size."""
        if self.data is None:
            self.load_data()

        if self.full_image:
            patch = self.data  # Return the entire data if full image is selected
        else:
            patch = self.data[self.x_start:self.x_start + self.patch_size,
                              self.y_start:self.y_start + self.patch_size, :]

        # Transpose the patch if the transpose_segmask option is enabled
        if self.transpose_segmask:
            patch = patch.transpose((1, 0, 2))  # Swap x and y axes for the patch
            print(f"Transposed patch with shape: {patch.shape}")
        else:
            print(f"Extracted patch with shape: {patch.shape}")

        return patch    

    def extract_slices(self):
        """Extract 2D slices of specified size from the loaded data for each channel."""
        if self.data is None:
            print("Data not loaded. Call load_data() first.")
            return None

        # Dictionary to store slices for each channel
        slices = {}
        for channel in self.channels:
            slices[channel] = self.data[self.x_start:self.x_start + self.patch_size, 
                                        self.y_start:self.y_start + self.patch_size, 
                                        channel]
            print(f"Extracted slice for channel {channel} with shape: {slices[channel].shape}")

        return slices

    # def generate_segmentation_mask(self):
    #     """Generate segmentation mask by selecting the class with the highest probability for each pixel."""
    #     if self.data is None:
    #         self.load_data()

    #     # If full_image is True, generate mask for the entire image
    #     if self.full_image:
    #         print("Generating segmentation mask for the entire image.")
    #         self.segmentation_mask = np.argmax(self.data, axis=-1)

    #         # Transpose the segmentation mask if specified
    #         if self.transpose_segmask:
    #             self.segmentation_mask = self.segmentation_mask.T
    #             print("Segmentation mask transposed for entire image.")
        
    #     # If full_image is False, generate mask for a specific patch
    #     else:
    #         patch = self.extract_patch()
    #         self.segmentation_mask = np.argmax(patch, axis=-1)

    #     # Transpose the segmentation mask for the patch if specified
    #     if self.transpose_segmask:
    #         self.segmentation_mask = self.segmentation_mask.T
    #         print("Segmentation mask transposed for patch.")

    #     print("Segmentation mask shape:", self.segmentation_mask.shape)
    #     return self.segmentation_mask
    def generate_segmentation_mask(self):
        """Generate the segmentation mask by selecting the class with the highest probability for each pixel."""
        if self.data is None:
            self.load_data()

        # Generate mask for the entire image or a specific patch
        if self.full_image:
            self.segmentation_mask = np.argmax(self.data, axis=-1)
        else:
            patch = self.extract_patch()
            self.segmentation_mask = np.argmax(patch, axis=-1)

        # Rotate the mask by 90 degrees counterclockwise if transpose_segmask is set
        if self.transpose_segmask:
            self.segmentation_mask = np.rot90(self.segmentation_mask, k=4)  # 90 degrees counterclockwise
            print("Segmentation mask rotated by 90 degrees counterclockwise.")

        print("Segmentation mask shape:", self.segmentation_mask.shape)
        return self.segmentation_mask



    def plot_segmentation_mask(self, save_path=None):
        """
        Plot the generated segmentation mask with custom colors and legend.

        Args:
            save_path (str, optional): Path to save the segmentation mask image. Default is None.
        """
        if self.segmentation_mask is None:
            self.generate_segmentation_mask()
        
        # Generate label-color dictionary
        print("Generating label-color dictionary for segmentation classes.")
        label_color_dict = {}
        colors = plt.cm.get_cmap("Set1").colors  # Use Set1 colormap for distinct colors
        for class_name, label in self.label_dict.items():
            label_color_dict[label] = (class_name, 255 * np.array(colors[label]))

        # Create an RGB image for the segmentation mask based on label_color_dict
        overlay = np.zeros((*self.segmentation_mask.shape, 3), dtype=np.uint8)
        for label, (class_name, color) in label_color_dict.items():
            mask = self.segmentation_mask == label
            overlay[mask] = color

        # Plot the segmentation mask with colors
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title("Segmentation Mask" if self.full_image else "Patch Segmentation Mask", fontsize=32)
        plt.axis("off")

        # Add legend to the plot
        legend_handles = [
            Patch(color=np.array(color) / 255, label=class_name)
            for label, (class_name, color) in label_color_dict.items()
        ]
        plt.legend(handles=legend_handles, loc='lower center', ncol=len(self.label_dict), fontsize=12, title="Classes", title_fontsize=14)

        # Save the figure if a path is specified
        if save_path:
            plt.savefig(save_path, bbox_inches="tight",dpi=600)
            print(f"Segmentation mask image saved to {save_path}")
        
        plt.show()
        

    def overlay_segmentation_mask(self, wsi_path, show_side_by_side=False, save_path=None):
        """
        Display either just the segmentation overlay or a side-by-side comparison of the WSI patch and the segmentation overlay.

        Args:
            wsi_path (str): Path to the WSI file.
            show_side_by_side (bool): If True, show the original WSI patch and overlay side-by-side. Default is False.
            save_path (str, optional): Path to save the comparison image. Default is None.
        """
        # Ensure the segmentation mask has been generated and transposed if needed
        if self.segmentation_mask is None:
            self.generate_segmentation_mask()
        # Generate label-color dictionary using self.label_dict
        print("Generating label-color dictionary for segmentation classes.")
        label_color_dict = {}
        colors = cm.get_cmap("Set1").colors
        for class_name, label in self.label_dict.items():
            label_color_dict[label] = (class_name, 255 * np.array(colors[label]))

        # Open the WSI file with WSIReader and obtain the 40x mpp value
        print("Opening WSI file and obtaining 40x mpp resolution.")
        wsi_reader = WSIReader.open(wsi_path)
        mpp_40x = wsi_reader.convert_resolution_units(40, "power", "mpp")
        print(f"Using 40x resolution in mpp: {mpp_40x}")
        # print the dimension of the wsi file
        print("WSI dimensions:", wsi_reader.slide_dimensions(mpp_40x, "mpp"))

        # Use read_rect to directly extract the patch from the WSI at the desired coordinates
        print(f"Extracting WSI patch at coordinates (x_start={self.x_start}, y_start={self.y_start}) with size {self.patch_size}x{self.patch_size} at 40x mpp resolution.")
        wsi_patch = wsi_reader.read_rect(location=(self.x_start, self.y_start), size=(self.patch_size, self.patch_size), resolution=mpp_40x, units="mpp")
        print(f"WSI patch extracted with shape: {wsi_patch.shape}")

        # Generate the overlay
        print("Creating segmentation overlay.")
        overlay = self.create_overlay(wsi_patch, self.segmentation_mask, label_color_dict)

        # Display based on the show_side_by_side flag
        if show_side_by_side:
            print("Displaying side-by-side comparison.")
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))

            # Left subplot: Original H&E WSI patch
            axes[0].imshow(wsi_patch)
            axes[0].set_title("Original H&E WSI Patch", fontsize=16)
            axes[0].axis("off")

            # Right subplot: Overlay of the segmentation mask
            axes[1].imshow(overlay)
            axes[1].set_title("Segmentation Overlay", fontsize=16)
            axes[1].axis("off")

            # Add a legend below the side-by-side plot
            legend_handles = [
                Patch(color=np.array(color) / 255, label=class_name)
                for label, (class_name, color) in label_color_dict.items()
            ]
            fig.legend(handles=legend_handles, loc='lower center', ncol=len(self.label_dict), fontsize=12, title="Classes", title_fontsize=14)
            fig.suptitle("WSI Patch and Semantic Segmentation Overlay", fontsize=20)
        else:
            print("Displaying only the segmentation overlay.")
            plt.figure(figsize=(6, 6))
            plt.imshow(overlay)
            plt.title("Segmentation Overlay", fontsize=16)
            plt.axis("off")

            # Add legend to the single overlay plot
            legend_handles = [
                Patch(color=np.array(color) / 255, label=class_name)
                for label, (class_name, color) in label_color_dict.items()
            ]
            plt.legend(handles=legend_handles, loc='lower center', ncol=len(self.label_dict), fontsize=12, title="Classes", title_fontsize=14)

        # Save the figure if a path is specified
        if save_path:
            plt.savefig(save_path, bbox_inches="tight",dpi=600)
            print(f"Image saved to {save_path}")
        
        plt.show()

    def create_overlay(self, wsi_patch, mask, label_color_dict, alpha=0.5):
        """
        Create an overlay of the segmentation mask on the WSI patch.

        Args:
            wsi_patch (np.ndarray): The WSI patch as a background image.
            mask (np.ndarray): The segmentation mask.
            label_color_dict (dict): Dictionary with label-color mappings.
            alpha (float): Transparency level for the overlay.

        Returns:
            np.ndarray: The overlay image.
        """
        overlay = np.copy(wsi_patch)

        for label, (class_name, color) in label_color_dict.items():
            # Create a mask for the current class
            class_mask = (mask == label)

            # Overlay the color with specified transparency
            for c in range(3):  # Assuming RGB channels
                overlay[..., c] = np.where(class_mask, 
                                        overlay[..., c] * (1 - alpha) + color[c] * alpha, 
                                        overlay[..., c])

        return overlay
    

    def plot_slices(self, slices, label_dict=None, save_path=None):
        """
        Plot the extracted 2D slices for each channel as subplots and optionally save the image.

        Args:
            slices (dict of int: np.ndarray): Dictionary of slices to plot by channel.
            label_dict (dict, optional): Dictionary mapping channel indices to class names.
            save_path (str, optional): Path to save the plot. Default is None.
        """
        if not slices:
            print("No data to plot.")
            return

        # Invert label_dict if provided, to map channel index to class name
        if label_dict:
            label_dict = {v: k for k, v in label_dict.items()}
        
        num_channels = len(slices)
        fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 9))
        if num_channels == 1:
            axes = [axes]  # Ensure axes is iterable even for a single subplot

        # Plot each channel with corresponding class name
        for i, (channel, slice_data) in enumerate(slices.items()):
            ax = axes[i]
            im = ax.imshow(slice_data, cmap='viridis')
            # Use label from label_dict if available, otherwise fallback to "Channel {channel}"
            class_name = label_dict.get(channel, f"Channel {channel}")
            ax.set_title(class_name, fontsize=32)
            ax.axis("off")

        # Add a single color bar for all subplots
        cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
        # cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.05, pad=0.1)

        cbar.set_label("Probability", fontsize=32)
        cbar.ax.tick_params(labelsize=28)
        # fig.suptitle("Probability Maps for Each Class", fontsize=35, y=0.85)
        fig.suptitle(
            f"Probability Maps for Each Class\n"
            f"x_start: {self.x_start}, y_start: {self.y_start}, patch_size: {self.patch_size}",
            fontsize=35, y=0.85
        )

    
        # Save the figure if a path is specified
        if save_path:
            plt.savefig(save_path, bbox_inches="tight",dpi=600)
            print(f"Plot saved to {save_path}")
        
        plt.show()


    def generate_and_save_patch_masks(self, save_dir, patch_size=10000):
        """
        Generate and save segmentation masks as patches.

        Args:
            save_dir (str): Directory to save the patch segmentation masks.
            patch_size (int): Size of each patch to process and save.
        """
        if self.data is None:
            self.load_data()

        height, width, num_channels = self.data.shape
        print(f"Image dimensions: {height} x {width} with {num_channels} channels.")

        # Iterate over the image in steps of patch_size
        for y_start in range(0, height, patch_size):
            for x_start in range(0, width, patch_size):
                # Define the patch region
                patch = self.data[y_start:y_start + patch_size, x_start:x_start + patch_size, :]
                
                # Generate segmentation mask for the current patch
                segmentation_patch = np.argmax(patch, axis=-1)
                print(f"Generated segmentation mask for patch at ({x_start}, {y_start}) with shape: {segmentation_patch.shape}")
                
                # Plot and save the segmentation mask for the current patch
                self.plot_and_save_segmentation_patch(segmentation_patch, x_start, y_start, patch_size, save_dir)

    def plot_and_save_segmentation_patch(self, segmentation_patch, x_start, y_start, patch_size, save_dir):
        """
        Plot and save a segmentation patch.

        Args:
            segmentation_patch (np.ndarray): The segmentation mask for the patch.
            x_start (int): Starting x-coordinate of the patch.
            y_start (int): Starting y-coordinate of the patch.
            patch_size (int): Size of the patch.
            save_dir (str): Directory to save the patch image.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Define the file name with x_start, y_start, and patch size information
        save_path = os.path.join(save_dir, f"segmentation_mask_x{x_start}_y{y_start}_size{patch_size}.png")
        
        # Plot and save the patch segmentation mask
        plt.figure(figsize=(10, 10))
        plt.imshow(segmentation_patch, cmap="tab10")
        plt.title(f"Segmentation Mask (x={x_start}, y={y_start}, size={patch_size})", fontsize=16)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight",dpi=600)
        print(f"Saved segmentation patch to {save_path}")
        plt.close()


def main(args):
    # Define label dictionary
    label_dict = {"Tumour": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}

    # Initialize the visualizer object
    # visualizer = WSISegmentVisualizer(args.wsi_path, args.output_path, label_dict)
    # Parse expected_shape as a tuple from the command-line argument
    expected_shape = tuple(map(int, args.expected_shape.split(',')))

    # Initialize the visualizer object with the expected shape and label dictionary
    visualizer = WSISegmentVisualizer(
        wsi_path=args.wsi_path,
        output_path=args.output_path,
        expected_shape=expected_shape,
        label_dict=label_dict,
        # num_channels=args.num_channels
    )
    
    # Execute task based on argument
    if args.task == "load_wsi":
        visualizer.load_wsi(args.resolution)
        
    
    if args.task == "read_mpp_wsi":
        visualizer.read_mpp_wsi()

    # elif args.task == "load_output":
    #     visualizer.load_output()
    elif args.task == "load_output":
        output, dtype, min_val, max_val = visualizer.load_output(chunk_size=args.chunk_size)
        print("Output Shape:", output.shape)
        print("Data Type:", dtype)
        print("Min Value:", min_val)
        print("Max Value:", max_val)

    # Perform task based on args.task
    if args.task == "npy_plot":

        # Define label dictionary
        label_dict = {"Tumour": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}
        
        # Instantiate NpyImagePlotter and execute the plot task
        plotter = NpyImagePlotter(
            file_path=args.npy_file_path,
            channels=args.channels,
            x_start=args.x_start,
            y_start=args.y_start,
            patch_size=args.patch_size
        )
        plotter.load_data()
        slices = plotter.extract_slices()
        plotter.plot_slices(slices, label_dict=label_dict, save_path=args.save_path)

    elif args.task == "segmentation_overlay":

        # Define label dictionary
        label_dict = {"Tumour": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}
        # Instantiate NpyImagePlotter for segmentation overlay task
        plotter = NpyImagePlotter(
            file_path=args.npy_file_path,
            x_start=args.x_start,
            y_start=args.y_start,
            patch_size=args.patch_size,
            label_dict=label_dict,  # Pass label_dict here
            # full_image=args.full_image # Optional, set to False for patch-based overlay
            # HAVE THE NEXT transpose_segment line ONLY if you want to transpose the segmentation mask x & y from the original way
            transpose_segmask=args.transpose_segmask #
        )

        # Generate segmentation mask and overlay it on the WSI overview
        plotter.load_data()
        plotter.generate_segmentation_mask()
        plotter.overlay_segmentation_mask(
            wsi_path=args.wsi_path,
            show_side_by_side=args.show_side_by_side,  # Pass the flag here
            save_path=args.overlay_save_path
        )

    # Handle the new task for plotting the full segmentation mask
    elif args.task == "plot_full_segmentation_mask" or args.full_image:
        # Plot full segmentation mask
        plotter = NpyImagePlotter(
            file_path=args.npy_file_path,
            label_dict=label_dict,
            full_image=args.full_image  # Pass full image mode here
        )
        plotter.load_data()
        plotter.generate_segmentation_mask()
        plotter.plot_segmentation_mask(save_path=args.save_path)
    
    elif args.task == "visualize_channel":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.visualize_channel(args.channel_index)
    
    elif args.task == "visualize_channel_x_y_patch":
        visualizer.load_output() # Load output
        visualizer.visualize_channel_x_y_patch(
            channel_index = args.channel_index, 
            start_x=args.start_x, 
            start_y=args.start_y, 
            patch_size=args.patch_size
        )
        
    elif args.task == "save_patch_segmentation_masks":
        plotter = NpyImagePlotter(
            file_path=args.npy_file_path,
            label_dict=label_dict
        )
        plotter.load_data()
        plotter.generate_and_save_patch_masks(
            save_dir=args.save_dir,
            patch_size=args.patch_size  # Patch size can be adjusted as needed
        )


    elif args.task == "save_channel_images":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.save_channel_images(args.save_dir)
    
    elif args.task == "save_channel_images_x_y_patch":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.save_channel_images_x_y_patch(
            save_dir=args.save_dir, 
            start_x=args.start_x, 
            start_y=args.start_y, 
            patch_size=args.patch_size
        )

    elif args.task == "plot_class_histogram":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.plot_class_histogram(args.save_class_hist)
    elif args.task == "plot_channel_distribution":
        visualizer.load_output()
        visualizer.plot_channel_distribution(args.save_channel_prob_hist)

    elif args.task == "visualize_probability_maps_with_subplots":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.visualize_probability_maps_with_subplots(args.save_visualization)

    elif args.task == "generate_segmentation_mask":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.generate_segmentation_mask()
        if args.save_segmentation_mask:
            visualizer.save_segmentation_mask(args.save_segmentation_mask)

    elif args.task == "visualize_segmentation_results":
        visualizer.load_output()  # Ensure output is loaded
        visualizer.visualize_segmentation_results(args.save_visualization)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="WSI Segmentation Visualizer")
    parser.add_argument("--wsi_path", type=str, required=False, help="Path to the WSI file")
    parser.add_argument("--output_path", type=str, required=False, help="Path to the segmentation output (.npy file)")
    parser.add_argument("--resolution", type=float, default=0.25, help="Resolution for WSI loading")
    parser.add_argument("--task", type=str, required=True,
                        choices=["load_wsi", "read_mpp_wsi", "load_output", "visualize_channel", 
                                 "visualize_channel_x_y_patch", "save_channel_images", "save_channel_images_x_y_patch", 
                                 "plot_class_histogram","plot_channel_distribution","generate_segmentation_mask",
                                 "visualize_probability_maps_with_subplots", "generate_segmentation_mask", 
                                 "visualize_segmentation_results","npy_plot","segmentation_overlay","plot_full_segmentation_mask",
                                 "save_patch_segmentation_masks"],
                        help="Task to execute")
    
    parser.add_argument("--channel_index", type=int, default=0, help="Channel index for visualization (for visualize_channel task)")
    parser.add_argument("--start_x", type=int, default=0, help="Starting x-coordinate of the patch (for visualize_channel_x_y_patch)")
    parser.add_argument("--start_y", type=int, default=0, help="Starting y-coordinate of the patch (for visualize_channel_x_y_patch)")
    parser.add_argument("--patch_size", type=int, default=512, help="Size of the patch to visualize (for visualize_channel_x_y_patch)")
    parser.add_argument("--expected_shape", type=str, default="135168,105472,5", help="Expected shape of the output as 'height,width,channels'")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for processing min and max with memory mapping")

    # New arguments for npy_plot
    parser.add_argument("--npy_file_path", type=str, help="Path to the .npy file for npy_plot task.")
    parser.add_argument("--channel", type=int, default=0, help="Channel to extract for npy_plot.")
    parser.add_argument("--x_start", type=int, default=0, help="Starting x-coordinate for npy_plot.")
    parser.add_argument("--y_start", type=int, default=0, help="Starting y-coordinate for npy_plot.")
    parser.add_argument("--save_path", type=str, help="Path to save the extracted 2D slice image.")
    # parser.add_argument("--channels", type=int, nargs='+', default=[0], help="Channels to extract and visualize for npy_plot.")
    parser.add_argument("--channels", type=int, nargs='+', required="npy_plot" in sys.argv, help="Channels to extract and visualize for npy_plot")
    parser.add_argument("--overlay_save_path", type=str, help="Path to save the WSI overlay image.")
    parser.add_argument("--show_side_by_side", action="store_true", help="If set, shows a side-by-side comparison of the original WSI patch and segmentation overlay")
    parser.add_argument("--full_image", action="store_true", help="If set, processes the entire segmentation mask image.")
    parser.add_argument("--transpose_segmask", action="store_true", help="If set, transposes the segmentation mask along x and y.")

    
    parser.add_argument("--save_dir", type=str, default="./visualizations", help="Directory to save channel images")
    parser.add_argument("--save_class_hist", type=str, default="./class_histogram.png", help="Path to save class histogram")
    parser.add_argument("--save_channel_prob_hist", type=str, default="./channel_prob_histogram.png", help="Path to save channel probability histogram")
    parser.add_argument("--save_segmentation_mask", type=str, help="Path to save the segmentation mask if generated")
    parser.add_argument("--save_visualization", type=str, help="Path to save visualizations if applicable")

    args = parser.parse_args()
    main(args)



# USE EXAMPLES

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/visualize_semantic_segmentation2.py \
#     --task segmentation_overlay \
#     --npy_file_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --x_start 72001 \
#     --y_start 44001 \
#     --patch_size 10000 \
#     --overlay_save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/wsi_semanticseg_sidebyside_image_72001_44001_10000_transposedmask2.png \
#     --show_side_by_side \
#     --transpose_segmask


# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/visualize_semantic_segmentation2.py \
#    --task segmentation_overlay \
#    --npy_file_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#    --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#    --x_start 108001 \
#    --y_start 44001 \
#    --patch_size 4000 \
#    --overlay_save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/wsi_semanticseg_sidebyside_image_108001_44001_4000_transposedmask_SANITY.png \
#    --show_side_by_side \
#    --transpose_segmask

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/visualize_semantic_segmentation2.py \
#     --task segmentation_overlay \
#     --npy_file_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --x_start 72001 \
#     --y_start 24001 \
#     --patch_size 4000 \
#     --overlay_save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/wsi_semanticseg_sidebyside_image_72001_24001_4000_transposedmask_SANITY2.png \
#     --show_side_by_side \
#     --transpose_segmask


# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/visualize_semantic_segmentation.py \
#     --task segmentation_overlay \
#     --npy_file_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --x_start 72001 \
#     --y_start 24001 \
#     --patch_size 4000 \
#     --overlay_save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/wsi_semanticseg_sidebyside_image_72001_24001_4000_transposedmask_SANITY.png \
#     --show_side_by_side \
#     --transpose_segmask