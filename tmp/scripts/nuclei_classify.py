"""
Combining H&E, Nuclei Segmentation, and Semantic Segmentation for Visualization --> "Classifying each nuclei"

1) option to have figure of subplots of original H&E histopathology patch, nuclei segmentation mask, semantic segmentation mask, 
2) option to have an overlay of original H&E histopathology patch, nuclei segmentation mask, semantic segmentation mask together, 
3) option to display side by side original H&E histopathology patch, nuclei segmentation mask, semantic segmentation mask, 4) option to create a subfolder in a given directory to save original H&E histopathology patch, nuclei segmentation mask, semantic segmentation mask in individually without titles but only the images

"""

import os
import argparse
import matplotlib.pyplot as plt
from visualize_semantic_segmentation2 import WSISegmentVisualizer, NpyImagePlotter
from polygon_to_masks import NucleiSegmentationMask, QA_NucleiMaskAreaAnalysis
from tiatoolbox.wsicore.wsireader import WSIReader
from matplotlib.patches import Patch
import numpy as np
import matplotlib.cm as cm
from matplotlib.pyplot import imshow


class SegmentationVisualizer:
    def __init__(self, wsi_path, csv_path, segmentation_output_path, x_start, y_start, patch_size=4000, label_dict=None, transpose_segmask=False):
        """
        Initialize the SegmentationVisualizer with required paths and parameters.

        Args:
            wsi_path (str): Path to the WSI file for H&E extraction.
            csv_path (str): Path to the CSV file containing nuclei segmentation polygons.
            segmentation_output_path (str): Path to the semantic segmentation mask .npy file.
            x_start (int): Starting x-coordinate for the patch to extract.
            y_start (int): Starting y-coordinate for the patch to extract.
            patch_size (int): Size of the patch to extract.
            label_dict (dict): Dictionary mapping semantic segmentation classes to indices.
            transpose_segmask (bool): Whether to transpose the segmentation mask.
        """
        self.wsi_path = wsi_path
        self.csv_path = csv_path
        self.segmentation_output_path = segmentation_output_path
        self.patch_size = patch_size
        self.label_dict = label_dict or {"Tumor": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}

        # Initialize Nuclei and Semantic Plotter Objects
        self.nuclei_mask = NucleiSegmentationMask(
            csv_path, 
            patch_size=(patch_size, patch_size)
        )
        # QA AreaInPixels from the csv file 
        self.qa_nuclei_mask = QA_NucleiMaskAreaAnalysis(
            csv_path=self.csv_path,
            mask_path=None,
            output_path=self.segmentation_output_path,
            # offset=None,
            # original_data=None,
            # binary_mask = None,
            # areas_from_mask = [],
            # areas_from_csv = []
      
        )   

        # Determine x_start, y_start was given: flexible, if given, then that's it; if not, extract it from the csv polygon filename
        if x_start is None or y_start is None:
            x_start, y_start = self.nuclei_mask.offset
        self.x_start = x_start
        self.y_start = y_start

        self.semantic_plotter = NpyImagePlotter(
            file_path=segmentation_output_path,
            label_dict=self.label_dict,
            x_start=self.x_start,
            y_start=self.y_start,
            patch_size=patch_size,
            transpose_segmask=transpose_segmask  # Pass transpose option here
        )

        # Load data for nuclei and semantic masks
        self.nuclei_mask.load_data()
        self.nuclei_mask.create_mask()
        self.semantic_plotter.load_data()
        self.semantic_plotter.generate_segmentation_mask()
        self.segmentation_mask = self.semantic_plotter.generate_segmentation_mask()
        self.mpp = self.semantic_plotter.read_mpp_wsi(self.wsi_path)
        # self.wsi_patch = self._extract_he_patch()
        # self.overlay_nuclei_contours = self.nuclei_mask.overlay_contour(self.wsi_patch)

    def plot_combined_subplots(self, wsi_path, side_by_side = False, save_path=None):
        """
        Display subplots of the H&E patch, nuclei segmentation mask, and semantic segmentation mask.
        """
        # Extract the H&E patch
        he_patch = self._extract_he_patch()

        # Use overlay_segmentation_mask to get the semantic segmentation overlay
        semantic_overlay = self.semantic_plotter.overlay_segmentation_mask(self.wsi_path, show_side_by_side=False)
        print(f"semantic_overlay data type: {type(semantic_overlay)}")
        print(f"semantic_overlay shape: {semantic_overlay.shape}")
        # print(f"Actual semantic_overlay: {semantic_overlay}")
        # Create subplots to display H&E, Nuclei mask, and Semantic Segmentation Overlay
        fig, axes = plt.subplots(1, 3, figsize=(30,10))
        
        # Original H&E Patch
        axes[0].imshow(he_patch)
        axes[0].set_title("Original H&E Patch", fontsize=32)
        axes[0].axis("off")

        # Nuclei Segmentation Mask
        axes[1].imshow(self.nuclei_mask.mask, cmap='gray')
        axes[1].set_title("Nuclei Segmentation Mask", fontsize=32)
        axes[1].axis("off")

        # Semantic Segmentation Overlay
        axes[2].imshow(semantic_overlay)
        axes[2].set_title("Semantic Segmentation Overlay", fontsize=32)
        axes[2].axis("off")

        # Add a legend below the Semantic Segmentation Overlay
        label_color_dict = {
            label: (class_name, 255 * np.array(color))
            for class_name, label, color in zip(self.label_dict.keys(), self.label_dict.values(), plt.cm.Set1.colors)
        }
        legend_handles = [
            Patch(color=np.array(color) / 255, label=class_name)
            for label, (class_name, color) in label_color_dict.items()
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=len(self.label_dict), fontsize=28, title="Classes", title_fontsize=32)

        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=600)
            print(f"Combined subplot saved at {save_path}")
        plt.show()
      


    def overlay_segmentation_mask_nuclei_semantic(self, wsi_path, save_path=None):
        """
        Overlay the H&E patch with nuclei contours and the semantic segmentation mask.

        Args:
            wsi_path (str): Path to the WSI file.
            save_path (str, optional): Path to save the combined overlay image. Default is None.
        """
        # Generate the segmentation mask using NpyImagePlotter
        self.segmentation_mask = self.semantic_plotter.generate_segmentation_mask()
        print("Segmentation mask generated.")
        print("Segmentation mask shape:", self.segmentation_mask.shape)
        print("Unique values in segmentation mask:", np.unique(self.segmentation_mask))

        # Generate label-color dictionary for legend
        label_color_dict = {
            label: (class_name, 255 * np.array(color))
            for class_name, label, color in zip(self.label_dict.keys(), self.label_dict.values(), plt.cm.Set1.colors)
        }

        # Extract the H&E patch
        wsi_patch = self._extract_he_patch()
        print(f"H&E patch extracted with shape: {wsi_patch.shape}")

        # Overlay nuclei contours on H&E
        # self.overlay_nuclei_contours = self.nuclei_mask.overlay_contour(wsi_patch)
        self.overlay_nuclei_contours = self.nuclei_mask.overlay_contour(wsi_patch)

        print("Nuclei contours overlay created.")
        # print(f"Data type of overlay image: {type(self.overlay_nuclei_contours)}")
        print(f"Nuclei Contours Overlay image shape: {self.overlay_nuclei_contours.shape}")

        # Ensure segmentation mask shape matches the overlay shape
        if self.segmentation_mask.shape != self.overlay_nuclei_contours.shape[:2]:
            raise ValueError("Segmentation mask shape does not match overlay shape.")

        # Use create_overlay to overlay the segmentation mask onto the H&E patch with contours
        # final_overlay = self.semantic_plotter.create_overlay(wsi_patch, self.segmentation_mask,self.overlay_nuclei_contours, label_color_dict=label_color_dict, alpha=0.8)
        final_overlay = self.semantic_plotter.create_overlay_two_overlay(wsi_patch, self.segmentation_mask,self.overlay_nuclei_contours, label_color_dict=label_color_dict, alpha=0.8)

        # Next step: convert color of nuclei_contour based on the class of the nuclei where it's covered by from the semantic segmentation

        # Display the final overlay with legend
        plt.figure(figsize=(10, 10))
        plt.imshow(final_overlay)
        plt.title("Overlay of H&E with Nuclei Contours and Semantic Segmentation", fontsize=20)
        plt.axis("off")

        # Add a legend for the segmentation classes
        legend_handles = [
            Patch(color=np.array(color) / 255, label=class_name)
            for label, (class_name, color) in label_color_dict.items()
        ]
        plt.legend(handles=legend_handles, loc='lower center', ncol=len(self.label_dict), fontsize=12, title="Classes", title_fontsize=14)

        # Save the figure if a path is specified
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=400)
            print(f"Overlay image saved to {save_path}")

        plt.show()


    def color_nuclei_contours(wsi_patch, nuclei_contours, segmentation_mask, label_color_dict, show_legend=True):
        """
        Change the color of nuclei contours based on the corresponding semantic segmentation class and display a legend.

        Args:
            wsi_patch (np.ndarray): The WSI patch as a background image (shape: (H, W, 3)).
            nuclei_contours (np.ndarray): The nuclei contours overlay (shape: (H, W, 3)).
            segmentation_mask (np.ndarray): The segmentation mask (shape: (H, W)).
            label_color_dict (dict): Dictionary with label-color mappings.
            show_legend (bool): Whether to display a legend of class colors. Default is True.

        Returns:
            np.ndarray: The overlay image with class-colored nuclei contours.
        """
        # Debugging: Check type of label_color_dict immediately
        print("Type of label_color_dict at function start:", type(label_color_dict))
        print(f"label_color_dict is {label_color_dict}")
        # HOW DID THIS SUDDENTLY BECOME NUMPY ARRAY AND NOT A DICTIONARY ANYMORE?!!!!
        if not isinstance(label_color_dict, dict):
            raise TypeError("label_color_dict should be a dictionary.")

        # Make a copy of the WSI patch to overlay the colored nuclei contours
        final_overlay = np.copy(wsi_patch)

        # Debugging: Print label_color_dict contents before starting the loop
        print("Contents of label_color_dict before loop:", label_color_dict)

        for label, (class_name, color) in label_color_dict.items():
            # Identify where the segmentation mask matches the current label
            class_mask = (segmentation_mask == label)

            # Ensure the mask has three channels for RGB broadcasting
            if class_mask.ndim == 2:
                class_mask = np.stack([class_mask] * 3, axis=-1)

            # Directly set the nuclei contour color to the class color without blending
            for c in range(3):  # RGB channels
                final_overlay[..., c] = np.where(
                    class_mask[..., c] & (nuclei_contours[..., c] > 0),  # Apply only where there are contours
                    color[c],
                    final_overlay[..., c]
                )

        # Display the final image with the legend if required
        plt.figure(figsize=(10, 10))
        plt.imshow(final_overlay)
        plt.axis("off")

        if show_legend:
            legend_handles = [
                Patch(color=np.array(color) / 255, label=class_name)
                for label, (class_name, color) in label_color_dict.items()
            ]
            plt.legend(handles=legend_handles, loc='lower center', ncol=len(label_color_dict), fontsize=12,
                    title="Nuclei Contour Classes", title_fontsize=14, bbox_to_anchor=(0.5, -0.05))

        plt.show()

        return final_overlay



    def overlay_colored_nuclei_contours(self, wsi_path, mpp, save_path=None):
        """
        Overlay the H&E patch with nuclei contours and the semantic segmentation mask.

        Args:
            wsi_path (str): Path to the WSI fdile.
            save_path (str, optional): Path to save the combined overlay image. Default is None.
        """
        # Generate the segmentation mask using NpyImagePlotter
        self.segmentation_mask = self.semantic_plotter.generate_segmentation_mask()
        print("Segmentation mask generated.")
        print("Segmentation mask shape:", self.segmentation_mask.shape)
        print("Unique values in segmentation mask:", np.unique(self.segmentation_mask))

        # Define the label dictionary and color map
        label_color_dict = {
            label: (class_name, (int(255 * color[0]), int(255 * color[1]), int(255 * color[2])))
            for label, (class_name, color) in enumerate(zip(self.label_dict.keys(), plt.cm.Set1.colors))
        }
        print("overlay_label_color_dict confirmed as dictionary:", label_color_dict)

        # Extract the H&E patch
        wsi_patch = self._extract_he_patch()
        print(f"H&E patch extracted with shape: {wsi_patch.shape}")

        print(f"mpp: {self.mpp['power']}")
        # Overlay nuclei contours on H&E
        overlay_image, classified_nuclei, nuclei_num_per_patch = self.nuclei_mask.overlay_colored_contours(wsi_patch, self.segmentation_mask, self.label_dict, label_color_dict, self.mpp)
        print("Nuclei contours overlay created.")
        print(f"Data type of overlay image: {type(overlay_image)}")
        print(f"Nuclei Contours Overlay image shape: {overlay_image.shape}")
        print(f"Classified nuclei data type: {type(classified_nuclei)}")
        # print(f"Classified nuclei['AreaInPixels']: {classified_nuclei['AreaInPixels']}")



        # Display the final overlay with legend
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay_image)
        plt.title("Overlay of H&E with Class-Colored Nuclei Contours", fontsize=20)
        plt.axis("off")

        # Add a legend for the segmentation classes
        legend_handles = [
            Patch(color=np.array(color) / 255, label=class_name)
            for label, (class_name, color) in label_color_dict.items()
        ]
        plt.legend(handles=legend_handles, loc='lower center', ncol=len(self.label_dict), fontsize=12, title="Classes", title_fontsize=14)

        # Save the figure if a path is specified
        if save_path:
            # plt.savefig(save_path, bbox_inches="tight", dpi=1000)
            plt.savefig(save_path, bbox_inches="tight", dpi=600)
            print(f"Overlay image saved to {save_path}")

        plt.show()
        # for nucleus in classified_nuclei:
        #     print(f"AreaInPixels: {nucleus['AreaInPixels']}")

        print(f"This patch has {nuclei_num_per_patch} nuclei.")
        return overlay_image, classified_nuclei, nuclei_num_per_patch

    # QA AreaInPixels from classified_nuclei against the original AreaInPixels from the polygon csv 


    def extract_areas_from_classified_nuclei(self, classified_nuclei):
        """
        Extracts the AreaInPixels from the classified nuclei.

        Args:
            classified_nuclei (list): List of dictionaries containing classified nuclei information.

        Returns:
            list: List of AreaInPixels values.
        """
        return [nucleus['AreaInPixels'] for nucleus in classified_nuclei]

    def plot_overlapping_histogram(self, areas_from_classified_nuclei):
        """
        Plots an overlapping histogram of AreaInPixels from classified nuclei and original CSV.

        Args:
            areas_from_classified_nuclei (list): List of AreaInPixels values from classified nuclei.
        """
        mean_area_classified = np.mean(areas_from_classified_nuclei)
        mean_area_csv = np.mean(self.areas_from_csv)
        plt.figure(figsize=(12, 8))
        plt.hist(areas_from_classified_nuclei, bins=100, range=(0, 5000), alpha=0.5, label=f"Classified Nuclei (Mean Area: {mean_area_classified:.2f})", color='blue')
        plt.hist(self.areas_from_csv, bins=100, range=(0, 5000), alpha=0.5, label=f"CSV Data (Mean Area: {mean_area_csv:.2f})", color='green')
        plt.xlabel("Area in Pixels", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.title("Comparison of AreaInPixels Distributions", fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_path)
        plt.show()
        print(f"Histogram saved at {self.output_path}")
        print(f"Mean Area from Classified Nuclei: {mean_area_classified:.2f}")
        print(f"Mean Area from CSV: {mean_area_csv:.2f}")
        print(f"Difference in Mean Area: {abs(mean_area_classified - mean_area_csv):.2f} pixels")

    def run_analysis(self, segmentation_mask, color_map, mpp):
        """
        Runs the QA analysis by classifying nuclei and plotting the overlapping histogram.

        Args:
            segmentation_mask (numpy.ndarray): The semantic segmentation mask.
            color_map (dict): Dictionary mapping class indices to colors.
            mpp (float): Microns per pixel.
        """
        classified_nuclei, _ = self.classify_nuclei(segmentation_mask, color_map, mpp)
        areas_from_classified_nuclei = self.extract_areas_from_classified_nuclei(classified_nuclei)
        self.plot_overlapping_histogram(areas_from_classified_nuclei)

    def _extract_he_patch(self):
        """
        Helper method to extract H&E patch using TiaToolbox's WSIReader at 40x mpp.
        """
        print("Opening WSI file and extracting patch.")
        wsi_reader = WSIReader.open(self.wsi_path)
        mpp_40x = wsi_reader.convert_resolution_units(40, "power", "mpp")
        return wsi_reader.read_rect(location=(self.x_start, self.y_start), size=(self.patch_size, self.patch_size), resolution=mpp_40x, units="mpp")


def main(args):
    visualizer = SegmentationVisualizer(
        wsi_path=args.wsi_path,
        csv_path=args.csv_path,
        segmentation_output_path=args.segmentation_output_path,
        # commented out x_start, y_start since they are given in the init method (directly from the extraction of csv filename)
        x_start=args.x_start,
        y_start=args.y_start,
        patch_size=args.patch_size,
        transpose_segmask=args.transpose_segmask # Pass transpose option here
    )

    if args.task == "combined_subplots":
        visualizer.plot_combined_subplots(wsi_path=args.wsi_path,save_path=args.save_path)
    elif args.task == "overlay_combined":
        visualizer.overlay_segmentation_mask_nuclei_semantic(wsi_path=args.wsi_path,save_path=args.save_path)
    elif args.task == "side_by_side":
        visualizer.plot_side_by_side(save_path=args.save_path)
    elif args.task == "save_individual":
        visualizer.save_individual_images(save_dir=args.save_dir)
    elif args.task == "overlay_colored_nuclei_contours":
        visualizer.overlay_colored_nuclei_contours(wsi_path=args.wsi_path, save_path=args.save_path, mpp=visualizer.mpp)

    elif args.task == "qa_classified_nuclei":

        qa = visualizer.qa_nuclei_mask
        # Example usage
        _, classified_nuclei, nuclei_num_per_patch = visualizer.overlay_colored_nuclei_contours(wsi_path=args.wsi_path, save_path=args.save_path, mpp=visualizer.mpp)

        # Obtain 'AreaInPixels' from classified_nuclei regardless of class
        nuclei_classify_area = [nucleus['AreaInPixels'] for nucleus in classified_nuclei]
        print(f"Total number of nuclei in this patch: {nuclei_num_per_patch}")

        # The original csv file with the AreaInPixels
        original_nuclei_csv = visualizer.qa_nuclei_mask.load_data()
        nuclei_original_area = original_nuclei_csv['AreaInPixels']
        # print(f"nuclei_original_area: {nuclei_original_area}")

        # Plot the two distributions as overlapping histograms
        output_path = args.save_path
        # output_path = "/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/qa_nuclei_areainpixels_histogram.png"  # Specify the path to save the histogram
        qa.plot_overlapping_histogram_classified_nuclei(nuclei_classify_area, nuclei_original_area, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Visualization Script")
    parser.add_argument("--task", type=str, required=True, choices=["combined_subplots", 
                        "overlay_combined", "side_by_side", "save_individual", "overlay_colored_nuclei_contours",
                        "qa_classified_nuclei"],
                        help="Task to execute")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file for nuclei segmentation")
    parser.add_argument("--segmentation_output_path", type=str, required=True, help="Path to the .npy file for semantic segmentation")
    parser.add_argument("--x_start", type=int, help="Starting x-coordinate of the patch")
    parser.add_argument("--y_start", type=int, help="Starting y-coordinate of the patch")
    parser.add_argument("--patch_size", type=int, default=4000, help="Size of the patch to extract")
    parser.add_argument("--save_path", type=str, help="Path to save the plot (for single plot tasks)")
    parser.add_argument("--save_dir", type=str, help="Directory to save individual images when using save_individual")
    parser.add_argument("--transpose_segmask", action="store_true", help="Transpose segmentation mask if needed")

    args = parser.parse_args()
    main(args)
    

#Use Case

# 1. subplot figure with the original H&E patch, nuclei segmentation mask, and semantic segmentation mask.
# Works well! Code pushed
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task combined_subplots \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --x_start 108001 \
#     --y_start 44001 \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_combined_he_nuclseg_semseg.png \
#     --transpose_segmask

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task combined_subplots \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --x_start 108001 \
#     --y_start 44001 \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_combined_he_nuclseg_semseg2.png \
#     --transpose_segmask

# Works well, code pushed
# 2. overlay figure with the original H&E patch, nuclei segmentation mask, and semantic segmentation mask.
# this did not include the semantic segmentation mask overlay
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task overlay_combined \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --x_start 108001 \
#     --y_start 44001 \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_overlay_combined.png \
#     --transpose_segmask

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task overlay_combined \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --x_start 108001 \
#     --y_start 44001 \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_overlay_combined3.png \
#     --transpose_segmask


# python nuclei_classify.py --task overlay_combined \
#                           --wsi_path /path/to/wsi_file \
#                           --csv_path /path/to/csv_file \
#                           --segmentation_output_path /path/to/segmentation_mask.npy \
#                           --x_start 1000 \
#                           --y_start 1000 \
#                           --patch_size 4000 \
#                           --save_path /path/to/save_overlay.png

# 3. class colored nuclei contour overlap with the original H&E patch

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task overlay_colored_nuclei_contours \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_overlay_colored_nuclei_contours.png \
#     --transpose_segmask

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task overlay_colored_nuclei_contours \
#     --wsi_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs \
#     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_overlay_colored_nuclei_contours2.png \
#     --transpose_segmask

# Try on a TCGA-CESC case 
# /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task overlay_colored_nuclei_contours \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs \
#     --csv_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz/cesc_polygon/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/92001_8001_4000_4000_0.2501_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/92001_8001_4000_overlay_colored_nuclei_contours.png \
#     --transpose_segmask

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task overlay_colored_nuclei_contours \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs \
#     --csv_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz/cesc_polygon/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/136001_4001_4000_4000_0.2501_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --patch_size 4000 \
#     --save_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/136001_4001_4000_overlay_colored_nuclei_contours2.png \
#     --transpose_segmask

# LOOPING  THROUGH THIS ENTIRE CSV FOLDER --> THEN FIGURE OUT A WAY FOR DOING THIS IN ZIPPED FOLDER 
# KILL THE SAVED ON E DRIVE PROCESS, JUST SAVE ON ./DATA, SINCE WRITING ON E DRIVE IS SLOWING THINGS DOWN
# BUT CHECK HOW MUCH SPACE IS LEFT ON THE ./DATA, NEED TO TRANSFER TO BELUGA FROM TIME TO TIME 
# LAUNCH THE NARVAL, BELUGA, AND CEDAR  


# I'M HERE!!!
## THINK ABOUT WHETHER TO HAVE THE classified_nuclei outptus from overlay_colored_nuclei_contours saved for all patches 
### GOOD TO PROCESSING THE ENTIRE FOLDER OF POLYGONS (HAVE THAT FROM NARVAL TO HERE IF NEEDED) SINCE NO GPUS USED 

# QA the finally classified nuclei from a patch's AreaInPixels still matches with the original histogram from AreaInPixels from the original polygon csv 

# THEN RUN THIS ON THE ENTIRE FOLDER OF POLYGONS FOR THIS WSI, OPTIMIZE SPEED

# THEN TRANSFER THIS ONTO NARVAL 
# Only calculate, no save_path needed, --transpose_segmask is needed!

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task qa_classified_nuclei \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs \
#     --csv_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz/cesc_polygon/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/136001_4001_4000_4000_0.2501_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --transpose_segmask \
#     --save_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/136001_4001_4000_4000_qa_nuclei_areainpixels_histogram.png


# Try another patch
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
#     --task qa_classified_nuclei \
#     --wsi_path /Data/Yujing/Segment/tmp/tcga_cesc_svs/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs \
#     --csv_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz/cesc_polygon/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/92001_8001_4000_4000_0.2501_1-features.csv \
#     --segmentation_output_path /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
#     --transpose_segmask \
#     --save_path /Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/92001_8001_4000_4000_qa_nuclei_areainpixels_histogram.png


