"""
This script is used to visualize the WSI and the mask side by side.
For visualization purpose in method Figure 1 of the manuscript.

Author: Yujing Zou
Date: Dec, 2024

"""

import argparse
import logging
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from tiatoolbox import logger
from tiatoolbox.wsicore.wsireader import WSIReader
import matplotlib.patches as patches

mpl.rcParams["figure.dpi"] = 600  # High-resolution figure
mpl.rcParams["figure.facecolor"] = "white"  # For better visibility
plt.rcParams.update({"font.size": 5})


class WSIVisualizer:
    def __init__(self, wsi_path, output_prefix, combos, power):
        self.wsi_path = wsi_path
        self.output_dir = output_prefix  # Final output directory
        self.combos = combos  # List of (x_start, y_start, patch_size) tuples
        self.power = power
        self.wsi = None
        self.scaling_factor = None

    def load_wsi(self):
        logger.info(f"Reading the WSI from {self.wsi_path}")
        self.wsi = WSIReader.open(input_img=self.wsi_path)
        print("Slide info:")
        pprint(self.wsi.info.as_dict())

    def calculate_scaling(self):
        # Calculate scaling factor based on magnification
        self.scaling_factor = 40 / self.power  # Scale down from 40x to user-defined power
        logger.info(f"Scaling factor calculated: {self.scaling_factor}")

    def save_thumbnail(self):
        # Save WSI thumbnail without any rectangles
        wsi_thumb = self.wsi.slide_thumbnail(resolution=self.power, units="power")
        output_path = os.path.join(self.output_dir, "thumbnail.png")
        os.makedirs(self.output_dir, exist_ok=True)
        plt.imsave(output_path, wsi_thumb)
        logger.info(f"Saved WSI thumbnail at: {output_path}")
        return wsi_thumb

    def save_tissue_mask(self):
        # Generate tissue mask and save
        logger.info("Generating tissue mask...")
        mask_reader = self.wsi.tissue_mask(resolution=self.power, units="power")
        mask_wsi = mask_reader.slide_thumbnail(resolution=self.power, units="power")

        output_path = os.path.join(self.output_dir, "tissue_mask.png")
        os.makedirs(self.output_dir, exist_ok=True)
        plt.imsave(output_path, mask_wsi, cmap="gray")
        logger.info(f"Saved tissue mask at: {output_path}")

    def save_combined_patch_overlay(self, wsi_thumb):
        # Save overlay with all patch rectangles on the thumbnail
        fig, ax = plt.subplots()
        ax.imshow(wsi_thumb)
        ax.axis("off")

        # Draw all patch rectangles
        for x_start, y_start, patch_size in self.combos:
            scaled_x = int(x_start / self.scaling_factor)
            scaled_y = int(y_start / self.scaling_factor)
            scaled_patch_size = int(patch_size / self.scaling_factor)
            rect = patches.Rectangle(
                (scaled_x, scaled_y), scaled_patch_size, scaled_patch_size,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            ax.add_patch(rect)

        output_path = os.path.join(self.output_dir, "wsi_selected_patches_overlay.png")
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        logger.info(f"Saved combined patch overlay at: {output_path}")
        plt.close()


def main():
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting WSI Visualization Script")

    parser = argparse.ArgumentParser(
        description="Visualize WSI and draw multiple patches for manuscript figures."
    )
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to the WSI file.")
    parser.add_argument("--output_prefix", type=str, required=True, help="Output directory for saving images.")
    parser.add_argument("--power", type=float, default=1.25, help="Magnification power (default: 1.25x).")
    parser.add_argument("--combos", nargs='+', required=True,
                        help="List of X_START, Y_START, PATCH_SIZE combos, e.g., '28001 68001 4000'")

    args = parser.parse_args()

    # Parse combos into tuples
    combos = []
    for combo in args.combos:
        split_combo = combo.split()
        if len(split_combo) == 3:
            combos.append(tuple(map(int, split_combo)))
        else:
            logger.error(f"Invalid combo format: {combo}. Expected 3 values.")
            exit(1)

    visualizer = WSIVisualizer(
        wsi_path=args.wsi_path,
        output_prefix=args.output_prefix,
        combos=combos,
        power=args.power,
    )

    visualizer.load_wsi()
    visualizer.calculate_scaling()
    wsi_thumb = visualizer.save_thumbnail()
    visualizer.save_tissue_mask()
    visualizer.save_combined_patch_overlay(wsi_thumb)

    logger.info("Visualization complete.")


if __name__ == "__main__":
    main()


# =============================================================================
# Slide info:
# {'axes': 'YXS',
#  'file_path': PosixPath('/Data/Yujing/Segment/tmp/tcga_cesc_svs/69037019-9df5-493e-8067-d4078d78e518/TCGA-MA-AA3X-01Z-00-DX1.44657CDB-53F1-4DED-AE54-2251118565EA.svs'),
#  'level_count': 4,
#  'level_dimensions': ((139440, 89100),
#                       (34860, 22275),
#                       (8715, 5568),
#                       (2178, 1392)),
#  'level_downsamples': [1.0, 4.0, 16.001077586206897, 64.01532962857414],
#  'mpp': (0.2525, 0.2525),
#  'objective_power': 40.0,
#  'slide_dimensions': (139440, 89100),
#  'vendor': 'aperio'}
# =============================================================================
