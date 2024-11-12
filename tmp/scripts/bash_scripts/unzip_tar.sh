#!/bin/bash

# TO UNZIP THE TAR FILES
# Path to the tar.gz file
TAR_FILE="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz"

# Destination directory
DEST_DIR="/Data/Yujing/Segment/tmp/tcga_cesc_visualizations/6edad00e-0e5b-42bc-a09d-ea81b1011c20/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs/TCGA-C5-A1MI-01Z-00-DX1.93D2D53E-C990-4CEE-AF61-A841A3798A74.svs.tar.gz"

# Unzip the tar.gz file
tar -xzvf "$TAR_FILE" -C "$DEST_DIR"