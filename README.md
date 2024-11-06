# TiaToolbox 

# Semantic Segmentation 

# How to install gdc-client for data transfer from TCGA data 

- `conda create -n gdc-client-env python=3.8 -y`
- `conda activate gdc-client-env`
- `wget https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip`
- `unzip /home/yujing/gdc-client_2.3_Ubuntu_x64.zip`
- `gdc-client --version`: now can see gdc-client as an executable 
- `mv gdc-client /home/yujing/miniconda3/envs/gdc-client-env/bin/`
- `gdc-client --version`

# To download TCGA data example 
- `conda activate gdc-client-env`
- manifest file path: `/home/yujing/dockhome/Multimodality/Segment/tmp/manifest/gdc_manifest.2024-11-03.txt`
- cd to where you'd like the 
- `gdc-client download -m/home/yujing/dockhome/Multimodality/Segment/tmp/manifest/gdc_manifest.2024-11-03.txt`

# Semantic segmetation 
- on test slides `./dockhome/Multimodality/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs`
- scripts: 
    - dockhome/Multimodality/Segment/tmp/scripts/semantic_segmentation.py
    - dockhome/Multimodality/Segment/tmp/scripts/visualize_semantic_segmentation.py
        - ./visualize_semantic_segmentation.py visualizes the segmentation outpus 
        - WSI outputs 5 channels of probability maps for each class: label_dict = {"Tumour": 0, "Stroma": 1, "Inflammatory": 2, "Necrosis": 3, "Others": 4}

1. Must map pixel by pixel of semantic segmentation output to the Pan-Cancer-Nuclei-Seg .csv 4k by 4k files 
2. Merge semantic segmentation output to the Pan-Cancer-Nuclei-Seg .csv 4k by 4k files
- Once a pixel class is obtained, within a QAed segmented nuclei (number of pixels), majority vote for classification of nuclei class
    - PixelInAreas was already reported in the Pan-Cancer-Nuclei-Seg .csv files
    - Output for each patch, PixelInAreas vector for each class
    - Output for each WSI: concatenation of PixelInAreas vectors for each class ffrom each patch 
