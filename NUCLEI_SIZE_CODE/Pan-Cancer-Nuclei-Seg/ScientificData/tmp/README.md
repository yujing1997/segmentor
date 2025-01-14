
# Pan-Cancer-Nuclei-Seg 10 cancer types (Scientific Data paper)
- Create nuclei segmentation masks from the results Polygon column csv files 
    - class `NucleiSegmentationMask` from `./tmp/scripts/polygon_to_masks.py`

- To QA PixelInArea from a converetd binary mask matches with the originial results of PixelInArea in csv 
    - class `QA_NucleiMaskAreaAnalysis` from `./tmp/scripts/polygon_to_masks.py` 


## Download original svs file from TCGA
- compare if nuclei segmentation mask is accurate for this patch (subplots): done

## Activate gdc-client virtual environment
module load StdEnv/2020  
module load python/3.8
source ~/env_gdc/bin/activate


## Run gdc-client download 
- `gdc-client download -m/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/manifest/gdc_manifest.2024-11-03.txt`
- `gdc-client download -m/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/manifest_files/gdc_manifest.2024-11-11.txt`
    - cd /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc
- there are some .csv from the statistical plan `/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/Results/Harry_df/DATAFRAME_MERGED.csv` that didn't exist in the first round of manifest download from the ../gdc_manifest.2024-11-11.txt, it's because some werene't "DX..." in the filename like the rest of the batch 
    - this new batch has been downloaded to `/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/manifest_files/gdc_manifest.2024-11-17.txt`
     - cd /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc
     - `gdc-client download -m/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/manifest_files/gdc_manifest.2024-11-17.txt`
     - `gdc-client download -m/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/manifest_files/gdc_manifest.2024-11-18.txt`
     - `gdc-client download -m/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/manifest_files/gdc_manifest.2024-11-19.txt`

## Semantic Segmentaion (TiaToolbox)
- on this same patch, compare if tumor segmentation mask is accurate for this patch (subplots)


## Determine if a segmented nuclei pixels overlap with semantic segmented classes (%?)
- what to do for cases where parts of the nucleus is in two different classes? Majority vote?

## Visual inspection for classed segmented nuclei 

## Histogram of nuclei size for each class 

## Run this on a few more patches from each results folder 

## "Match" and "Join" the nuclei segmentation results with the semantic segmentation results
- concatenate nuclei area for the whole patch after processing through each entire svs polygon folders

## For reproducibility purpose
- run Pan-Cancer-Nuclei-Seg nuclei segmentation code on new slides in case future work needs to be done on new slides 

# Merging with the nuclei seg with the semantic seg
with patch /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/blca_polygon/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs/108001_44001_4000_4000_0.2277_1-features.csv, since it has more tumor regions, to match the nuclei segmentation with the semantic segmentation 

# Source of Scientific Data paper: 
- [link](https://stonybrookmedicine.app.box.com/v/cnn-nuclear-segmentations-2019/folder/75593818392?page=7)



# To activate HIPT_Embedding_Env
module load StdEnv/2020 opencv hdf5 geos/3.10.2
source ~/HIPT_Embedding_Env/bin/activate
module load python/3.10 gcc/9.3.0 arrow/13
echo "HIPT_Embedding_Env Python environment activated"



# To make a new virtual env for semantic segmentation 
Since the cv2 errors from the HIPT_Embedding_Env, we shall make a new virtual environment 

- Create the virtual env
pip install --user virtualenv
virtualenv ~/envs/semanticseg310
- Activate the virtual env
module load StdEnv/2023
module load python/3.10.13  # Adjust version if needed
source ~/envs/semanticseg310/bin/activate
- Activate Huizhong's environment 
module load gcc/12.3 python/3.10.13 opencv/4.10.0  hdf5/1.14.2 geos/3.12.0
source envs/semanticseg_hz/bin/activate



## request for GPU on interactive session
# request salloc GPU Narval 

- To launch a GPU job
`sbatch --time=1:00:00 --account=rrg-bengioy-ad --gres=gpu:1 job.sh`

- To get an interactive session
<!-- `salloc --time=3:00:00 --account=def-senger --gres=gpu:1` -->
salloc --time=03:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=32G
    - Resulted in > 1.5 per WSI estimated. 
    - Proton only took ~18 min per WSI --> ended this salloc session adn requesting more 
salloc --time=03:00:00 --gres=gpu:2 --cpus-per-task=16 --mem=48G
salloc --time=01:00:00 --gres=gpu:2 --cpus-per-task=8 --mem=64G
salloc --time=03:00:00 --account=def-senger --gres=gpu:2 --cpus-per-task=16 --mem=32G
salloc --time=03:00:00 --account=def-senger --gres=gpu:1 --cpus-per-task=16 --mem=32G
salloc --time=7:00:00 --account=def-senger --gres=gpu:1 --cpus-per-task=4 --mem=16G
salloc --time=05:00:00 --account=def-senger --gres=gpu:1 --cpus-per-task=4 --mem-per-cpu=4G


- In case the session died, and you want to get back to the srun of the requested interactive sessoin 
    - squeue -u yujingz: to obtain the NODELIST to locate the interactive node
    - ssh <NODELIST> (ex.ssh ng30707)
    - 
- check how much VRAM you've requested
    - nvidia-smi
    - nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv

- Setting up batch size for GPU memory
    - Observe Maximum Utilization: When memory usage is close to capacity (e.g., around 38–39 GB for a 40 GB A100), this batch size is likely optimal
        - batch size = available GPU memory/memory per sample
        - what is one sample? 


# In an interactive session, test out memories with GPU allocations 
- for a job array, the entire job array slurm cannot be launched. However, you can defined some
    - export SLURM_ARRAY_TASK_ID=4 #should match row four of the SAMPLE_SHEET 
    - bash /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/scripts/bash_scripts/semantic_segmentation3_narval_YZ.sh

## within folder /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc

find if subfolder with name of a given string exists
find /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc -type d -name "your_subfolder_name"

find /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc -type d -name "8c3b3974-8bf4-4c97-8783-1bfbadeae50c"


find /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc -type d -name "04bf9ef4-b261-4df9-9b7a-b9f6d7fb1d45"

## To see all the specific jobs within the job array launched 

- squeue -a -r | egrep '(yujingz|harryg)'    |   nl

## Command to check a number of files in a folder
cd /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc: Is a directory you want to see the number of files in 
 - ls -1 | wc -l
 - outptus: 159 (as of 12:51AM 2024-11-14)

cd /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks2/tcga_cesc
 - ls -l | wc -l
 - outputs: 5 (as of 12:52AM 2024-11-14)

The rest of outputs were on Proton: 
cd /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask
 - ls -l | wc -l
 - outputs: 12 (as of 12:52AM 2024-11-14)

cd /Data/Yujing/Segment/tmp/tcga_cesc_semantic_mask_qa
 - ls -l | wc -l
 - outputs: 33 (as of 12:52AM 2024-11-14)

Job still running on Narval (as of 12:52AM 2024-11-14)
 -  9
Total semantic segmentation to be done: 
 - 159 + 5 + 12 + 33 + 9 = 218
 - Not sure where the discrepancies between the total 

================================================================
Check how many subfolders there are in the wsi_path parent dir:
cd /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc
 - ls -l | wc -l
 - outputs: 230 including one manifest_files folder, so 229 
================================================================


## nuclei_classify.py, making this work with a WSI from patch level
- Adjusting paths from Proton to file systems in Narval
- Unzipping the polygon folder --> compute --> delete the unzipped folder (in tmp folder)
    - comparing to speed of reading files within zipped folder without unzipping it & deleting it 
    - Or, the crude way (benefiting speed but not storage): unzip the folder, run the code, but not delete the folder in case deleting takes time 
    - the first unzipped folder was 214MB, for 250 unzipped folders, it would be 53.5GB, which is ok
    - but put the unzipped folder in the `/home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/something...` folder

This is the syntax for working on Proton (working example)

python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_classify.py \
     --task overlay_colored_nuclei_contours \
     --wsi_path /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/53a336f8-63b0-4333-ac0c-18eb56dbe430/TCGA-EA-A1QT-01Z-00-DX1.3CF9DD48-AC8F-4DE1-A155-5F2DE5A58976.svs \
     --csv_path /home/yujing/dockhome/Multimodality/Segment/tmp/blca_polygon/108001_44001_4000_4000_0.2277_1-features.csv \
     --segmentation_output_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/wsi_segmentation_results2_0.2277mpp_40x/0.raw.0.npy \
     --patch_size 4000 \
     --save_path /Data/Yujing/Segment/tmp/blca_svs/30e4624b-6f48-429b-b1d9-6a6bc5c82c5e/visualizations/patch_108001_44001_4000/108001_44001_4000_overlay_colored_nuclei_contours.png \
     --transpose_segmask

     - Will need to streamline the wsi_path (like for the segment bash scripts)
     - Need to save results and append from per-patch for WSI, currently csv_path handles a single path of polygon csv file --> need to parallelize if possible 
     - segmentation_output_path needs to be streamlined like for the segment bash scripts
     - save_path does not need to be enabled since it is for visualization purposes only
     - --transpose_segmask must be kept

     - Need to double check with calculation of area methods comparisons to the previous JGH data code version, how does the correction factor apply? Need to add another 
     - Added RadiusInMicrons in the classify_nuclei method in NucleiSegmentationMask class in ./polygon_to_masks.py 
     - The correction histogram step is for after the raw distributions are obtained

# nuclei_classify.py & polygon_to_masks.py
- working versions for per patch inputs and outputs, handles plotting etc. 
- need to give absolute paths, examples at the bottom of script.

# nuclei_classify_wsi.py 
- developmenet/working versions for WSI inputs and outputs for raw distribution results, no plotting 
- associated with ./tmp/scripts/bash_scrip/nuclei_classify_extract.sh
    - where a job array with the number tasks being the # of rows in SAMPLE_SHEET submitted to Narval. 
    - some patches had an IndexError for originally used for plotting (I think!) for nuclei_classify_wsi.py, but QAed with the original nuclei_classify_wsi, shows the same number of classified nuclei for the IndexError patches. Need to surveillence!!

- check how many files there are in a folder 
cd to that folder, then ls -1 | wc -l

# For semantic segmentation:
- run_semantic_seg_for_loop2.sh was from Proton's semantic segmentation for loop run (not for Narval)
- For Narval, it's specified in the ./scripts/bash_scripts/sbatch_it.txt 

# Finding the number subdirectories in a given parent folder
- find /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc -maxdepth 1 -type d | wc -l
- find /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon -maxdepth 1 -type d | wc -l
- find /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc -maxdepth 1 -type d | wc -l
- find /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc -maxdepth 1 -type d | wc -l
- find /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/nuclei_classify_results -maxdepth 1 -type d | wc -l

# Look for specific existence of subfolder with string mentions from a given parent folder 
    - For example, I want to see if the subfolder with string "TCGA-C5-A907-01Z-00-DX1" that exists in `/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/run_partition/progress_track/2024-11-17/matched_cases_cesc_polygon_20241117.tsv`: 
- ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon/*/ | grep 'TCGA-C5-A907-01Z-00-DX1'
        - this returns the path that exists 
    - For a case that might not exist: TCGA-DG-A2KK-01Z-00-DX1.248D92C3-AD5B-4F3C-AE5C-6F8C70C7AE87.svs
- ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon/*/ | grep 'TCGA-DG-A2KK-01Z-00-DX1.248D92C3-AD5B-4F3C-AE5C-6F8C70C7AE87.svs'

# Found the cases that existed in cesc_polygon folder but was not included in the statistical plan
- ran script: `/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/Results/Harry_df/check_cesc_polygon_statsplan.py`
- The cases that didn't exist in the original statistiacal plan are in `/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/Results/Harry_df/ROUND6/cesc_polygon_subfolders_notyetdone_20241117.tsv`
- Check: /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/Results/Harry_df/ROUND6/only_in_tsv2.tsv 
    - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon/*/ | grep 'TCGA-VS-A9V4-01Z-00-DX1.F71A69E3-050E-4CA9-A074-BAA03739D542.svs'
    - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/*/ | grep 'TCGA-VS-A9V4-01Z-00-DX1.F71A69E3-050E-4CA9-A074-BAA03739D542.svs'
    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep 'TCGA-VS-A9V4-01Z-00-DX1.F71A69E3-050E-4CA9-A074-BAA03739D542.svs'

    - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon/*/ | grep 'TCGA-VS-A9V3-01Z-00-DX1.6625C7BA-2F8A-41E9-871B-EF8C96999A74.svs'
    - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/*/ | grep 'b7d512db-6be3-4c2a-a6f0-ca4b96cb0c8c'
    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep 'TCGA-VS-A9V3-01Z-00-DX1.6625C7BA-2F8A-41E9-871B-EF8C96999A74.svs'
    
        - Check if the following ones were already done: `/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/Results/Harry_df/ROUND6/only_in_tsv2.tsv` 
    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep 'b0286604-6834-46d9-a7c8-55fe5f8ec1cc' (exists)

    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep 'b7d512db-6be3-4c2a-a6f0-ca4b96cb0c8c' (does not exist)
       **!!!** - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon/*/ | grep 'TCGA-VS-A9UY-01Z-00-DX1.6EB5B19F-98C8-4B9C-8FF4-F53ACD163BE5.svs' (cesc_polygon exists, semantic mask needs generating!) 

    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep 'e4ec449c-992e-437a-90bc-b9e1336997bd' (exists)
    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep '395ebb75-aef5-4c42-89d6-858730aab44f' (exists)
    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep '5f85d494-ec97-4b75-b041-f70638d6d7f2' (does not exist)
       **!!!** - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/data-files/cesc_polygon/*/ | grep 'TCGA-VS-A9UM-01Z-00-DX1.D921E19D-7C09-4543-9940-09B3FBA3624B.svs' (cesc_polygon exists, semantic mask needs generating!)

    - ls -d /home/yujingz/projects/rrg-senger-ab/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/semantic_masks/tcga_cesc/*/ | grep '05c3cb03-5eb7-4b48-acfb-f66abab9d502' (exists)

- Check those that existed in cesc_polygon but were not in /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/manifest_files/gdc_sample_sheet.2024-11-11.tsv or /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/run_partition/progress_track/2024-11-17/run2_2024_11_17.tsv:

    - saved at: /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/tmp/Results/Harry_df/ROUND6/not_existing.tsv 
        - They need to be downloaded from TCGA-CESC GDC again! Need to find their outcomes if possible!
        - check they for sure don't yet exist in the svs-files path 
        - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/*/ | grep '93b70704-2174-4a05-b91f-b763fc61ee20'
        - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/*/ | grep 'bc5d40d2-2539-4e02-91c0-4e20df41246b'
        - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/*/ | grep 'b7d512db-6be3-4c2a-a6f0-ca4b96cb0c8c'
    - to search for subfolder that might be nested at any level within the parent folder
        
        - find /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/ -type d -name '*TCGA-XS-A8TJ*'
        - ls -d /home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/svs-files/tcga_cesc/*/ | grep 'TCGA-VS-A9UY-01Z-00-DX1.6EB5B19F-98C8-4B9C-8FF4-F53ACD163BE5.svs'


- Need to download another set of tcga-cesc svs files that weren't downloaded
    - `gdc-client download -m/home/yujingz/scratch/NUCLEI_SIZE_CODE/Pan-Cancer-Nuclei-Seg/ScientificData/filemaps/tcga_cesc/manifest_files/gdc_manifest.2024-11-19.txt`
    - then need to get the other two DNE cases a rerun in this time's run SAMPLE_SHEET 

### Initially done on Cedar 

# Running the last 19 cases whose cesc_polygon files did not already exist in the ScientificData 
- their docker for producing the nuclei segmentation binary mask did not work
- therefore, the hovernet instance segmentation pre-trained model from TiaToolbox was used to obtain the classified nuclei for those 19 cases. The following bash scripts chain was used: 
    1. `run_instance_seg_nuclei_Cedar_YZ.sh`: to run the instance segmentation on Cedar for a list of cases
    2. `nucleus_instance_seg_raw_dist_Cedar.sh`: accepts the .dat result from `run_instance_seg_nuclei_Cedar_YZ.sh` and produces the raw distribution of nuclei for each case
    3. `run_nuclei_distribution_analyzer.sh`: accepts the raw distribution of nuclei for each case and produces the corrected histogram for the nuclei size distribution mean and std for each case 

# Distribution analysis chain 
- Cases whose cesc_polygons from the ScientificData paper is available used semantic segmentation, then match with the nuclei segmentation, then raw distribution --> histogram correction for final results: 
    - 

- The ones whose cesc_polygons did not exist, followed the instance segmentation route, then raw distribution --> histogram correction for final results: 
    - `/home/yujingz/scratch/NUCLEI_SIZE_CODE/ScientificData/tmp_Narval/tmp/scripts/bash_scripts/run_instance_seg_nuclei_Cedar_YZ.sh`
        - used `/home/yujingz/scratch/NUCLEI_SIZE_CODE/ScientificData/tmp_Narval/tmp/scripts/nucleus_instance_seg.py`
    - `/home/yujingz/scratch/NUCLEI_SIZE_CODE/ScientificData/tmp_Narval/tmp/scripts/bash_scripts/nucleus_instance_seg_raw_dist_Cedar.sh`
        - used `/home/yujingz/scratch/NUCLEI_SIZE_CODE/ScientificData/tmp_Narval/tmp/scripts/nucleus_instance_seg_raw_dist.py`
    - `/home/yujingz/scratch/NUCLEI_SIZE_CODE/ScientificData/tmp_Narval/tmp/scripts/bash_scripts/run_nuclei_distribution_analyzer.sh`
        - used `/home/yujingz/scratch/NUCLEI_SIZE_CODE/ScientificData/tmp_Narval/tmp/scripts/nuclei_distribution_analyzer.py`


### Initially done on Proton
#### Manuscript Methodology Figure visuals
Note: the following scripts consist of the making of each component of the manuscript methodology figure to be reproducible. With the semantic_seg step, we first randomly generated 5 patches (experiemnts) of the semantic segmentation of 4k by 4k patches of a WSI that matches with the cesc_polygon csv file names. We visually selected representable ones including a variety type of tissue classes and coverage (did not select patches with lots of white space background for exapmle). Each bash script has the option to do the random patch selection visualization or with a defined X_START, Y_START, and PATCH_SIZE manually. Once the ones to show in the manuscript were picked, they were correspondingly generated for the nuclei_classify_overlay, and the nuclei binary segmentation. 

- semantic_seg: /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/bash_scripts/Manuscript_Visualizations/semantic_seg_visual_progressbar.sh
- semantic_seg + cesc_polygon nuclei binary seg nuclei_classify_overlay: /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/bash_scripts/Manuscript_Visualizations/nuclei_overlay_visual.sh
- nuclei binary segmentation mask: based on /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/polygon_to_masks.py

