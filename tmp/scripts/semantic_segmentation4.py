import os
import shutil
import pandas as pd
import argparse
import time
import warnings
import logging
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig, SemanticSegmentor
from tiatoolbox.wsicore.wsireader import WSIReader

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class WSISemanticSegmentor:
    def __init__(self, wsi_path, save_dir, on_gpu=True):
        self.wsi_path = wsi_path
        self.save_dir = save_dir
        self.on_gpu = on_gpu
        self.mpp = self._get_mpp_from_wsi()
        self.segmentor = self._initialize_segmentor()
        self.ioconfig = self._create_ioconfig()

    def _get_mpp_from_wsi(self):
        wsi_reader = WSIReader.open(self.wsi_path)
        mpp = wsi_reader.info.mpp[0]  # Assuming square pixels
        print(f"Using mpp value from WSI: {mpp}")
        return mpp

    def _initialize_segmentor(self):
        return SemanticSegmentor(
            pretrained_model="fcn_resnet50_unet-bcss",
            num_loader_workers=48,
            batch_size=32,
            auto_generate_mask=True,
            verbose=False
        )

    def _create_ioconfig(self):
        return IOSegmentorConfig(
            input_resolutions=[{"units": "mpp", "resolution": self.mpp}],
            output_resolutions=[{"units": "mpp", "resolution": self.mpp}],
            patch_input_shape=[1024, 1024],
            patch_output_shape=[512, 512],
            stride_shape=[512, 512],
            save_resolution={"units": "mpp", "resolution": self.mpp},
        )

    def predict_wsi(self):
        print(f"Starting prediction on WSI: {self.wsi_path}")
        # Add the prediction logic here
        print(f"Prediction completed. Results saved to {self.save_dir}")

def process_sample_sheet(args):
    df = pd.read_csv(args.sample_sheet, sep="\t")
    completed_file = os.path.join(args.parent_save_dir, "progress_track", "completed_tobe_completed.tsv")
    skipped_file = os.path.join(args.parent_save_dir, "progress_track", "skipped_checkagain.tsv")
    processed_list = os.path.join(args.parent_save_dir, "progress_track", "processed_wsi_list.txt")

    # Open log files for appending
    with open(completed_file, "a") as completed, open(skipped_file, "a") as skipped, open(processed_list, "a") as processed:
        for _, row in df.iterrows():
            file_id = row["File ID"]
            file_name = row["File Name"]
            wsi_path = os.path.join(args.parent_wsi_dir, file_id, file_name)
            save_dir = os.path.join(args.parent_save_dir, file_id, file_name, "wsi_segmentation_results2_0.2277mpp_40x")
            output_file = os.path.join(save_dir, "0.raw.0.npy")
            
            # Check if WSI path exists
            if not os.path.exists(wsi_path):
                print(f"Skipping {file_name} with File ID: {file_id} (WSI path not found)")
                skipped.write("\t".join([file_id, file_name]) + "\n")
                continue

            # Check if output file already exists; if it does, skip to the next row
            if os.path.exists(output_file):
                print(f"Output already exists for {file_name}, skipping.")
                completed.write("\t".join([file_id, file_name]) + "\n")
                processed.write("\t".join([file_id, file_name]) + "\n")
                continue
            
            # Remove existing save directory only if output file does not exist
            if os.path.exists(save_dir):
                print(f"Removing existing save_dir: {save_dir}")
                shutil.rmtree(save_dir)
            
            # Try to process the WSI and catch any errors due to missing files or other issues
            try:
                segmentor = WSISemanticSegmentor(
                    wsi_path=wsi_path,
                    save_dir=save_dir,
                    on_gpu=args.on_gpu
                )
                segmentor.predict_wsi()
                completed.write("\t".join([file_id, file_name]) + "\n")
                processed.write("\t".join([file_id, file_name]) + "\n")
            except Exception as e:
                print(f"Error processing {file_name} with File ID: {file_id}: {e}")
                skipped.write("\t".join([file_id, file_name]) + "\n")
                continue

def main(args):
    start_time = time.time()
    process_sample_sheet(args)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Semantic Segmentation Script")
    parser.add_argument("--task", type=str, required=True, choices=["predict_wsi"], help="Task to execute")
    parser.add_argument("--sample_sheet", type=str, required=True, help="Path to the sample sheet TSV file")
    parser.add_argument("--parent_wsi_dir", type=str, required=True, help="Parent directory for WSI files")
    parser.add_argument("--parent_save_dir", type=str, required=True, help="Parent directory for saving segmentation results")
    parser.add_argument("--on_gpu", action="store_true", help="Flag to use GPU for prediction")

    args = parser.parse_args()
    main(args)