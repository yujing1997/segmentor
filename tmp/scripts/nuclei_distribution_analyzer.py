"""
Histogram correction for the raw distribution of the classified_nuclei data from ./scripts/bash_scripts/nuclei_classify_extract.sh
The results reads in the JSON format 

The original structure of the JSON file is as follows:

            classified_nuclei.append({
                'majority_class': int(majority_class),
                'AreaInPixels': area_in_pixels,
                'perimeter': perimeter,
                'AreaInSquareMicrons': area_in_square_microns,
                'RadiusInMicrons': equivalent_radius_microns,
                'mpp': mpp
            })

"""


import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Tuple, Any, Optional
import codecs  # For delimiter decoding
from tqdm import tqdm  # For progress bar
from logging.handlers import RotatingFileHandler  # For log rotation (optional)

# Configure global logger as the root logger
logger = logging.getLogger()


class HistogramCorrector:
    """
    Handles histogram correction and computes statistical measures.
    """

    def __init__(self, solution_space_size: int = 200, correction_factor: float = 0.05):
        """
        Initializes the HistogramCorrector with correction parameters.

        Parameters
        ----------
        solution_space_size : int
            Number of bins for histogram correction.
        correction_factor : float
            Scaling factor applied during histogram correction.
        """
        self.solution_space_size = solution_space_size
        self.correction_factor = correction_factor

    def correct_histogram(self, distribution: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Corrects the histogram for a given distribution.

        Parameters
        ----------
        distribution : List[float]
            Raw distribution data (e.g., 'RadiusInMicrons').

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Corrected histogram counts and bin edges.
        """
        # Convert to NumPy array for efficient processing
        data = np.array(distribution)

        if data.size == 0:
            logger.warning("Empty distribution provided for correction.")
            return np.array([]), np.array([])

        # Define histogram bins based on data range
        bins = np.linspace(0, np.max(data) * 1.5, self.solution_space_size)

        # Compute histogram
        counts, bin_edges = np.histogram(data, bins=bins, density=False)

        # Apply correction (placeholder: scaling counts)
        corrected_counts = counts * self.correction_factor

        logger.debug(f"Histogram corrected with factor {self.correction_factor}.")

        return corrected_counts, bin_edges

    def compute_statistics(self, corrected_counts: np.ndarray, bin_edges: np.ndarray) -> Tuple[float, float]:
        """
        Calculates the mean and standard deviation from the corrected histogram.

        Parameters
        ----------
        corrected_counts : np.ndarray
            Corrected histogram counts.
        bin_edges : np.ndarray
            Edges of the histogram bins.

        Returns
        -------
        Tuple[float, float]
            Mean and standard deviation of the distribution.
        """
        if corrected_counts.size == 0 or bin_edges.size == 0:
            logger.warning("Empty histogram data provided for statistics computation.")
            return float('nan'), float('nan')

        # Calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Normalize counts to form a probability density function (PDF)
        pdf = corrected_counts / np.sum(corrected_counts)

        # Compute mean
        mean = np.sum(bin_centers * pdf)

        # Compute standard deviation
        variance = np.sum(((bin_centers - mean) ** 2) * pdf)
        std_dev = np.sqrt(variance)

        logger.debug(f"Computed statistics - Mean: {mean}, Std Dev: {std_dev}")

        return mean, std_dev


class NucleiDistributionProcessor:
    """
    Processes nuclei distribution data based on a sample sheet and computes statistics.
    """

    def __init__(
        self,
        sample_sheet_path: str,
        parent_save_dir: str,
        output_dir: str,
        delimiter: str = ',',
        num_workers: int = 1,
        apply_histogram_correction: bool = True,
        histogram_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initializes the processor with paths and settings.

        Parameters
        ----------
        sample_sheet_path : str
            Path to the sample sheet CSV file.
        parent_save_dir : str
            Parent directory where nuclei classify result folders are located.
        output_dir : str
            Directory to save the output CSV files.
        delimiter : str
            Delimiter used in the sample sheet CSV file.
        num_workers : int
            Number of parallel workers to use.
        apply_histogram_correction : bool
            Flag to apply histogram correction. If False, computes mean and std directly.
        histogram_params : Optional[Dict[str, Any]]
            Parameters for histogram correction.
        """
        self.sample_sheet_path = sample_sheet_path
        self.parent_save_dir = parent_save_dir
        self.output_dir = output_dir
        self.delimiter = delimiter
        self.num_workers = num_workers
        self.apply_histogram_correction = apply_histogram_correction
        self.histogram_params = histogram_params if histogram_params else {}

        # Initialize HistogramCorrector if needed
        self.corrector = HistogramCorrector(**self.histogram_params) if self.apply_histogram_correction else None

        # Initialize result dictionaries for classes 0 to 4
        self.results = {i: [] for i in range(5)}

        # List to keep track of skipped samples
        self.skipped_samples = []

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def read_sample_sheet(self) -> pd.DataFrame:
        """
        Reads the sample sheet into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the sample sheet data.
        """
        try:
            df = pd.read_csv(self.sample_sheet_path, sep=self.delimiter)
            logger.info(f"Sample sheet loaded with {len(df)} rows.")
            logger.debug(f"Sample sheet columns: {list(df.columns)}")
            logger.debug(f"First 5 rows of the sample sheet:\n{df.head()}")
            return df
        except FileNotFoundError:
            logger.error(f"Sample sheet file '{self.sample_sheet_path}' not found.")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Sample sheet file '{self.sample_sheet_path}' is empty.")
            raise
        except Exception as e:
            logger.error(f"Error reading sample sheet: {e}")
            raise

    def process_sample(self, row: pd.Series) -> Dict[int, Dict[str, Any]]:
        """
        Processes a single sample row: applies histogram correction and computes statistics for each class.

        Parameters
        ----------
        row : pd.Series
            A row from the sample sheet DataFrame.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            Dictionary mapping class numbers to their respective results.
        """
        # Debug: Print the entire row
        logger.debug(f"Processing row: {row.to_dict()}")

        # Check if 'File Name' exists in the row
        if 'File Name' not in row:
            logger.error("Column 'File Name' not found in the row.")
            raise KeyError("'File Name' column is missing in the sample sheet.")

        file_name = row['File Name']
        logger.debug(f"Extracted File Name: '{file_name}'")

        result_folder = os.path.join(self.parent_save_dir, file_name)
        logger.debug(f"Constructed result folder path: '{result_folder}'")

        if not os.path.isdir(result_folder):
            logger.warning(f"Result folder '{result_folder}' does not exist. Skipping sample '{file_name}'.")
            self.skipped_samples.append(file_name)
            return {}

        sample_results = {}

        for class_num in range(5):
            json_file = os.path.join(result_folder, f"class_{class_num}_nuclei.json")
            logger.debug(f"Checking for JSON file: '{json_file}'")

            if not os.path.isfile(json_file):
                logger.warning(f"JSON file '{json_file}' does not exist. Skipping class {class_num} for sample '{file_name}'.")
                continue

            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                logger.debug(f"Loaded JSON file '{json_file}' successfully.")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON format in file '{json_file}'. Skipping class {class_num} for sample '{file_name}'.")
                continue

            # Extract 'RadiusInMicrons' distribution
            try:
                radius_distribution = data['RadiusInMicrons']
                if not isinstance(radius_distribution, list):
                    raise ValueError("'RadiusInMicrons' should be a list.")
                # Log only the count of distribution values
                logger.info(f"'RadiusInMicrons' distribution: [List of {len(radius_distribution)} values]")
            except KeyError:
                logger.error(f"'RadiusInMicrons' key missing in JSON file '{json_file}'. Skipping class {class_num} for sample '{file_name}'.")
                continue
            except ValueError as ve:
                logger.error(f"{ve} in file '{json_file}'. Skipping class {class_num} for sample '{file_name}'.")
                continue

            if self.apply_histogram_correction and self.corrector:
                # Apply histogram correction
                corrected_counts, bin_edges = self.corrector.correct_histogram(radius_distribution)

                # Compute statistics from corrected histogram
                mean, std_dev = self.corrector.compute_statistics(corrected_counts, bin_edges)
            else:
                # Compute mean and std dev directly from the distribution
                if len(radius_distribution) == 0:
                    logger.warning(f"No 'RadiusInMicrons' data in file '{json_file}'. Skipping class {class_num} for sample '{file_name}'.")
                    continue

                # Convert list to NumPy array for computation
                radius_array = np.array(radius_distribution)

                # Compute mean and standard deviation
                mean = np.mean(radius_array)
                std_dev = np.std(radius_array)

            # Store results
            sample_results[class_num] = {
                'File Name': file_name,
                'Mean_RadiusInMicrons': mean,
                'Std_RadiusInMicrons': std_dev
            }

            logger.info(f"Processed '{json_file}': Mean={mean:.2f}, Std Dev={std_dev:.2f}")

        return sample_results

    def process_all_samples(self) -> None:
        """
        Processes all samples in the sample sheet, optionally in parallel.
        """
        df = self.read_sample_sheet()

        total_samples = len(df)
        if self.num_workers > 1:
            logger.info(f"Processing samples with {self.num_workers} parallel workers.")
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all sample processing tasks
                futures = {executor.submit(self.process_sample, row): idx for idx, row in df.iterrows()}

                # Initialize tqdm progress bar
                for future in tqdm(as_completed(futures), total=total_samples, desc="Processing Samples"):
                    try:
                        sample_result = future.result()
                        for class_num, result in sample_result.items():
                            self.results[class_num].append(result)
                    except KeyError as ke:
                        logger.error(f"KeyError: {ke}.")
                    except Exception as e:
                        logger.error(f"Error processing a sample: {e}")
        else:
            logger.info("Processing samples sequentially.")
            # Initialize tqdm progress bar for sequential processing
            for idx, row in tqdm(df.iterrows(), total=total_samples, desc="Processing Samples"):
                try:
                    sample_result = self.process_sample(row)
                    for class_num, result in sample_result.items():
                        self.results[class_num].append(result)
                except KeyError as ke:
                    logger.error(f"KeyError in sample at index {idx}: {ke}. Skipping.")
                except Exception as e:
                    logger.error(f"Error processing sample at index {idx}: {e}. Skipping.")

    def write_results(self) -> None:
        """
        Writes the results to separate CSV files for each class.
        """
        for class_num, records in self.results.items():
            if not records:
                logger.warning(f"No records found for class {class_num}. Skipping CSV generation.")
                continue

            df = pd.DataFrame(records)
            csv_file = os.path.join(self.output_dir, f"class_{class_num}_results.csv")
            try:
                df.to_csv(csv_file, index=False)
                logger.info(f"Wrote results to '{csv_file}'.")
            except IOError as e:
                logger.error(f"Failed to write to '{csv_file}': {e}")

    def write_summary(self) -> None:
        """
        Writes a summary of skipped samples to a CSV file.
        """
        if not self.skipped_samples:
            logger.info("No samples were skipped. Summary report not generated.")
            return

        summary_file = os.path.join(self.output_dir, "skipped_samples_summary.csv")
        df = pd.DataFrame(self.skipped_samples, columns=['File Name'])
        try:
            df.to_csv(summary_file, index=False)
            logger.info(f"Wrote skipped samples summary to '{summary_file}'.")
        except IOError as e:
            logger.error(f"Failed to write summary report: {e}")

    def run(self) -> None:
        """
        Executes the processing pipeline: processing samples and writing results.
        """
        self.process_all_samples()
        self.write_results()
        self.write_summary()


def main(
    sample_sheet: str,
    parent_save_dir: str,
    output_dir: str,
    delimiter: str,
    num_workers: int,
    histogram_correction: bool,
    solution_space_size: int,
    correction_factor: float
) -> None:
    """
    Main function to initiate the nuclei distribution processing.

    Parameters
    ----------
    sample_sheet : str
        Path to the sample sheet CSV file.
    parent_save_dir : str
        Parent directory where nuclei classify result folders are located.
    output_dir : str
        Directory to save the output CSV files and log file.
    delimiter : str
        Delimiter used in the sample sheet CSV file.
    num_workers : int
        Number of parallel workers to use.
    histogram_correction : bool
        Whether to apply histogram correction.
    solution_space_size : int
        Number of bins for histogram correction.
    correction_factor : float
        Scaling factor applied during histogram correction.
    """
    # Decode escape sequences in delimiter
    try:
        decoded_delimiter = codecs.decode(delimiter, 'unicode_escape')
    except Exception as e:
        print(f"Error decoding delimiter '{delimiter}': {e}. Using original delimiter.")
        decoded_delimiter = delimiter

    # Ensure output directory exists before setting up log file
    os.makedirs(output_dir, exist_ok=True)
    # Define log file path
    log_file_path = os.path.join(output_dir, 'nuclei_distribution_analyzer.log')

    # Configure the root logger
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create console handler with INFO level
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)  # Console handler set to INFO

    # Create file handler with DEBUG level and log rotation (optional)
    f_handler = RotatingFileHandler(log_file_path, maxBytes=10**6, backupCount=5)
    f_handler.setLevel(logging.DEBUG)  # File handler set to DEBUG to capture all logs

    # Create formatters and add them to handlers
    c_format = logging.Formatter('%(levelname)s: %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    # Log configuration completion
    logger.info("Logging is set up. Logs will be saved to the console and to the log file.")

    # Prepare histogram parameters if correction is applied
    histogram_params = {}
    if histogram_correction:
        histogram_params['solution_space_size'] = solution_space_size
        histogram_params['correction_factor'] = correction_factor
        logger.info(f"Histogram correction parameters: solution_space_size={solution_space_size}, correction_factor={correction_factor}")
    else:
        logger.info("Histogram correction is disabled. Computing statistics directly from raw data.")

    # Initialize the processor
    processor = NucleiDistributionProcessor(
        sample_sheet_path=sample_sheet,
        parent_save_dir=parent_save_dir,
        output_dir=output_dir,
        delimiter=decoded_delimiter,
        num_workers=num_workers,
        apply_histogram_correction=histogram_correction,
        histogram_params=histogram_params
    )
    processor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nuclei Distribution Analyzer: Computes mean and standard deviation of nuclei sizes, with optional histogram correction."
    )
    parser.add_argument(
        '--sample_sheet',
        type=str,
        required=True,
        help='Path to the sample sheet CSV file.'
    )
    parser.add_argument(
        '--parent_save_dir',
        type=str,
        required=True,
        help='Parent directory where nuclei classify result folders are located.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save the output CSV files and log file.'
    )
    parser.add_argument(
        '--delimiter',
        type=str,
        default=',',
        help='Delimiter used in the sample sheet CSV file. Default is comma (,). Use "\\t" for tab.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of parallel workers to use. Default is 1 (sequential processing).'
    )
    parser.add_argument(
        '--histogram_correction',
        action='store_true',
        help='Apply histogram correction. If not set, computes mean and std directly without correction.'
    )
    parser.add_argument(
        '--solution_space_size',
        type=int,
        default=200,
        help='Number of bins for histogram correction. Default is 200.'
    )
    parser.add_argument(
        '--correction_factor',
        type=float,
        default=0.05,
        help='Scaling factor applied during histogram correction. Default is 0.05.'
    )

    args = parser.parse_args()

    main(
        sample_sheet=args.sample_sheet,
        parent_save_dir=args.parent_save_dir,
        output_dir=args.output_dir,
        delimiter=args.delimiter,
        num_workers=args.num_workers,
        histogram_correction=args.histogram_correction,
        solution_space_size=args.solution_space_size,
        correction_factor=args.correction_factor
    )


# USE CASE EXAMPLE: 
# module load StdEnv/2023
# module load python/3.10.13
# source ~/envs/semanticseg310/bin/activate

# OLD SCRIPT
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_distribution_analyzer.py \
#     --sample_sheet /Data/Yujing/Segment/tmp/tcga_cesc_manifest/gdc_sample_sheet.2024-11-11.tsv \
#     --parent_save_dir /Data/Yujing/Segment/tmp/nuclei_classify_results \
#     --output_dir /Data/Yujing/Segment/tmp/corrected_histogram_nuclei_results \
#     --delimiter "\t" \
#     --num_workers 8

# CURRENT SCRIPT (For the semantic_seg matching with cesc_polygon ScientificData batch)

# With Histogram Correction:
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_distribution_analyzer.py \
#     --sample_sheet /Data/Yujing/Segment/tmp/tcga_cesc_manifest/gdc_sample_sheet.2024-11-11.tsv \
#     --parent_save_dir /Data/Yujing/Segment/tmp/nuclei_classify_results \
#     --output_dir /Data/Yujing/Segment/tmp/corrected_histogram_nuclei_results \
#     --delimiter "\t" \
#     --num_workers 8
#     --histogram_correction \
#     --solution_space_size 300 \
#     --correction_factor 0.05

# Without Histogram Correction:
# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_distribution_analyzer.py \
#     --sample_sheet /Data/Yujing/Segment/tmp/tcga_cesc_manifest/gdc_sample_sheet.2024-11-11.tsv \
#     --parent_save_dir /Data/Yujing/Segment/tmp/nuclei_classify_results \
#     --output_dir /Data/Yujing/Segment/tmp/uncorrected_histogram_nuclei_results \
#     --delimiter "\t" \
#     --num_workers 8

# ================================================================================================================
# For the instance_seg hovernet (whose cesc_polygon from ScientificData batch is not available) batch 
# With Histogram Correction:

# python /home/yujing/dockhome/Multimodality/Segment/tmp/scripts/nuclei_distribution_analyzer.py \
#     --sample_sheet /home/yujing/dockhome/Multimodality/Segment/tmp/manifest/run_4_instanceseg_2024_11_27_DxOnly_Proton.tsv \
#     --parent_save_dir /Data/Yujing/Segment/tmp/Instance_Segmentation/nuclei_instance_classify_results \
#     --output_dir /Data/Yujing/Segment/tmp/Instance_Segmentation/corrected_histogram_nuclei_results \
#     --delimiter "\t" \
#     --num_workers 8
#     --histogram_correction \
#     --solution_space_size 300 \
#     --correction_factor 0.05
