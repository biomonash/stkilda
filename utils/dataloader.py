import os
from glob import glob
from typing import Tuple, List, Optional
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from utils.config_finder import find_config_path
import yaml

class DataLoader:
    def __init__(
        self,
        window_size: float = 3.0,
        clip_pad_max_ratio: float = 0.9,
        target_sr: int = 24000
    ):
        """
        Constructor for DataLoader class. Used to clip annotated calls using selection tables and store in output directory

        @Params
            window_size (float): Duration of window 
            clip_pad_max_ratio (float): Ratio relative to window size of buffer to be used before and after each Penguin call
            target_sr (int): The sampling rate at which all audio should be
        """
        self._window_size = window_size
        self._clip_pad_max_ratio = clip_pad_max_ratio
        self.target_sr = target_sr


    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> "DataLoader":
        """
        Use settings found in config.yaml file for data clipping and 
        preparation.

        Uses the default hardcoded path but can be changed by the optional parameter 'config_path'

        @Params:
            config_path (str): The file path for the config.yaml file (OPTIONAL)
        """

        if config_path is None:
            config_path = find_config_path(__file__)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            shared_cfg = config["shared"]
            audio_cfg = config["audio_config"]
            merged_cfg = {
                "window_size": shared_cfg["window_size"],
                "clip_pad_max_ratio": audio_cfg["clip_pad_max_ratio"],
                "target_sr": shared_cfg["sr"]
            }

        except KeyError as e:
            raise ValueError(
                f"config.yaml is missing expected key {e}. Check that 'shared.sr', 'shared.window_size', and 'audio_config' are present."
            ) from e

        return cls(**merged_cfg)

    
    def data_clip(
        self,
        original_files_path: str,
        selection_tables_file_paths: str,
        output_dir: str,
        noise_padding: bool
    ) -> None:
        """
        Extracts annotated sounds from audio files using their selection tables and classifies them into 'Penguin' folder or 'noise' folder 
        at the path in 'output_dir'. Annotations which were not labelled are put into a 'Unlabelled' folder at 'output_dir'.

        @Params:
            original_files_path (str): Path of folder containing the original audio recordings
            selection_tables_file_paths (str): Path of the folder containing the selection tables of the audio files
            output_dir (str): Path for the 'Penguin' folder and 'noise' folder to contain the clipped annotated sound 
            noise_padding (bool): Flag to choose whether to pad start and end of penguin call annotations with background noise from the original file. Amount of passing is clip_pad_max_ratio * window_size
        """

        # Create file paths to Penguin, noise and unlabelled folders
        penguin_out = Path(output_dir) / "Penguin"
        noise_out = Path(output_dir) / "noise"
        unlabelled_out = Path(output_dir) / "Unlabelled_Annotations"

        # Create the Penguin, noise and unlabelled folders at 'output_dir' if they do not already exist
        penguin_out.mkdir(parents=True, exist_ok=True)
        noise_out.mkdir(parents=True, exist_ok=True)
        unlabelled_out.mkdir(parents=True, exist_ok=True)

        # Obtain the file paths of all the selection tables
        selection_files = glob(os.path.join(selection_tables_file_paths, "*.txt"))

        # Duration by which to pad Penguin call with background noise at the start and the end
        pad_duration = self._clip_pad_max_ratio * self._window_size

        # Loop through each selection table
        for sel_file in selection_files:
            # Obtain the corresponding audio file name
            base_name = os.path.basename(sel_file).split('.')[0].replace("-selection", "").replace("_selection_table", "")
            # Obtain file path to audio file
            audio_path = os.path.join(original_files_path, f"{base_name}.wav")

            if not os.path.exists(audio_path):
                print(f"Audio file missing at {audio_path}")
                continue

            # Obtain the selections in the selection table
            df_sel = pd.read_csv(sel_file, sep="\t")

            # Iterate through every selection in the file
            for idx, row in df_sel.iterrows():
                # Obtain the begin time and end time of the selection
                begin_time = row["Begin Time (s)"]
                end_time = row["End Time (s)"]
                selection_id = row["Selection"]

                # Get label of annotation
                label_type = row.get("Annotation", "No Label")

                # If noise padding must be done, we alter the begin time and end time of clipping
                if noise_padding and "penguin" in str(label_type).lower():
                    audio_duration = sf.info(audio_path).duration
                    begin_time = max(0, begin_time - pad_duration)
                    end_time = min(audio_duration, end_time + pad_duration)

                # Load the annotated audio from the file
                segment, sr = librosa.load(
                    audio_path,
                    sr = self.target_sr,
                    offset= begin_time,
                    duration= end_time - begin_time
                )

                # Save the audio clip based on its label
                if str(label_type) == "No Label":
                    out_path = unlabelled_out / f"{base_name}_unlabelled_clip_{selection_id}.wav"
                elif "penguin" in str(label_type).lower():
                    out_path = penguin_out / f"{base_name}_{label_type}_{selection_id}.wav"
                else:
                    out_path = noise_out / f"{base_name}_{label_type}_{selection_id}.wav"

                if len(segment) > 0:
                    sf.write(out_path, segment, int(sr))

    def data_load(
        self,
        original_files_folder_path: str,
        selection_tables_folder_path: str,
        noise_padding: bool
    ) -> List[Tuple[object, int]]:
        """
        Method to load and return labelled, clipped audio files based on annotations in selections tables at 'selection_tables_folder_paths' with
        the corresponding audio files found at 'original_files_folder_path'.

        Penguin calls are labelled as 1 while non-penguin sounds are labelled as zero

        @Params:
            original_files_folder_path (str): Path of folder which contains original audio files
            selection_tables_folder_path (str): Path of folder which contains selection tables
            noise_padding (bool): Flag to choose whether to pad start and end of penguin call annotations with background noise from the original file. Amount of passing is clip_pad_max_ratio * window_size

        @Returns:
            - A list containing tuples of the form (samples of clipped audio, label for audio)
        """

        # List to contain the labelled data
        labelled_data = []
        # Duration by which to pad Penguin call with background noise at the start and the end
        pad_duration = self._clip_pad_max_ratio * self._window_size

        # File paths of the different selection files
        selection_files = glob(os.path.join(selection_tables_folder_path, "*.txt"))


        # Loop through all selection files
        for sel_file in selection_files:
            # Obtain file name of the corresponding audio file of selection file
            base_name = os.path.basename(sel_file).split('.')[0].replace("-selection", "").replace("_selection_table", "")
            # Get the file path of the corresponding file path
            audio_path = os.path.join(original_files_folder_path, f"{base_name}.wav")

            # Check if the audio file exists at file path
            if not os.path.exists(audio_path):
                print(f"No audio file found at {audio_path}")
                continue

            # Get the audio duration
            audio_duration = sf.info(audio_path).duration

            # Obtain the selections
            df_sel = pd.read_csv(sel_file, sep="\t")

            """ Loop through all entries in the selection table"""
            for idx, row in df_sel.iterrows():
                # Obtain the begin time and end time of the annotation
                begin_time = row["Begin Time (s)"]
                end_time = row["End Time (s)"]

                # Get label of annotation
                label_type = row.get("Annotation", "No Label")

                if label_type == "No Label":
                    print(f"Selection {idx} for {base_name}.wav is not labelled")
                    continue

                if noise_padding and "penguin" in str(label_type).lower():
                    begin_time = max(0, begin_time - pad_duration)
                    end_time = min(audio_duration, end_time + pad_duration)

                # Clip the annotation
                segment, sr = librosa.load(
                                    audio_path,
                                    sr = self.target_sr,
                                    offset= begin_time,
                                    duration= end_time - begin_time
                                )

                label = 1 if "penguin" in str(label_type).lower() else 0

                labelled_data.append((segment, label))

        return labelled_data
