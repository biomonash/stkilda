import os
import json
from glob import glob
from pathlib import Path
from typing import Optional, List, Tuple
import yaml
import pandas as pd
import librosa
import soundfile as sf
import numpy as np


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

    def data_clip(self, original_files_path, selection_tables_file_paths,
                  output_dir, noise_padding:bool):

        penguin_out = Path(output_dir) / "Penguin_Clips"
        meta_out = Path(output_dir) / "Penguin_MetaData"

        penguin_out.mkdir(parents=True, exist_ok=True)
        meta_out.mkdir(parents=True, exist_ok=True)

        selection_files = glob(
            os.path.join(selection_tables_file_paths, "*.tsv")
        )

        padding_duration = self._clip_pad_max_ratio * self._window_size

        for sel_file in selection_files:

            base_name = Path(sel_file).name.replace(
                ".Table.1.selections.tsv", ""
            )

            audio_path = os.path.join(
                original_files_path, f"{base_name}.wav"
            )

            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                continue

            df_sel = pd.read_csv(sel_file, sep="\t")
            df_sel.columns = df_sel.columns.str.strip()

            for idx, row in df_sel.iterrows():

                begin_time = row["Begin Time (s)"]
                end_time = row["End Time (s)"]
                annotation = row["Annotation"]

                begin_time_short = int(begin_time)
                end_time_short = int(end_time)

                if noise_padding and annotation.lower() == "penguin":
                    audio_duration = sf.info(audio_path).duration
                    begin_time = max(0, begin_time - padding_duration)
                    end_time = min(audio_duration, end_time + padding_duration)

                duration = end_time - begin_time

                if duration <= 0:
                    print(f"Invalid duration in {sel_file}, row {idx}")
                    continue

                
                segment, sr = librosa.load(
                    audio_path,
                    sr=self.target_sr,
                    offset=begin_time,
                    duration=duration
                )

                if len(segment) == 0:
                    print(f"Empty audio segment: row {idx}")
                    continue

                general_file_name = f"{base_name}_{begin_time_short}_{end_time_short}"
                
                if annotation.lower() == "penguin" :
                    file_name = f"{general_file_name}_call"
                    audio_output_path = penguin_out / f"{file_name}.wav"
                    json_output_path = meta_out / f"{file_name}.json"
                else:
                    file_name = f"{general_file_name}_noise"
                    audio_output_path = penguin_out / f"{file_name}.wav"
                    json_output_path = meta_out / f"{file_name}.json"

                
                if len(segment) > 0:
                    sf.write(audio_output_path, segment, sr)

                    self.penguin_json(
                        begin_time,
                        end_time,
                        row.get("Annotation"),
                        row.get("call_type"),
                        row.get("condition"),
                        row.get("distance"),
                        row.get("counts"),
                        json_output_path
                    )

                    print(f"Saved: {file_name}")

        print("All clipping completed!")

    def penguin_json(self, begin_time, end_time, annotation, call_type, condition,
                     distance, count, json_path):
        data = {
            "begin_time": self.clean_values_json(begin_time),
            "end_time": self.clean_values_json(end_time),
            "annotation": self.clean_values_json(annotation),
            "call_type": self.clean_values_json(call_type),
            "condition": self.clean_values_json(condition),
            "distance": self.clean_values_json(distance),
            "count": self.clean_values_json(count)
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def clean_values_json(self, value):
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value


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
