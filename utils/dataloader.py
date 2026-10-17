import os
import json
from glob import glob
from pathlib import Path
from utils.config_finder import find_config_path
from typing import Optional, List, Tuple, Set
import yaml
import pandas as pd
import librosa
import soundfile as sf
import numpy as np


class DataLoader:

    acceptable_selection_file_extensions : list[str] = ["tsv", "txt"]
    acceptable_audio_file_extensions : list[str] = ["wav", "mp3", "flac"]

    required_columns : Set[str] = {
            "Annotation",
            "Begin Time (s)",
            "End Time (s)",
            "call_type",
            "condition",
            "distance",
            "count"
        }

    def __init__(
        self,
        window_size: float = 3.0,
        clip_pad_max_ratio: float = 0.9,
        target_sr: int = 24000
    ):
        """
        Constructor for DataLoader class. Used to clip annotated calls using selection tables and store in output directory.

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

        @Pre-conditions:
            - config.yaml file exists at {config_path} and contain all necessary keys

        @Returns
            Class object with state taken from values in config.yaml file
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

    def data_clip(self, original_files_path: Path, selection_tables_file_paths: Path,
                  output_dir: Path, noise_padding:bool):

        """
        Method to extract labelled selections from their corresponding audio files and, save the extracted audio clips and their corresponding
        metadata from their selection tables to the folders 'Audio_Clips' and 'Audio_MetaData'. Silence padding done if a clip could not be
        padded fully padded with noise by the padding duration amount.


        @Params
            original_files_path (Path):  Directory containing source audio files.
            selection_tables_file_paths (Path): Directory containing selection tables.
            output_dir (Path): file path of folder where the 'Audio_Clips' and 'Audio_MetaData' folders will be created
            noise_padding (bool): flag as to whether to apply noise padding to the extracted clips.

        @Pre-conditions:
            - selection tables must have the format '{Audio_File_Name}.Table.1.selections.{extension}'
            - 2 audio files cannot have the same file name (even with different extensions)
            - acceptable selection tables file types: ['tsv', 'txt']
            - acceptable audio file types: ['.wav', 'mp3', '.flac']
            - separator for selection table file: '\\t'
            - selection tables must have the following columns:
                - Annotation
                - Begin Time (s)
                - End Time (s)"
                - call_type
                - condition
                - distance
                - count

        @Post-conditions:
            - All Audio clips extracted stored at {output_dir}/Audio_Clips
            - All metadata for extracted Audio clips stored at {output_dir}/Audio_MetaData
            - Resulting audio clips will all be in .wav format

        @Returns:
            None
        """

        # Convert input paths to Path objects.
        original_files_path = Path(original_files_path)
        selection_tables_file_paths = Path(selection_tables_file_paths)
        output_dir = Path(output_dir)

        # Validate input directories before performing any processing.
        self.check_dir_exists([original_files_path, selection_tables_file_paths])

        # Create output directories. Stop if they cannot be created.
        penguin_out = output_dir / "Audio_Clips"
        meta_out = output_dir / "Audio_MetaData"

        try:
            penguin_out.mkdir(parents=True, exist_ok=True)
            meta_out.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Could not create output directories in {output_dir}: {exc}"
            ) from exc

        # Read all selection tables of acceptable file path
        selection_files = [
            file_path
            for ext in self.acceptable_selection_file_extensions
            for file_path in glob(os.path.join(selection_tables_file_paths, f"*.{ext}"))
        ]

        # Calculate the padding duration
        padding_duration = self._clip_pad_max_ratio * self._window_size

        # Loop through all selection files
        for sel_file in selection_files:

            # Extract the base name of the audio file from its selection table name
            base_name = Path(sel_file).name
            for ext in self.acceptable_selection_file_extensions:
                base_name = base_name.removesuffix(f".Table.1.selections.{ext}")

            # Obtain the audio path for that selection table
            audio_path = next(
                (
                    os.path.join(original_files_path, f"{base_name}.{ext}")
                    for ext in self.acceptable_audio_file_extensions
                    if os.path.exists(os.path.join(original_files_path, f"{base_name}.{ext}"))
                ),
                None
            )

            # If audio path does not exist, output an error and skip file
            if audio_path is None:
                print(f"Audio file not found for: {base_name}")
                continue

            # Read and validate the selection table.
            try:
                df_sel = pd.read_csv(sel_file, sep="\t")
                df_sel.columns = df_sel.columns.str.strip()

            except (OSError, UnicodeError, pd.errors.ParserError, ValueError) as exc:
                print(f"Skipping unreadable selection table {sel_file}: {exc}")
                continue

            missing_cols = self.check_for_missing_cols(required_cols=self.required_columns, present_cols=set(df_sel.columns))

            if missing_cols:
                print(
                    f"Skipping {sel_file} : missing required columns: "
                    f"{', '.join(sorted(missing_cols))}"
                )

                continue

            # Obtain the total duration for the audio
            audio_duration = librosa.get_duration(path=audio_path)

            # Load the audio samples and sampling rate for the audio clip
            # Also, re-samples the sampling rat of audio clip to the value in config.yaml
            audio_data, sr = librosa.load(
                audio_path,
                sr=self.target_sr,
                duration=audio_duration
            )

            # Loop through all selections in the selection table
            for idx, row in df_sel.iterrows():

                # Obtain the begin time for the selection
                begin_time = new_begin_time = row["Begin Time (s)"]
                # Obtain the end time for the selection
                end_time = new_end_time = row["End Time (s)"]
                # Obtain the Annotation for the selection
                annotation = row["Annotation"]

                # If noise padding must be appplied and the clip is a penguin sound, update the begin time and end time of the annotation
                if noise_padding and annotation.lower() == "penguin":
                    new_begin_time = max(0, begin_time - padding_duration)
                    new_end_time = min(audio_duration, end_time + padding_duration)

                # Calculate the duration of the audio clip
                duration = new_end_time - new_begin_time

                # If duration is less than 0 for an audio clip
                if duration <= 0:
                    print(f"Invalid duration in {sel_file}, row {idx} for file {base_name}")
                    continue

                # Extract the audio clip from the audio data
                segment= audio_data[int(new_begin_time * self.target_sr): int(new_end_time * self.target_sr)]

                # Check if segment loaded is valid
                if len(segment) <= 0:
                    print(f"Empty audio segment: row {idx} for file {base_name}")
                    continue

                # If noise padding should be applied, the audio clip is a penguin and the noise padding could not
                # be completely applied, fill the rest of padding duration with silence padding
                if noise_padding and annotation.lower() == "penguin":
                    # Fill in missing pad time at start with silence
                    if (begin_time - padding_duration) < 0:
                        pad_samples = int((padding_duration - begin_time) * self.target_sr)
                        segment = np.pad(segment, (pad_samples, 0), mode="constant", constant_values=0.0)

                    # Fill in missing pad time at end with silence
                    if (end_time + padding_duration) > audio_duration:
                        pad_samples = int((end_time + padding_duration - audio_duration) * self.target_sr)
                        segment = np.pad(segment, (0, pad_samples), mode="constant", constant_values=0.0)

                # Obtain file name for audio clip
                general_file_name = f"{base_name}_{int(begin_time)}_{int(end_time)}"

                
                if annotation.lower() == "penguin" :
                    file_name = f"{general_file_name}_call"
                    audio_output_path = penguin_out / f"{file_name}.wav"
                    json_output_path = meta_out / f"{file_name}.json"
                else:
                    file_name = f"{general_file_name}_noise"
                    audio_output_path = penguin_out / f"{file_name}.wav"
                    json_output_path = meta_out / f"{file_name}.json"

                # Save the audio clip and its metadata
                if len(segment) > 0:
                    sf.write(audio_output_path, segment, int(self.target_sr))

                    self.penguin_json(
                        begin_time,
                        end_time,
                        row.get("Annotation"),
                        row.get("call_type"),
                        row.get("condition"),
                        row.get("distance"),
                        row.get("count"),
                        json_output_path
                    )

                    print(f"---Saved: {file_name} ----")

        print("All clipping completed!")

    def penguin_json(self, begin_time, end_time, annotation, call_type, condition,
                     distance, count, json_path):
        """
        Save the metadata associated with an extracted audio clip to a JSON file.

        The method collects the selection's timing information and annotation
        metadata into a dictionary. Each value is passed through
        clean_values_json() to ensure that missing values and NumPy data types
        are converted into JSON-compatible values before being written to disk.

        @Params:
            begin_time (float): Start time of the original audio selection in seconds.
            end_time (float): End time of the original audio selection in seconds.
            annotation (str): Annotation label associated with the selection,
                such as 'penguin' or 'noise'.
            call_type (str): Type of call associated with the selection, if available.
            condition (str): Condition associated with the selection, if available.
            distance (float): Distance associated with the selection, if available.
            count (int): Number of calls or count value associated with the selection,
                if available.
            json_path (Path or str): File path where the JSON metadata will be saved.

        @Pre-conditions:
            - The supplied metadata values correspond to an extracted audio clip.
            - The parent directory of json_path exists.
            - The output path is writable.

        @Post-conditions:
            - A JSON file containing the selection metadata is created at json_path.
            - The JSON file is formatted with an indentation of four spaces.
            - Missing values are represented as JSON null.
            - NumPy scalar values are converted to their corresponding Python types.

        @Returns:
            None
        """

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
        """
        Convert missing values and NumPy scalar values into JSON-compatible values.

        This method prepares metadata values for JSON serialization by converting
        missing values, such as pandas NaN values, to None, and converting NumPy
        scalar types to their equivalent native Python types. Other values are
        returned unchanged.

        @Params:
            value (Any): Value to be checked and converted before JSON serialization.

        @Pre-conditions:
            - The input value can be checked using pandas.isna().
            - If the value is a NumPy scalar, it supports conversion using item().

        @Post-conditions:
            - Missing values are returned as None and serialized as JSON null.
            - NumPy scalar values are converted to native Python values.
            - Other values are returned unchanged.

        @Returns:
            Any: A JSON-compatible value, or the original value if no conversion
                is required.
        """

        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value


    ### HELPER CLASSES TO REDUCE BLOATING ###

    def check_dir_exists(self, dir_paths: List[Path]) -> None:
        for directory in dir_paths:
            if not directory.exists():
                raise FileNotFoundError(
                    f"Input directory does not exist: {directory}"
                )
            if not directory.is_dir():
                raise NotADirectoryError(
                    f"Input path is not a directory: {directory}"
                )

    def check_for_missing_cols(self, required_cols: Set[str], present_cols: Set[str]) -> Set[str]|None:
            
            missing_columns = required_cols - present_cols

            if missing_columns:
                return missing_columns

            return None