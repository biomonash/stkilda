import os
from glob import glob
from typing import Tuple, List
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

"""Class Module DataLoader Initiation"""
class DataLoader:
    """ Values requied for the audio file clipping"""
    def __init__(
        self,
        window_size: float = 3.0,
        clip_pad_max_ratio: float = 0.9,
        target_sr: int = 24000
    ):
        """Replacing variable names"""
        self._window_size = window_size
        self._clip_pad_max_ratio = clip_pad_max_ratio
        self.target_sr = target_sr

    """Processing the clipped audio clips"""
    def data_clip(
        self,
        original_files_path: str,
        selection_tables_file_paths: str,
        clipped_audio: str
    ) -> None:
        """The penguine and noise sounds are divided into 2 folders"""
        penguin_out = Path(clipped_audio) / "Penguin"
        noise_out = Path(clipped_audio) / "noise"

        """Create folders if it does not exist"""
        penguin_out.mkdir(parents=True, exist_ok=True)
        noise_out.mkdir(parents=True, exist_ok=True)

        """ Locate the selection table file"""
        selection_files = glob(os.path.join(selection_tables_file_paths, "*.txt"))

        """Loop through the selection table"""
        for sel_file in selection_files:
            base_name = os.path.basename(sel_file).split('.')[0].replace("-selection", "").replace("_selection_table", "")
            audio_path = os.path.join(original_files_path, f"{base_name}.wav")

            if not os.path.exists(audio_path):
                print(f"Audio file missing at {audio_path}")
                continue
            """Resampling to the sr"""
            signal, sr = librosa.load(audio_path, sr=self.target_sr)
            df_sel = pd.read_csv(sel_file, sep="\t")

            """Idenitfying start and end timestamp columns from selection table"""
            start_col = [c for c in df_sel.columns if "Begin Time" in c or "start" in c.lower()][0]
            end_col = [c for c in df_sel.columns if "End Time" in c or "end" in c.lower()][0]

            """ Loop through each annotation row in the selection table"""
            for idx, row in df_sel.iterrows():
                begin_time = row[start_col]
                end_time = row[end_col]

                start_sample = max(0, int(begin_time * sr))
                end_sample = min(len(signal), int(end_time * sr))
                segment = signal[start_sample:end_sample]

                label_type = row.get("Annotation", "Penguin")
                if "penguin" in str(label_type).lower():
                    out_path = penguin_out / f"{base_name}_clip_{idx}.wav"
                else:
                    out_path = noise_out / f"{base_name}_noise_{idx}.wav"

                if len(segment) > 0:
                    sf.write(out_path, segment, sr)

    """ Extracting the audio clips with 1 and 0 labelling"""
    def data_load(
        self,
        original_files_folder_path: str,
        selection_tables_folder_paths: str
    ) -> List[Tuple[object, int]]:
        
        """ Extracting the labelled audio clips"""
        labelled_data = []
        """ Calculating the padding duration"""
        pad_duration = self._clip_pad_max_ratio * self._window_size

        """ Locate the selection table file"""
        selection_files = glob(os.path.join(selection_tables_folder_paths, "*.txt"))

        """ Looping through all audio files to match them with the selection table rows"""
        for sel_file in selection_files:
            base_name = os.path.basename(sel_file).split('.')[0].replace("-selection", "").replace("_selection_table", "")
            audio_path = os.path.join(original_files_folder_path, f"{base_name}.wav")

            if not os.path.exists(audio_path):
                continue

            """ Resample the audio files to sr"""
            signal, sr = librosa.load(audio_path, sr=self.target_sr)
            """ Calculate the length of the recording"""
            total_duration = librosa.get_duration(y=signal, sr=sr)
            df_sel = pd.read_csv(sel_file, sep="\t")

            """ Comparing the start and end timestamps in the selection table columns"""
            start_col = [c for c in df_sel.columns if "Begin Time" in c or "start" in c.lower()][0]
            end_col = [c for c in df_sel.columns if "End Time" in c or "end" in c.lower()][0]

            """ Loop through all entries in the selection table"""
            for _, row in df_sel.iterrows():
                begin_time = max(0.0, row[start_col] - pad_duration)
                end_time = min(total_duration, row[end_col] + pad_duration)

                """ Converting padded timestamps"""
                start_sample = int(begin_time * sr)
                end_sample = int(end_time * sr)
                clipped_audio = signal[start_sample:end_sample]

                """ Annoted audio files to 0 and 1 based on the selection table row annotations"""
                annotation = str(row.get("Annotation", "Penguin")).lower()
                label = 1 if "penguin" in annotation else 0
                
                """ Store the labelled audio clips as a tuple list"""
                labelled_data.append((clipped_audio, label))

        """ Return the tuple list of labelled audio clips"""
        return labelled_data
