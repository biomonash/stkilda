import os
import json
from glob import glob
from pathlib import Path

import pandas as pd
import librosa
import soundfile as sf
import numpy as np


class DataLoader:

    def __init__(self, target_sr=32000):
        self.target_sr = target_sr

    def data_clip(self, original_files_path, selection_tables_file_paths,
                  output_dir):

        penguin_out = Path(output_dir) / "Penguin_Clips"
        meta_out = Path(output_dir) / "Penguin_MetaData"

        penguin_out.mkdir(parents=True, exist_ok=True)
        meta_out.mkdir(parents=True, exist_ok=True)

        selection_files = glob(
            os.path.join(selection_tables_file_paths, "*.tsv")
        )

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
                
                if annotation == "penguin" :
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
