# Generate metadata.csv which will be used in `load_dataset`

import pandas as pd
import csv 
import json
import os

json_obj = pd.read_json('./slurp/train.jsonl', lines=True)

metadata_dict = dict()
for idx in range(len(json_obj)):
    recordings = json_obj['recordings'][idx]
    entities = json_obj['entities'][idx]
    for i in range(len(recordings)):
        if recordings[i]['file'] not in metadata_dict:
            metadata_dict[recordings[i]['file']] = "_"
        for j in range(len(entities)):
            metadata_dict[recordings[i]['file']] += f"{entities[j]['type']}_"

# Write to CSV
with open('./audio/slurp_real/train/metadata.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["file_name", "label"]
    writer.writerow(header)

    for key, value in metadata_dict.items():
        writer.writerow([f'data/{key}', value])
