# Reference: https://huggingface.co/docs/datasets/audio_load
# Reference: https://huggingface.co/docs/datasets/audio_dataset

import json
from datasets import load_dataset, DatasetDict

raw_datasets = DatasetDict()

raw_datasets = load_dataset(
    "audiofolder", 
    data_dir="./audio/slurp_real/train",
)
print(raw_datasets)
print(raw_datasets['train'][0])

if "audio" not in raw_datasets["train"].column_names:
    raise ValueError(
        f"--audio_column_name audio not found in dataset. "
    )

if "label" not in raw_datasets["train"].column_names:          # Error
    raise ValueError(
        f"--label_column_name label not found in dataset. "
    )