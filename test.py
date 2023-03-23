# Reference: https://huggingface.co/docs/datasets/audio_load

from datasets import load_dataset, DatasetDict

raw_datasets = DatasetDict()

raw_datasets['train'] = load_dataset(
    "audiofolder", 
    data_dir="./audio/slurp_real",
    split="train"
)

if "audio" not in raw_datasets["train"].column_names:
    raise ValueError(
        f"--audio_column_name audio not found in dataset. "
    )

if "label" not in raw_datasets["train"].column_names:          # Error
    raise ValueError(
        f"--label_column_name label not found in dataset. "
    )

# dataset = load_dataset("audiofolder", data_files=["./audio/slurp_real/audio-1501754435.flac", "./audio/slurp_real/audio-1501407267-headset.flac"])
# print(dataset['train'][0])