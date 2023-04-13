#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
import sys
import torch
import warnings
from dataclasses import dataclass, field
from random import randint
from torch import nn
from typing import Any, Dict, List, Optional, Union

import datasets
import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset, ClassLabel

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

require_version("datasets>=1.14.0", "To fix: pip install -r examples/pytorch/audio-classification/requirements.txt")


def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the training audio paths and labels."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=None, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder`"
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`."
                "Only make use of `--freeze_feature_encoder`."
            )

class customTrainingArguments(TrainingArguments):
    def __init__(self,*args, **kwargs):
        super(customTrainingArguments, self).__init__(*args, **kwargs)

    @property
    # @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        Name the device the number you use.
        """
        return torch.device("cuda")

    @property
    # @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.
        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        # Make sure `self._n_gpu` is properly setup.
        # _ = self._setup_devices
        # I set to one manullay
        self._n_gpu = 1
        return self._n_gpu


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, customTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_audio_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the audio classification task.
    raw_datasets = DatasetDict()
    raw_datasets['train'] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.train_split_name,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    print(f'raw_datasets: {raw_datasets}')
    print(raw_datasets['train'].num_rows)
    n_train_samples = raw_datasets['train'].num_rows
    print(f'n_train_samples = {n_train_samples}, len(raw_datasets["train"]) = {len(raw_datasets["train"])}')

    raw_datasets["eval"] = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        split=data_args.eval_split_name,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if data_args.audio_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--audio_column_name {data_args.audio_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--audio_column_name` to the correct audio column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    if data_args.label_column_name not in raw_datasets["train"].column_names:
        raise ValueError(
            f"--label_column_name {data_args.label_column_name} not found in dataset '{data_args.dataset_name}'. "
            "Make sure to set `--label_column_name` to the correct text column - one of "
            f"{', '.join(raw_datasets['train'].column_names)}."
        )

    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name or model_args.model_name_or_path,
        return_attention_mask=model_args.attention_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # `datasets` takes care of automatically loading and resampling the audio,
    # so we just need to set the correct target sampling rate.
    raw_datasets = raw_datasets.cast_column(
        data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    )

    model_input_name = feature_extractor.model_input_names[0]

    def train_transforms(batch):
        """Apply train_transforms across a batch."""
        subsampled_wavs = []
        for audio in batch[data_args.audio_column_name]:
            wav = random_subsample(
                audio["array"], max_length=data_args.max_length_seconds, sample_rate=feature_extractor.sampling_rate
            )
            subsampled_wavs.append(wav)
        inputs = feature_extractor(subsampled_wavs, sampling_rate=feature_extractor.sampling_rate)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch[data_args.label_column_name])

        return output_batch

    def val_transforms(batch):
        """Apply val_transforms across a batch."""
        wavs = [audio["array"] for audio in batch[data_args.audio_column_name]]
        inputs = feature_extractor(wavs, sampling_rate=feature_extractor.sampling_rate)
        output_batch = {model_input_name: inputs.get(model_input_name)}
        output_batch["labels"] = list(batch[data_args.label_column_name])

        return output_batch

    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    # labels = raw_datasets["train"].features[data_args.label_column_name].names # Original code
    labels = list(set(raw_datasets["train"][data_args.label_column_name]))
    print(f'len(labels): {len(labels)}')
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        finetuning_task="audio-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    # freeze the convolutional waveform encoder
    if model_args.freeze_feature_encoder:
        model.freeze_feature_encoder()

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            raw_datasets["train"] = (
                raw_datasets["train"].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        raw_datasets["train"].set_transform(train_transforms, output_all_columns=False)

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            raw_datasets["eval"] = (
                raw_datasets["eval"].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        raw_datasets["eval"].set_transform(val_transforms, output_all_columns=False)

    # Training_dynamic
    class DatamapTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.training_instance_probs: List[float] = []
            self.training_instance_corrects: List[bool] = []

        def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
        ) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.
            Subclass and override to inject custom behavior.
            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets
                    under the argument `labels`. Check your model's documentation for all accepted arguments.
            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

            predict_probs = outputs["logits"].detach().softmax(dim=-1).cpu()
            predict_labels = predict_probs.argmax(dim=-1)
            gold_labels = inputs["labels"].cpu()
            correct = predict_labels.eq(gold_labels)

            # get probability of gold label of each instance
            gold_probs = predict_probs.gather(
                dim=-1, index=gold_labels.unsqueeze(-1)
            ).squeeze(-1)

            self.training_instance_probs.extend(gold_probs.tolist())
            self.training_instance_corrects.extend(correct.tolist())

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

            return loss.detach()

    # Initialize our trainer
    trainer = DatamapTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        print(f'metrics: {metrics}')
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else n_train_samples
        )
        metrics["train_samples"] = min(max_train_samples, n_train_samples)

        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

        # training_instance_dynamic_gold_probs: (num_instances, num_epochs)
        training_instance_dynamic_gold_probs = np.array(trainer.training_instance_probs)
        training_instance_dynamic_gold_probs = (
            training_instance_dynamic_gold_probs.reshape(metrics["train_samples"], -1)
        )
        # training_instance_gold_prob_means: (num_instances)
        training_instance_gold_prob_means = np.mean(
            training_instance_dynamic_gold_probs, axis=1
        )
        # training_instance_gold_prob_stds: (num_instances)
        training_instance_gold_prob_stds = np.std(
            training_instance_dynamic_gold_probs, axis=1
        )

        # training_instance_correct: (num_instances)
        training_instance_correct = np.array(trainer.training_instance_corrects)
        training_instance_correct = training_instance_correct.reshape(
            metrics["train_samples"], -1
        )
        # training_instance_correct_means: (num_instances)
        training_instance_correct_means = np.mean(training_instance_correct, axis=1)
        training_dynamics = {
            "gold_prob_means": training_instance_gold_prob_means.tolist(),
            "gold_prob_stds": training_instance_gold_prob_stds.tolist(),
            "correct_means": training_instance_correct_means.tolist(),
        }

        # Save training dynamics
        output_dynamics_file = os.path.join(
            training_args.output_dir, f"audio_training_dynamics.json"
        )
        with open(output_dynamics_file, "w") as writer:
            json.dump(training_dynamics, writer)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "audio-classification",
        "dataset": data_args.dataset_name,
        "tags": ["audio-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()