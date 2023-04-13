#!/usr/bin/env bash

output_dir="${1:-../datamap_audio}"                 # Need to modify
dataset_name="${2:-yhfang/slurp_dataset_audio}" 
seed="${3:-0}"
model_name="${4:-facebook/hubert-large-ls960-ft}"
train_epochs="${5:-5}"
batch_size="${6:-2}"
lr="${7:-"1e-5"}"
label_column_name="${8:-intent}"

python ./run_audio_classification.py \
    --dataset_name "${dataset_name}" \
    --label_column_name "${label_column_name}" \
    --model_name_or_path "${model_name}" \
    --output_dir "${output_dir}" \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train True \
    --do_eval False \
    --learning_rate "${lr}" \
    --max_length_seconds 15 \
    --attention_mask False \
    --warmup_ratio 0.1 \
    --num_train_epochs "${train_epochs}" \
    --per_device_train_batch_size "${batch_size}" \
    --per_device_eval_batch_size "${batch_size}" \
    --dataloader_num_workers 4 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 1 \
    --seed "${seed}"