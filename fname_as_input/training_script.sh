#!/usr/bin/env bash
train_file="${1:-./dataset/train_fname.json}"
valid_file="${2:-./dataset/valid_fname.json}"
model_name="${3:-distilroberta-base}"
output_dir="${4:-./output}"
seed="${5:-1226}"
lr="${6:-"1e-5"}"

CUDA_VISIBLE_DEVICES=2 python run_text_classification.py \
    --max_seq_length 30 \
    --train_file "${train_file}" \
    --validation_file "${valid_file}" \
    --model_name_or_path "${model_name}" \
    --output_dir "${output_dir}" \
    --overwrite_output_dir True\
    --do_train True \
    --do_eval True \
    --do_predict False \
    --evaluation_strategy "epoch" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 d\
    --num_train_epoch 20 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --load_best_model_at_end \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --learning_rate "${lr}"