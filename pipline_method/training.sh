#!/usr/bin/env bash
train_file="${1:-./datasets/google/train_golden.json}"
valid_file="${2:-./datasets/google/valid_golden.json}"
model_name="${3:-distilroberta-base}"
output_dir="${4:-./output/google_golden}"
seed="${5:-1226}"
lr="${6:-"1e-5"}"

CUDA_VISIBLE_DEVICES=0 python run_text_classification.py \
    --max_seq_length 128 \
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
    --per_device_eval_batch_size 16 \
    --num_train_epoch 5 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed "${seed}" \
    --data_seed "${seed}" \
    --load_best_model_at_end \
    --metric_for_best_model "accuracy" \
    --greater_is_better True \
    --learning_rate "${lr}"