training_dynamics="${1:-./training_dynamic/google_golden.json}"
dataset_file="${2:-./datasets/google/train_golden.json}"
save_file="${3:-./training_dynamic/golden_correctness.json}"
threshold="${4:-1}"

python ../correctness_training_dynamics_analysis.py \
 --method "pipeline" \
 --training_dynamics_path "${training_dynamics}" \
 --filename_info "${dataset_file}" \
 --save_path "${save_file}" \
 --ambig_thres "$((100 - threshold))" \
 --hard_thres "${threshold}" \
 --easy_thres "$((100 - threshold))"