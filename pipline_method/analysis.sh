training_dynamics="${1:-./training_dynamic/golden_5.json}"
save_file="${2:-./training_dynamic/analysis/golden.json}"
threshold="${3:-5}"

python ../training_dynamics_analysis.py \
 --training_dynamics_path "${training_dynamics}" \
 --save_path "${save_file}" \
 --ambig_thres "$((100 - threshold))" \
 --hard_thres "${threshold}" \
 --easy_thres "$((100 - threshold))"