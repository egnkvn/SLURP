training_dynamics="${1}"
dataset_file="${2}"
save_file="${3}"
threshold="${4}"

python ../training_dynamics_analysis.py \
 --method "pipeline" \
 --training_dynamics_path "${training_dynamics}" \
 --filename_info "${dataset_file}" \
 --save_path "${save_file}" \
 --hard_thre "${threshold}"