#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <checkpoint_dir> <inference_data> <num_gpus>"
    exit 1
fi

checkpoint_dir=$1
inference_data=$2
num_gpus=$3

output_dir=../model_output_level_1
inference_log_dir=../model_output_h/teacher_logs

mkdir -p ${output_dir}
mkdir -p ${inference_log_dir}

for epoch_dir in "${checkpoint_dir}"/*; do
    echo "Checking directory: ${epoch_dir}"

    # Skip if not a directory
    if [ ! -d "${epoch_dir}" ]; then
        echo "Not a directory: ${epoch_dir}, skipping..."
        continue
    fi

    # Extract the epoch number or name from the directory name
    epoch_name=$(basename "${epoch_dir}")

    # Define the output and log file names based on the epoch
    output_file="${output_dir}/teacher_outputs_${epoch_name}.jsonl"
    log_file="${inference_log_dir}/teacher_inference_${epoch_name}.log"

    # Check if the output file already exists, skip if it does
    if [ -f "${output_file}" ]; then
        echo "Output file ${output_file} already exists, skipping ${epoch_name}..."
        continue
    fi

    model_checkpoint_path="${epoch_dir}"
    echo "Model checkpoint path: ${model_checkpoint_path}"

    # Run the inference using the current checkpoint
    echo "Running inference for ${epoch_name}..."
    python generator.py \
        --model_path ${model_checkpoint_path} --input_file ${inference_data} \
        --output_file ${output_file} \
        --max_gen_token 500 --batch_size 4 --beam_size 5 --num_seq 1 \
        --num_workers ${num_gpus} 2>&1 | tee -a ${log_file}
done