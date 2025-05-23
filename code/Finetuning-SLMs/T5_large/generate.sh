#!/bin/bash
output_dir=../model_output
inference_log_dir=../model_output/teacher_logs

model_checkpoint_path=$1
inference_data=$2
num_gpus=$3

mkdir -p ${output_dir}
mkdir -p ${inference_log_dir}

rm -f ${inference_log_dir}/teacher_inference.log;python generator.py \
            --model_path ${model_checkpoint_path} --input_file ${inference_data} \
            --output_file ${output_dir}/teacher_outputs.jsonl \
            --max_gen_token 500 --batch_size 4 --beam_size 5 --num_seq 1 \
            --num_workers ${num_gpus} 2>&1 | tee -a ${inference_log_dir}/teacher_inference.log