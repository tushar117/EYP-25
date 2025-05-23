#!/bin/bash
# this script will start the multi-gpu inference for phi-1.5 model
output_dir=./model_outputs
log_dir=./gen_logs
checkpoint_path="./EYP_model/checkpoint-93750" # please update the checkpoint path


mkdir -p ${output_dir}
mkdir -p ${log_dir}

eval_file=../data/test.jsonl

echo "working on EYP GPT2-medium"
rm -f ${log_dir}/phi_context.log;python generator_gpt_copy.py \
        --model_path ${checkpoint_path} \
        --input_file ${eval_file} \
        --output_file ${output_dir}/gpt2_eyp.jsonl \
        --num_workers 8 \
        --context_type "both" \
        --max_src_token 200 --max_gen_token 400 \
        --batch_size 16 --beam_size 2 --num_seq 1 2>&1 | tee -a ${log_dir}/gpt2_eyp.log