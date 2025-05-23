#!/bin/bash
# this script will start the multi-gpu inference for phi-1.5 model
output_dir=./model_outputs
log_dir=./gen_logs
peft_checkpoint_path="./eyp_phi_3/checkpoint-25000" # please update the checkpoint path


mkdir -p ${output_dir}
mkdir -p ${log_dir}

eval_file=../data/test.jsonl

echo "working on EYP phi-3"
rm -f ${log_dir}/phi_context.log;python generate_copy.py \
        --model_path ${peft_checkpoint_path} \
        --input_file ${eval_file} \
        --output_file ${output_dir}/phi_eyp_v3.jsonl \
        --num_workers 8 \
        --context_type "both" \
        --max_src_token 200 --max_gen_token 400 \
        --batch_size 16 --beam_size 2 --num_seq 1 2>&1 | tee -a ${log_dir}/phi_eyp.log