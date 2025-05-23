#!/bin/bash
log_dir="./inference_log"
mkdir -p ${log_dir}

#please update your hf_token

for level in $(seq 1 4)
do
    echo "working on level ${level} ..."
    export HF_TOKEN=<your_hf_token>
    rm -f ${log_dir}/lvl-${level}.log;python generate.py \
                    --model_path /data/users/user/ghosting/EYP/code/checkpoints/llama_3_8B_lvl_${level}/checkpoint-6248 \
                    --input_file /data/users/user/ghosting/EYP/dataset/test.tsv \
                    --batch_size 8 --level ${level} \
                    --output_file /data/users/user/ghosting/EYP/model_outputs/lvl${level}/output.jsonl \
                    --num_workers 8 2>&1 | tee -a ${log_dir}/lvl-${level}.log
done