#!/bin/bash
# used vllm_marlin environment

LOG_DIR=./log
mkdir -p ${LOG_DIR}

echo "starting inference"
rm -f ${LOG_DIR}/Phi3-medium-4k-instruct-inference.log;python generate.py \
            --model_path microsoft/Phi-3-medium-4k-instruct \
            --input_file /data/users/user/ghosting/EYP/llama_70B_inference/llama-70B-input.jsonl \
            --output_file sample_v2.jsonl \
            --batch_size 16 \
            --max_src_token 100 \
            --max_limit 100 \
            --max_gen_token 256 2>&1 | tee -a ${LOG_DIR}/Phi3-medium-4k-instruct-inference.log