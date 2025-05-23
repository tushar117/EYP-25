#!/bin/bash

LOG_DIR=./log
mkdir -p ${LOG_DIR}

for idx in $(seq 0 7)
do
    part=$((idx+1))
    echo "working on part : ${part}"
    export CUDA_VISIBLE_DEVICES=${idx}
    rm -f ${LOG_DIR}/Phi-3-medium-instruct-inference-${part}.log;CUDA_VISIBLE_DEVICES=${idx};python vllm_based.py \
                --model_path microsoft/Phi-3-medium-4k-instruct \
                --input_file ./phi-3-medium-14B-input/phi-3-medium-14B-input_part_${part}.jsonl \
                --output_file ./output_splits/part_${part}.jsonl \
                --batch_size 12 \
                --max_src_token 100 \
                --max_gen_token 256 2>&1 | tee -a ${LOG_DIR}/Phi-3-medium-instruct-inference-${part}.log &
done