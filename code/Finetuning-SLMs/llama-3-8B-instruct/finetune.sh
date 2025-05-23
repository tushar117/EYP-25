#!/bin/bash

# please update your hf_token

log_dir="./log"
mkdir -p ${log_dir}

for level in $(seq 1 4)
do
    echo "working on level ${level} ..."
    export EYP_LEVEL=${level}
    export HF_TOKEN=<your_hf_token>
    rm -f ${log_dir}/lvl-${level}.log;python -m torch.distributed.launch \
                    --nproc_per_node=8 llama_3_8B_script.py 2>&1 | tee -a ${log_dir}/lvl-${level}.log
done