#/bin/bash
# following command will execute model training on 8 GPUs
accelerate launch --multi_gpu --num_processes 8 finetuning_phi.py