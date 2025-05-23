#!/bin/bash
base_dir=./model_run_t5large

#model hyperparamters
epochs=50
batch_size=4 # change required as per the GPU memory
ga=2 # gradient accumulation
num_gpus=8 # change required as per the GPUs availability

model_run_path=${base_dir}/en_only/teacher/T5_ep${epochs}_bs${batch_size}_ga${ga}_gpus${num_gpus}
mkdir -p ${model_run_path}

# important directories
log_dir=${model_run_path}/logs
mkdir -p ${log_dir}
lightning_checkpoint_path=${model_run_path}/lightning_checkpoints # helps resume the training process
mkdir -p ${lightning_checkpoint_path}
hf_checkpoint_path=${model_run_path}/hf_checkpoints
mkdir -p ${hf_checkpoint_path}

rm -f ${log_dir}/cmdline_teacher.log;python main.py \
        --model_path t5-large \
        --batch_size ${batch_size} --checkpoint_path ${lightning_checkpoint_path} \
        --val_batch_size ${batch_size} --log_dir ${log_dir} \
        --weight_decay 0.0 --epochs ${epochs} \
        --hf_checkpoint ${hf_checkpoint_path} --grad_accum ${ga} \
        --num_worker 4 --gpus ${num_gpus} 2>&1 | tee -a ${log_dir}/cmdline_teacher.log
