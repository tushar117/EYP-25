#!/bin/bash
val_data=-1
train_data=-1

datadir=$1
log_dir=./logs

mkdir -p ${log_dir}

echo "working on validation data"
rm -f ${log_dir}/english_val.log;python upload_to_redis.py --file_path ${datadir}/val_t5.tsv --dtype val --data-limit ${val_data} --log-freq 100000 2>&1 | tee -a ${log_dir}/english_val.log

echo "working on training data"
rm -f ${log_dir}/english_train.log;python upload_to_redis.py --file_path ${datadir}/train_t5.tsv --dtype train --data-limit ${train_data} 2>&1 | tee -a ${log_dir}/english_train.log

