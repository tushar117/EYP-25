import argparse
import torch
from copy import deepcopy
import os
import sys
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import re
import math
from datetime import datetime
import numpy as np
import json
from vllm import LLM, SamplingParams


def load_txt(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as dfile:
        for line in dfile.readlines():
            if line[-1]=='\n':
                line = line[:-1]
            res.append(line)
    return res

def load_text_stream(load_path, max_limit=-1):
    index = 0
    with open(load_path, 'r', encoding='utf-8') as f:
        for row in f:
            index+=1
            input_text = row
            if input_text[-1]=='\n':
                input_text = input_text[:-1]
            if (max_limit>0 and index>max_limit):
                yield False, ""
            yield True, input_text
    yield False, ""

# modifiy the load_text_stream to function to have begin and end indexes
def load_text_stream_with_index(load_path, max_limit=-1, start_index:int=0):
    index = -1
    with open(load_path, 'r', encoding='utf-8') as f:
        for row in f:
            index+=1
            if index<start_index:
                continue
            input_text = row
            if input_text[-1]=='\n':
                input_text = input_text[:-1]
            if (max_limit>0 and index>=max_limit):
                yield False, "", index
            yield True, input_text, index
    yield False, "", index

def normalize_text(text):
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    return text

def truncate_string(input_string, max_words):
    max_avg_chars = 5*max_words # space + 4 char word = 200 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    return reconstructed_input[-max_avg_chars:]

def create_template(query):
    messages = []
    # adding instructions
    messages.append({"role": "user", "content": "You are an helpful AI assistant who respond to user queries in **less than 200 words**."})
    # adding bot response
    messages.append({"role": "assistant", "content": "sure, I will respond to user queries in less than 200 words."})
    # new inference
    messages.append({"role": "user", "content": query})
    return messages

def process_text(text, max_length, tokenizer):
    # prepare context
    text = normalize_text(text)
    text = truncate_string(text, 40)
    text_str = tokenizer.apply_chat_template(create_template(text), tokenize=False, add_generation_prompt=True)
    
    trimmed_text_str = text_str
    input_tokens = tokenizer.encode(trimmed_text_str)
    if len(input_tokens)>max_length:
        trimmed_text_str = tokenizer.decode(input_tokens[-max_length:])
    is_trimmed = trimmed_text_str!=text_str
    return text, trimmed_text_str, is_trimmed

def store_jsonl(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile:
        for line in res:
            json.dump(line, dfile, ensure_ascii=False)
            dfile.write('\n')
    print("writtent %d lines to json file : %s" % (len(res), file_path))       

def get_prefix_count(input_file, max_limit, start_index):
    # iterate over the input file and count the number of lines
    count=0
    stream = load_text_stream_with_index(input_file, max_limit, start_index)
    while True:
        status, curr_input, idx = next(stream)
        if status==False:
            break
        # check json loads
        temp = json.loads(curr_input.strip())
        assert "query" in temp, f"query not present in record : {idx}"
        count+=1
    print("[ok] all instances in file are json")
    return count

def inference(args):
    # loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, add_eos_token=True)
    # loading model
    llm_params = {"model": args.model_path, 
                "tokenizer": args.model_path,
                "tokenizer_mode": "auto",
                "dtype": "float16",
                "trust_remote_code": True,
                # "tensor_parallel_size": torch.cuda.device_count(),
                "max_seq_len_to_capture": 512}
    model = LLM(**llm_params)
    print("model loaded successfully")
    # model sampling params
    sampling_params = SamplingParams(max_tokens=args.max_gen_token)
    output_file = open(args.output_file, 'w', encoding='utf-8')
    prefix_count = args.prefix_count
    print("queries for inference : %d" % (prefix_count))
    args.inference_count = 0
    args.trimmed_prefixes = 0    
    
    print("starting the inference ...")
    batches = []
    records = []
    stream = load_text_stream_with_index(args.input_file, args.max_limit, 0)
    with tqdm(total=math.ceil(prefix_count/args.batch_size), desc="inference", unit=" lines", position=0, leave=True) as pbar:
        for status, raw_record, _ in stream: 
            if status==False:
                break
            record = json.loads(raw_record)
            _, query, is_trimmed = process_text(record["query"], args.max_src_token, tokenizer)
            batches.append(query)
            records.append(record)
            args.trimmed_prefixes+=is_trimmed
            # main logic
            if len(batches)==args.batch_size:
                model_response = model.generate(batches, sampling_params, use_tqdm=False)
                args.inference_count+=len(batches)
                assert len(batches)==len(model_response), f"unequal prefix and inference {len(batches)} != {len(model_response)}"
                for ip, op in zip(records, model_response):
                    temp_out = deepcopy(ip)
                    temp_out["response"] = op.outputs[0].text.strip()
                    temp_out["gen_tokens"] = len(op.outputs[0].token_ids)
                    output_file.write(json.dumps(temp_out, ensure_ascii=False) + "\n")
                batches = []
                records = []
                model_response = []
                pbar.update(1)

        # process the remaining batches
        if len(batches)>0:
            model_response = model.generate(batches, sampling_params, use_tqdm=False)
            args.inference_count+=len(batches)
            assert len(batches)==len(model_response), f"unequal prefix and inference {len(batches)} != {len(model_response)}"
            for ip, op in zip(records, model_response):
                temp_out = deepcopy(ip)
                temp_out["response"] = op.outputs[0].text.strip()
                temp_out["gen_tokens"] = len(op.outputs[0].token_ids)
                output_file.write(json.dumps(temp_out, ensure_ascii=False) + "\n")
            batches = []
            records = []
            model_response = []
            pbar.update(1)
        
        output_file.close()
        print("number of queries trimmed : %d / %d [%0.2f]" % (args.trimmed_prefixes, args.inference_count, args.trimmed_prefixes/args.inference_count))

# function to delete the file if it exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# function to print time delta in human readable format hh:mm:ss
def get_time_delta(time_delta):
    hours, rem = divmod(time_delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global model configuration
    parser.add_argument('--model_path', type=str, required=True,
                        help='')
    parser.add_argument('--input_file', type=str, required=True,
                        help='')
    parser.add_argument('--output_file', type=str, required=True,
                        help='')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='')
    parser.add_argument('--max_limit', type=int, default=-1,
                        help='')
    parser.add_argument('--max_gen_token', type=int, default=256,
                        help='')
    parser.add_argument('--max_src_token', type=int, default=40,
                        help='')
    args = parser.parse_args()

    global_start_time = datetime.utcnow()
    print("argument passed :")
    for arg in vars(args):
        print("%s - %s" % (arg, getattr(args, arg)))
    print('--'*30)
    args.prefix_count = get_prefix_count(args.input_file, args.max_limit, 0)
    print("total query count : %d" % args.prefix_count)
    inference_count = 0
    trimmed_prefix_count = 0
    delete_file_if_exists(args.output_file)
    inference(args)
    time_delta = (datetime.utcnow() - global_start_time)
    print("completed %d inference in %s" % (args.prefix_count, get_time_delta(time_delta)))