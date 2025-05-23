import argparse
import torch
import torch.multiprocessing as mp
from copy import deepcopy
import os
import sys
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
    )
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import math
from datetime import datetime
import numpy as np
import json
from peft import PeftModel
from transformers import set_seed

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
    max_avg_chars = 5*max_words # space + 4 char word = 5 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    return reconstructed_input[-max_avg_chars:]

def get_level_example(level):
    messages = []
    messages.append({"role": "user", "content": "##query##: swiss vacation plan"})
    if level==1:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland?"})
    elif level==2:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland for <Number of People> at a budget of <Budget>"})
    elif level==3:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland for <2 people> at a budget of <5000 Euros>"})
    else:
        messages.append({"role": "system", "content": "Can you create a vacation plan for Switzerland for <2 people/5 people/10 people> at a budget of <5000 Euros/200000 INR/5000 USD>"})
    return messages

def create_template(query, level):
    messages = []
    # adding instructions
    messages.append({"role": "system", "content": "You are an helpful AI assistant that enhances user queries submitted to conversational LLM-based chat bot system like ChatGPT or Microsoft Copilot. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding examples:
    messages.extend(get_level_example(level))
    # adding current user query
    messages.append({"role": "user", "content": f"##query##: {query}"})
    return messages

def process_text(query, level, max_length, tokenizer):
    # prepare context
    query = truncate_string(normalize_text(query), 40)
    text_str = tokenizer.apply_chat_template(create_template(query, level), tokenize=False, add_generation_prompt=True)
    trimmed_text_str = text_str
    input_tokens = tokenizer.encode(trimmed_text_str)
    if len(input_tokens)>max_length:
        trimmed_text_str = tokenizer.decode(input_tokens[-max_length:])
    is_trimmed = trimmed_text_str != text_str
    return trimmed_text_str, is_trimmed

def store_jsonl(res, file_path):
    with open(file_path, 'w', encoding='utf-8') as dfile:
        for line in res:
            json.dump(line, dfile, ensure_ascii=False)
            dfile.write('\n')
    print("writtent %d lines to json file : %s" % (len(res), file_path))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)        

def get_prefix_count(input_file, max_limit, start_index):
    # iterate over the input file and count the number of lines
    count=0
    stream = load_text_stream_with_index(input_file, max_limit, start_index)
    while True:
        status, curr_input, _ = next(stream)
        if status==False:
            break
        # check tsv files first column should be query
        records = curr_input.strip().split('\t')
        temp = records[0]
        count+=1
    print("[ok] all instances in file are json")
    return count

def inference(args):
    print("[worker-%d] using device : %s" % (args.worker_id, args.device))
    model_id="meta-llama/Meta-Llama-3-8B-Instruct"
    # loading the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_eos_token=True)
    tokenizer.pad_token = "<|reserved_special_token_0|>"  # use special token rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'left'

    #quantization setting
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # loading the base model
    base_model_reload = AutoModelForCausalLM.from_pretrained(
            model_id, 
            cache_dir="/data/users/user/ghosting/SLM/llama_cache",
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map = args.device,
            use_flash_attention_2=False)
    # loading the LoRA checkpoints
    model = PeftModel.from_pretrained(base_model_reload, args.model_path)
    model.config.eos_token_id = tokenizer.eos_token_id
    print("[worker-%d] model loaded successfully" % (args.worker_id))
    model = model.to(args.device).eval()
    print("[worker-%d] total number of parameters : %d" % (args.worker_id, count_parameters(model)))
    output_file = open(args.output_file, 'w', encoding='utf-8')
    prefix_count = args.prefix_count
    print("[worker-%d] prefixes for inference : %d" % (args.worker_id, prefix_count))
    print("[worker-%d] prefixes start index : %d, end_index : %d" % (args.worker_id, args.start_index, args.max_limit))
    args.inference_count = 0
    args.trimmed_prefixes = 0    
    with torch.no_grad():
        print("[worker-%d] starting the inference ..." % args.worker_id)
        batches = []
        actual_queries = []
        stream = load_text_stream_with_index(args.input_file, args.max_limit, args.start_index)
        with tqdm(total=math.ceil(prefix_count/args.batch_size), desc="inference", unit=" lines", position=0, leave=True) as pbar:
            for status, record, _ in stream: 
                if status==False:
                    break
                records = record.split('\t')
                query = records[0].strip()
                formatted_query, is_trimmed = process_text(query, args.level, args.max_src_token, tokenizer)
                
                actual_queries.append(query)
                batches.append(formatted_query)
                
                args.trimmed_prefixes+=is_trimmed
                # main logic
                if len(batches)==args.batch_size:
                    inputs = tokenizer.batch_encode_plus(batches, padding=True, return_tensors="pt")
                    input_length = 1 if model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
                    outputs = model.generate(
                        input_ids=inputs["input_ids"].to(args.device),
                        attention_mask=inputs["attention_mask"].to(args.device),
                        max_new_tokens=args.max_gen_token, 
                        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.9)
                    generated_tokens = outputs[:, input_length:]
                    res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    args.inference_count+=len(batches)
                    assert len(batches)==len(res), f"unequal prefix and inference {len(batches)} != {len(res)}"
                    for q, r in zip(actual_queries, res):
                        output_file.write(json.dumps({"query": q, "enhanced_query": r.strip()}, ensure_ascii=False) + "\n")
                    batches = []
                    res = []
                    actual_queries = []
                    torch.cuda.empty_cache()
                    pbar.update(1)

            # process the remaining batches
            if len(batches)>0:
                inputs = tokenizer.batch_encode_plus(batches, padding=True, return_tensors="pt")
                input_length = 1 if model.config.is_encoder_decoder else inputs["input_ids"].shape[1]
                outputs = model.generate(
                    input_ids=inputs["input_ids"].to(args.device),
                    attention_mask=inputs["attention_mask"].to(args.device),
                    max_new_tokens=args.max_gen_token, 
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9)
                generated_tokens = outputs[:, input_length:]
                res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                args.inference_count+=len(batches)
                assert len(batches)==len(res), f"unequal prefix and inference {len(batches)} != {len(res)}"
                for q, r in zip(actual_queries, res):
                    output_file.write(json.dumps({"query": q, "enhanced_query": r.strip()}, ensure_ascii=False) + "\n")
                batches = []
                res = []
                actual_queries = []
                torch.cuda.empty_cache()
                pbar.update(1)
        
        output_file.close()
        print("[worker-%d] number of prefixes trimmed : %d / %d [%0.2f]" % (args.worker_id,args.trimmed_prefixes, args.inference_count, args.trimmed_prefixes/args.inference_count))

# function to delete the file if it exists
def delete_file_if_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

# function to get the filename and parent directory given the complete file path
def create_worker_file_name(idx, file_path):
    file_name = os.path.basename(file_path)
    parent_dir = os.path.dirname(file_path)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"worker-{idx}-{file_name}")

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
    parser.add_argument('--max_src_token', type=int, default=256,
                        help='')
    parser.add_argument('--level', type=int, required=True,
                        help='')
    # if multiple gpus devices are available
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='total number of workers')                                     
    args = parser.parse_args()

    # creating if not present
    base_dir = os.path.dirname(args.output_file)
    os.makedirs(base_dir, exist_ok=True)
    
    global_start_time = datetime.utcnow()
    print("[global] argument passed :")
    for arg in vars(args):
        print("[global] %s - %s" % (arg, getattr(args, arg)))
    print('--'*30)
    args.global_prefix_count = get_prefix_count(args.input_file, args.max_limit, 0)
    print("[global] total prefix count : %d" % args.global_prefix_count)
    inference_count = 0
    trimmed_prefix_count = 0
    if args.num_workers==1:
        args.worker_id = 0
        args.start_index = 0
        args.prefix_count = args.global_prefix_count
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        inference(args)
    else:
        mp.set_start_method('spawn')
        # Multi-GPU inference
        bucket_size = math.ceil(args.global_prefix_count/args.num_workers)
        worker_configs = []
        # creating the config
        for idx in range(args.num_workers):
            worker_config = deepcopy(args)
            worker_config.worker_id = idx
            worker_config.start_index = idx*bucket_size
            worker_config.max_limit = min(worker_config.start_index + bucket_size, args.global_prefix_count)
            worker_config.prefix_count = (worker_config.max_limit - worker_config.start_index)
            worker_config.device = f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
            worker_config.output_file = create_worker_file_name(idx, os.path.abspath(args.output_file))
            worker_configs.append(worker_config)
        # spawning the process
        worker_processes = []
        print("[global] spawning %d inference processes" % args.num_workers)
        for idx in range(args.num_workers):
            p = mp.Process(target=inference, args=(worker_configs[idx],))
            p.start()
            worker_processes.append(p)
        # wait for all the process to finish
        for p in worker_processes:
            p.join()
        
        # merge intermediate outputs to final file 
        final_output_file = open(args.output_file, 'w', encoding='utf-8')
        for wc in worker_configs:
            stream = load_text_stream(wc.output_file)
            for status, data in stream:
                if status==False:
                    break
                final_output_file.write("%s\n" % data)
        final_output_file.close()
        
        # delete the intermediate files 
        for wc in worker_configs:
            delete_file_if_exists(wc.output_file)

    time_delta = (datetime.utcnow() - global_start_time)
    print("[global] completed %d inference in %s" % (args.global_prefix_count, get_time_delta(time_delta)))