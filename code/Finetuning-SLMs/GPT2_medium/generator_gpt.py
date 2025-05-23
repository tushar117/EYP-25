import argparse
import torch
import torch.multiprocessing as mp
from copy import deepcopy
import os
import sys
import json
from tqdm import tqdm
import re
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import math
from datetime import datetime
import json
from transformers import set_seed
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
import json

class TextData(Dataset):
    def __init__(self, tokenizer, data, src_max_seq_len):
        self.trimmed_prefix_count = 0
        self.tokenizer = tokenizer
        self.data = data
        self.src_max_seq_len = src_max_seq_len

    def preprocess(self, text, max_seq_len):
        tokenzier_args = {'text': text, 'truncation': True, 'pad_to_max_length': False, 
                                    'max_length': max_seq_len, 'return_attention_mask': True}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'], tokenized_data['attention_mask']

def __getitem__(self, idx):
        data_instance = self.data[idx]
        trimmed_data = data_instance
        input_tokens = self.tokenizer.encode(data_instance)
        # truncating the data from right side
        if len(input_tokens)-1>self.src_max_seq_len:
            self.trimmed_prefix_count+=1
            trimmed_data = self.tokenizer.decode(input_tokens[-self.src_max_seq_len:-1])

        # preparing the input
        input_ids, input_mask = self.preprocess(trimmed_data, self.src_max_seq_len)
        return input_ids, input_mask

def __len__(self):
        return len(self.data)

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
    text = text.lower().strip()
    text = re.sub('\s+', ' ', text)
    return text

def truncate_string(input_string, max_words):
    max_avg_chars = 5*max_words # space + 4 char word = 200 chars
    input_words = input_string.split(' ')[-max_words:]
    reconstructed_input = " ".join(input_words)
    return reconstructed_input[-max_avg_chars:]

def process_text(text):
    record = json.loads(text)
    query = truncate_string(record["query"].lower(), 400)
    return query

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
        # check json loads
        temp = json.loads(curr_input.strip())
        count+=1
    print("[ok] all instances in file are json")
    return count


def process_response(res):
    completions = res.split(" ###> ")
    if len(completions)==2:
        return completions[-1].rstrip()
    return ""


# test_set = load_dataset('../data', data_files={'test': 'test.jsonl'})

def inference(args):
    print("[worker-%d] using device : %s" % (args.worker_id, args.device))
    # loading tokenizer
    model_name = "openai-community/gpt2-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
    tokenizer.padding_side = "left"
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.to(args.device)
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
        stream = load_text_stream_with_index(args.input_file, args.max_limit, args.start_index)
        with tqdm(total=math.ceil(prefix_count/args.batch_size), desc="inference", unit=" lines", position=0, leave=True) as pbar:
            for status, prefix, _ in stream: 
                if status==False:
                    break
                batches.append(process_text(prefix))
                # main logic
                if len(batches)==args.batch_size:
                    inputs = tokenizer.batch_encode_plus(batches, padding=True, return_tensors="pt")
                    outputs = model.generate(
                        input_ids=inputs["input_ids"].to(args.device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        attention_mask=inputs["attention_mask"].to(args.device),
                        early_stopping=True,
                        max_new_tokens=200,
                        num_beams=5
                    )
                    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    res = [process_response(tz) for tz in res]
                    args.inference_count+=len(batches)
                    assert len(batches)==len(res), f"unequal prefix and inference {len(batches)} != {len(res)}"
                    for x, y in zip(batches, res):
                        output_file.write(json.dumps({"query": x, "enhanced_query": y}, ensure_ascii=False) + "\n")
                    batches = []
                    res = []
                    pbar.update(1)

            # process the remaining batches
            if len(batches)>0:
                inputs = tokenizer.batch_encode_plus(batches, padding=True, return_tensors="pt")
                outputs = model.generate(
                        input_ids=inputs["input_ids"].to(args.device),
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        attention_mask=inputs["attention_mask"].to(args.device),
                        early_stopping=True,
                        max_new_tokens=200,
                        num_beams=5
                    )
                res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                res = [process_response(tz) for tz in res]
                args.inference_count+=len(batches)
                assert len(batches)==len(res), f"unequal prefix and inference {len(batches)} != {len(res)}"
                for x, y in zip(batches, res):
                    output_file.write(json.dumps({"query": x, "enhanced_query": y}, ensure_ascii=False) + "\n")
                batches = []
                res = []
                pbar.update(1)
        output_file.close()
        print("[worker-%d] number of prefixes trimmed : %d / %d [%0.2f]" % (args.worker_id,args.trimmed_prefixes, args.inference_count, args.trimmed_prefixes/args.inference_count))
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
    parser.add_argument('--beam_size', type=int, default=5,
                        help='')
    parser.add_argument('--num_seq', type=int, default=1,
                        help='')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='')
    parser.add_argument('--max_limit', type=int, default=-1,
                        help='')
    parser.add_argument('--max_gen_token', type=int, default=5,
                        help='')
    parser.add_argument('--max_src_token', type=int, default=200,
                        help='')
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='')
    parser.add_argument('--context_len', type=int, default=-1,
                        help='')
    parser.add_argument("--context_type", type=str, default="both", choices=["last_user_msg", "suggestion_chips", "both", "only_prefix"], 
                        help='specify the context type')
    # if multiple gpus devices are available
    parser.add_argument('--num_workers', type=int, default=1, 
                        help='total number of workers')                                     
    args = parser.parse_args()

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