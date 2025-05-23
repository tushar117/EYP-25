from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

import math
import torch
from task_utils import normalize_text, truncate_string
import sys
from functools import partial 
import os
import random

torch.random.manual_seed(0)
IGNORE_INDEX=-100
MAX_SEQ_LEN=512

model_id="microsoft/Phi-3-mini-4k-instruct"
# loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_eos_token=True)
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

def create_template(query, enhanced_query):
    messages = []
    # adding instructions
    messages.append({"role": "user", "content": "Can you enhance some queries that I will give you. Please think about more relevant details to improve this query and make it more structured. Please do not answer the query, just give back the enhanced query."})
    # adding bot response
    messages.append({"role": "assistant", "content": "sure, I will not answer the queries. I will just rephrase the query you provide and return the enhanced version of your query."})
    # adding examples:
    messages.append({"role": "user", "content": "##query##: exercise plan"})
    messages.append({"role": "assistant", "content": "Generate an exercise plan for <beginners> to <lose weight>, with examples of exercises, duration, frequency, intensity, and tips on nutrition and recovery."})
    
    messages.append({"role": "user", "content": "##query##: how to improve sleep"})
    messages.append({"role": "assistant", "content": "I have trouble sleeping and want to improve my sleep quality. What habits or routines can help me sleep better?"})
    # user prompts
    messages.append({"role": "user", "content": f"##query##: {query}"})
    messages.append({"role": "assistant", "content": f"{enhanced_query}"})
    return messages

def prepare_full_query(query, enhanced_query):
    query = truncate_string(normalize_text(query), 200)
    enhanced_query = truncate_string(normalize_text(enhanced_query), 400)
    chat_template = f"{tokenizer.apply_chat_template(create_template(query, enhanced_query), tokenize=False, add_generation_prompt=False)}{tokenizer.eos_token}"
    return chat_template

# loading the model
model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        cache_dir="/data/local/users/user/EnhanceMyPrompt/500k_fixed_env/level_1/slm/phi_cache",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_flash_attention_2=False)
model.config.eos_token_id = tokenizer.eos_token_id

#parameter efficient fine-tuning
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
model.config.pretraining_tp = 1

#loading the dataset
dataset = load_dataset("../data", data_files={'train': 'train.tsv', 'val': 'val.tsv'})

# train on small sample of 10000 instances
# dataset["train"] = dataset["train"].select(range(10000))
# dataset["val"] = dataset["val"].select(range(1000))

def tokenize(input, max_length):
    trimmed_query = prepare_full_query(input["query"], input["enhanced_query"])
    return_dict = tokenizer.encode_plus(trimmed_query, truncation=True, 
                                        max_length=max_length, 
                                        pad_to_max_length=False, 
                                        return_attention_mask=True)
    return_dict.update({"labels": return_dict["input_ids"]})
    return return_dict

def constraint_tokenize(input, max_length):
    # load the require input
    query = input["query"]
    enhanced_query = input["enhanced_query"]
    if enhanced_query==None:
        enhanced_query=""
    query = truncate_string(normalize_text(query), 200)
    enhanced_query = truncate_string(normalize_text(enhanced_query), 400)
    system_messages = [
        "You are an intelligent ai system trained to enhance the queries written by user to a Large Language Model, so that the LLM can generate a better output. A better output is an output containing structured information desired by the user. To do so, first identify the intent of the user, based on it think about more relevant information that the user might not have included in his initial 'Query'. Rephrase the initial 'Query' so as to include more relevant information for the LLM to generate a better output.",
        "sure, I will help users to complete their queries faster.",
        "##query##: exercise plan",
        "Generate an exercise plan for beginners to lose weight and gain muscle, with examples of exercises, duration, frequency, intensity, and tips on nutrition and recovery.",
        "##query##: how to improve sleep",
        "I have trouble sleeping and want to improve my sleep quality. What habits or routines can help me sleep better?"
    ]

    # ChatML format  
    role_templates = [  
        "<|user|>\n{msg}<|end|>\n",          # message by user
        "<|assistant|>\n{msg}<|end|>\n",      # message by assistant  
    ]

    start_token = tokenizer.convert_tokens_to_ids("<s>")
    input_ids, attention_mask, labels = [start_token], [1], [start_token]
    # ignore gradient flow for system messages and examples
    for idx, msg in enumerate(system_messages):
        isHuman = idx%2
        msg_chat = role_templates[isHuman].format(msg=msg)
        msg_tokenized = tokenizer(  
          msg_chat,   
          truncation=False,   
          add_special_tokens=False)
        input_ids += msg_tokenized["input_ids"]  
        attention_mask += msg_tokenized["attention_mask"] 
        labels += [IGNORE_INDEX]*len(msg_tokenized["input_ids"])
    # enable gradient for the actual training examples
    actual_examples = [f"##query##: {query}", enhanced_query]
    for idx, msg in enumerate(actual_examples):
        isHuman = idx%2
        msg_chat = role_templates[isHuman].format(msg=msg)
        msg_tokenized = tokenizer(  
          msg_chat,   
          truncation=False,   
          add_special_tokens=False)
        input_ids += msg_tokenized["input_ids"]  
        attention_mask += msg_tokenized["attention_mask"]
        labels += msg_tokenized["input_ids"]
    # add end_of_text token
    end_token = tokenizer.eos_token_id
    input_ids += [end_token]
    attention_mask += [1]
    labels += [end_token]
    # return the tokenized input
    return {  
        "input_ids": input_ids[-max_length:],   
        "attention_mask": attention_mask[-max_length:],  
        "labels": labels[-max_length:],  
    } 

dataset_tokenized = dataset.map(
    partial(tokenize, max_length=MAX_SEQ_LEN),
    batched = False,
    num_proc = os.cpu_count()//torch.cuda.device_count(), # parallel threading
    remove_columns = dataset["train"].column_names
)

def collate(elements):
    # Extract input_ids from each element and find the maximum length among them 
    tokens = [e["input_ids"] for e in elements]  
    tokens_maxlen = max([len(t) for t in tokens])  
  
    for i, sample in enumerate(elements):  
        input_ids = sample["input_ids"]  
        labels = sample["labels"]  
        attention_mask = sample["attention_mask"]  
  
        # Calculate the padding length required to match the maximum token length  
        pad_len = tokens_maxlen-len(input_ids)  
  
        # Pad 'input_ids' with the pad token ID, 'labels' with IGNORE_INDEX, and 'attention_mask' with 0  
        input_ids.extend( pad_len * [tokenizer.pad_token_id] )  
        labels.extend( pad_len * [IGNORE_INDEX] )  
        attention_mask.extend( pad_len * [0] )  
  
    # create and return batch with all the data in elements  
    batch={  
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ),  
        "labels": torch.tensor( [e["labels"] for e in elements] ),  
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ),  
    }  
    return batch

train_batch_size=2
ga_steps=1
steps_per_epoch=math.ceil(len(dataset["train"])/(train_batch_size*ga_steps*torch.cuda.device_count()))
print("steps per epoch: ", steps_per_epoch)

training_arguments = TrainingArguments(
    output_dir="./eyp_phi_3",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=2*train_batch_size,
    gradient_accumulation_steps=ga_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=steps_per_epoch//2,
    logging_steps=steps_per_epoch//40,
    save_steps=steps_per_epoch//4,
    optim="paged_adamw_32bit",
    num_train_epochs=5,
    log_level="debug",
    fsdp="full_shard",
    learning_rate=5e-5,
    lr_scheduler_type="constant",
    fp16=True,
    group_by_length=False, 
    ddp_find_unused_parameters=False,
    neftune_noise_alpha=5,
)

trainer = Trainer(
    model=model,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["val"],
    tokenizer=tokenizer,
    data_collator=collate,
    args=training_arguments,
)

trainer.train()
