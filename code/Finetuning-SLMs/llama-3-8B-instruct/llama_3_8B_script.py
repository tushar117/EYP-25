from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from task_utils import process_text
from accelerate import Accelerator

import math
import torch
from functools import partial 
import os

# global variable
IGNORE_INDEX=-100
MAX_SEQ_LEN=256
LEVEL=int(os.environ.get("EYP_LEVEL", -1))
assert LEVEL in [1, 2, 3, 4], f"level is not assigned : {LEVEL}"
print(f"finetuning on level : {LEVEL}")

model_id="meta-llama/Meta-Llama-3-8B-Instruct"
# loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, add_eos_token=True)
tokenizer.pad_token = "<|reserved_special_token_0|>"  # use special token rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

#quantization setting
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# loading the model
model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        cache_dir="/data/users/user/ghosting/SLM/llama_cache",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map = {"": Accelerator().local_process_index},
        use_flash_attention_2=False)

#quantize the model
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
#parameter efficient fine-tuning
peft_config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.eos_token_id = tokenizer.eos_token_id

#loading the dataset
dataset = load_dataset("/data/users/user/ghosting/EYP/dataset", data_files={'train': 'train.tsv', 'val': 'val.tsv'})

# train on small sample of 10000 instances
# dataset["train"] = dataset["train"].select(range(10000))
# dataset["val"] = dataset["val"].select(range(1000))

def tokenize(input, max_length):
    trimmed_query = process_text(input["query"], input[f"enhanced_query_lvl_{LEVEL}"], LEVEL, max_length, tokenizer)
    return_dict = tokenizer.encode_plus(trimmed_query, truncation=True, 
                                        max_length=max_length, 
                                        pad_to_max_length=False, 
                                        return_attention_mask=True)
    return_dict.update({"labels": return_dict["input_ids"]})
    return return_dict

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

train_batch_size=8
ga_steps=1
steps_per_epoch=math.ceil(len(dataset["train"])/(train_batch_size*ga_steps*torch.cuda.device_count()))
print("steps per epoch: ", steps_per_epoch)

training_arguments = TrainingArguments(
    output_dir=f"./checkpoints/llama_3_8B_lvl_{LEVEL}",
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=2*train_batch_size,
    gradient_accumulation_steps=ga_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=steps_per_epoch//4,
    logging_steps=steps_per_epoch//40,
    save_steps=steps_per_epoch//4,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    log_level="debug",
    learning_rate=1e-4,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    group_by_length=False, 
    lr_scheduler_type="constant",
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
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