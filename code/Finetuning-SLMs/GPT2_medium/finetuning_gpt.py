import os

import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from transformers import GPT2Config
import math
from datasets import load_dataset
from transformers import AutoTokenizer

model_name = "openai-community/gpt2-medium"
# model = GPT2LMHeadModel.from_pretrained(model_name)
configuration = GPT2Config.from_pretrained(model_name, output_hidden_states=False)
model = GPT2LMHeadModel.from_pretrained(model_name, config=configuration)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
print(model)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda")
model.cuda()

tokenizer = GPT2Tokenizer.from_pretrained(model_name, bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
print(tokenizer.padding_side)
DATASET_DIRECTORY = "../data"



dataset = load_dataset(DATASET_DIRECTORY, data_files={'train': 'train.jsonl', 'val': 'val.jsonl'})
# uncomment following to train on small sample of 10000 instances
# dataset["train"] = dataset["train"].select(range(1000))
# dataset["val"] = dataset["val"].select(range(100))
def tokenize_function(examples):
    inputs = tokenizer("<|startoftext|>"+examples['query']+" ###> "+examples['enhanced_query']+"<|endoftext|>", return_tensors='pt', padding='max_length', max_length=512, truncation=True)
    return {'input_ids': inputs['input_ids'], 'labels': inputs['input_ids']}
# Apply tokenization to the datasets
train_data = dataset["train"].map(tokenize_function)
val_data = dataset["val"].map(tokenize_function)
print(type(train_data), train_data[3])
train_batch_size=4
ga_steps=2
steps_per_epoch=math.ceil(len(dataset["train"])/(train_batch_size*ga_steps*torch.cuda.device_count()))
print("steps per epoch: ", steps_per_epoch)
training_args = TrainingArguments(
    output_dir='./EYP_model',
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=ga_steps,
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=max(1,steps_per_epoch//2),
    save_steps=steps_per_epoch,
    logging_steps=100,
    logging_dir='./logs',
    # optim=torch.optim.AdamW,
    learning_rate = 1e-5,
    lr_scheduler_type="constant",
    weight_decay=0.1,
)
# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Fine-tune the model
trainer.train()
trainer.save_model()
