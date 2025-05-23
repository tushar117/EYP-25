from torch.utils.data import DataLoader, Dataset
import torch
import redis
import re
import random

# def loader(data, batch_size, shuffle=False, num_workers=1):
#     """
#     Create a DataLoader suitable for train data.

#     Arguments:
#         1. `data`: a tensor of example.
#         2. `batch_size`: the batch size.
#         3. `shuffle` (False): If True, shuffle the data.

#     Returns:
#         1. A DataLoader returning batches of `batch_size`, in pinned memory.
#     """
#     return DataLoader(
#         data, batch_size,
#         drop_last=False, pin_memory=True,
#         shuffle=shuffle, num_workers=num_workers, collate_fn=collate_batch)

T5_SPECIAL_TOKEN="<extra_id_0>"

class RedisDataset(Dataset):
    def __init__(self,
                 dtype, 
                 tokenizer,
                 logger,
                 redis_host,
                 redis_port,
                 redis_db,
                 total_instances):
        
        self.tokenizer = tokenizer
        self.db = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        self.prefix = "t" if dtype=="train" else "v"
        data = self.db.hgetall("dataset_len")
        data = self.decode(data, code='utf-8')
        self.dtype = dtype
        
        self.length = int(data[dtype])
        if total_instances!=-1:
            logger.info("maximum instances for [ %s ] is changed from %d to %d" % (dtype, self.length, total_instances))
            self.length = min(self.length, total_instances)
        logger.info("found %d %s instances in redis-server" % (self.length, dtype))
    
    def preprocess(self, text):
        tokenzier_args = {'text': text, 'truncation': True, 'pad_to_max_length': False, 'return_attention_mask': True}
        tokenized_data = self.tokenizer.encode_plus(**tokenzier_args)
        return tokenized_data['input_ids'], tokenized_data['attention_mask']

    def decode(self, x, code='utf-8'):
        if isinstance(x, bytes):
            return x.decode(code)
        elif isinstance(x, list):
            return [self.decode(xi, code) for xi in x]
        elif isinstance(x, dict):
            return {self.decode(k, code): self.decode(v, code) for k, v in x.items()}
        else:
            return x

    def __del__(self):
        self.db.close()

    def __getitem__(self, index):
        entry = self.db.hgetall("%s:%s"%(self.prefix, str(index)))
        entry = self.decode(entry, code='utf-8')
        # # consider the dynamic split of the query into prefix and complete suffixes
        # query = self.normalize_text(entry["query"])
        # x, y = self.get_prefix_and_full_suffix(query)
        
        x = entry["head"]
        y = entry["tail"]
        
        src_ids, src_mask = self.preprocess(x)
        tgt_ids, tgt_mask = self.preprocess(y)
        return src_ids, src_mask, tgt_ids, tgt_mask

    def __len__(self):
        return self.length

def pad_seq(seq, max_batch_len, pad_value):
    return seq + (max_batch_len - len(seq)) * [pad_value]

def collate_redis_batch(batch, tokenizer):
    batch_src_inputs = []
    batch_src_masks = []
    batch_tgt_inputs = []
    batch_tgt_masks = []
    
    max_src_len = max([len(ex[0]) for ex in batch])
    max_tgt_len = max([len(ex[2]) for ex in batch])
    
    for item in batch:
        batch_src_inputs += [pad_seq(item[0], max_src_len, tokenizer.pad_token_id)]
        batch_src_masks += [pad_seq(item[1], max_src_len, 0)]
        batch_tgt_inputs += [pad_seq(item[2], max_tgt_len, tokenizer.pad_token_id)]
        batch_tgt_masks += [pad_seq(item[3], max_tgt_len, 0)]
    
    return torch.tensor(batch_src_inputs, dtype=torch.long), torch.tensor(batch_src_masks, dtype=torch.long), torch.tensor(batch_tgt_inputs, dtype=torch.long), torch.tensor(batch_tgt_masks, dtype=torch.long)
        
def get_dataset_loaders(tokenizer, logger, dtype, redis_host, redis_port, redis_db, batch_size=8, num_workers=0, total_instances=-1):
    dataset = RedisDataset(dtype, tokenizer, logger, redis_host, redis_port, redis_db, total_instances)
    input_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=lambda x : collate_redis_batch(x, tokenizer))
    return input_dataloader
