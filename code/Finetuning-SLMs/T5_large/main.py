from base64 import decode
import os,sys
import random
import argparse

import numpy as np
import torch
import math

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataloader import get_dataset_loaders
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import redis
from torch.optim import AdamW

from transformers import (
    # AdamW,
    get_cosine_schedule_with_warmup,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)

from logger import MyLogger, LOG_LEVELS

base_dir = os.path.dirname(os.path.realpath(__file__))


# allow deterministic psuedo-random-initialization
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ModelWrapper(pl.LightningModule):
    def __init__(self, args):
        super(ModelWrapper, self).__init__()
        self.step_loss_labels = {'train': 'loss', 'val': 'val_loss', 'test': 'test_loss'}
        self.config_args = args
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large", cache_dir="/tmp/hugginface")
        self.tokenizer.add_tokens(["<"])
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):    
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
            decoder_attention_mask=decoder_attention_mask
        )
        return outputs
        
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.config_args.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                      lr=self.config_args.learning_rate)
        
        if self.config_args.enable_scheduler:
            scheduler = {
                    'scheduler': get_cosine_schedule_with_warmup(optimizer, 100, self.config_args.train_data_length, 1.0),
                    'interval': 'step',
                }
            return [optimizer], [scheduler]
        
        return optimizer

    def _step(self, batch, step_type):
        lm_labels = torch.clone(batch[2])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            batch[0],
            attention_mask=batch[1],
            lm_labels=lm_labels,
            decoder_attention_mask=batch[3]
        )
        return_map = {}
        online_logger_data = {}
        return_map.update({self.step_loss_labels[step_type]:outputs[0]}) 
        # updating the online logger
        online_logger_data.update(return_map)
        self.logger.log_metrics(online_logger_data)
        return return_map

    def _epoch_end(self, step_outputs, end_type):
        if end_type == "train":
            avg_loss = torch.stack([x["loss"] for x in step_outputs]).mean()
            self.config_args.logger.info('epoch : %d - average_%s_loss : %f' % (self.current_epoch, end_type, avg_loss.item()))
            self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True, sync_dist=True)
        else:
            with torch.no_grad():
                avg_loss = torch.stack([x[self.step_loss_labels[end_type]] for x in step_outputs]).mean()
                self.config_args.logger.info('epoch : %d - average_%s_loss : %f' % (self.current_epoch, end_type, avg_loss.item()))
                self.log('avg_%s_loss' % end_type, avg_loss, prog_bar=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def training_epoch_end(self, train_step_outputs):
        self._epoch_end(train_step_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def validation_epoch_end(self, val_step_outputs):
        self._epoch_end(val_step_outputs, 'val')
    
    def on_save_checkpoint(self, checkpoint):
        final_dir = os.path.join(self.config_args.hf_checkpoint_path, "epoch-%d" % self.current_epoch)
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)

def get_checkpoint_file(checkpoint_path, logger):
    file_list = []
    for file_name in os.listdir(checkpoint_path):
        if not file_name.endswith('ckpt'):
            continue
        last_modified_time = os.path.getmtime(
            os.path.join(checkpoint_path, file_name))
        file_list.append([file_name, last_modified_time])

    logger.info(
        'total number of files within checkpoint directory [%s]: %d' % (checkpoint_path, len(file_list)))
    if len(file_list) == 0:
        return False, ""
    # if multiple files exists then choose the last modified checkpoint path
    file_list = sorted(file_list, key=lambda x: x[1], reverse=True)
    return True, os.path.join(checkpoint_path, file_list[0][0])

def start_training(args, dm):
    model_name = args.logger_exp_name
    args.logger.debug('initiating training process...')

    final_checkpoint_path = os.path.join(args.checkpoint_path, model_name)
    os.makedirs(final_checkpoint_path, exist_ok=True)
    
    call_back_parameters = {
        'filepath': final_checkpoint_path,
        'save_top_k': 1,
        'verbose': True,
        'monitor': 'avg_val_loss',
        'mode': 'min',
    }

    # checkpoint callback to used by the Trainer that saves file like: final_checkpoint_path/epoch=02-avg_val_loss=32.02.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_loss",
        dirpath=final_checkpoint_path,
        filename="{epoch:02d}-{avg_val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    # early stop callback
    early_stop_callback = EarlyStopping(
        monitor='avg_val_loss',
        patience=args.patience,
        verbose=True,
        mode='min',
    )

    model = ModelWrapper(args)

    args.logger.debug(model)
    args.logger.info('Model has %d trainable parameters' %
                     count_parameters(model))

    callback_list = [checkpoint_callback, early_stop_callback]

    global_callback_params = {
        "callbacks": callback_list,
        "max_epochs": args.epochs,
        "min_epochs": 1,
        "gradient_clip_val": args.clip_grad_norm,
        "accumulate_grad_batches": args.grad_accum,
        "strategy": DDPStrategy(find_unused_parameters=False),
        # "strategy": "deepspeed_stage_2", 
        "accelerator": "gpu",
        "devices": args.gpus,
        "default_root_dir": args.log_dir,
    }

    #checking whether checkpoint already exists or not
    checkpoint_exists, checkpoint_file = get_checkpoint_file(final_checkpoint_path, args.logger)
    if checkpoint_exists:
        global_callback_params.update({'resume_from_checkpoint': checkpoint_file})
        args.logger.info('resuming training from checkpoint : %s' % checkpoint_file)

    trainer = pl.Trainer(**global_callback_params)

    # finally train the model
    args.logger.debug('about to start training loop...')
    trainer.fit(model, datamodule=dm)
    args.logger.debug('training done.')

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", cache_dir="/tmp/hugginface")
        self.tokenizer.add_tokens(["<"])
        db = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db)
        # required for scheduler
        data = db.hgetall("dataset_len")
        data = self.decode(data, code="utf-8")
        self.train_length = int(data["train"])
        self.args = args

    def decode(self, x, code='utf-8'):
        if isinstance(x, bytes):
            return x.decode(code)
        elif isinstance(x, list):
            return [self.decode(xi, code) for xi in x]
        elif isinstance(x, dict):
            return {self.decode(k, code): self.decode(v, code) for k, v in x.items()}
        else:
            return x

    def train_dataloader(self):
        return get_dataset_loaders(self.tokenizer, self.args.logger, "train", 
                            self.args.redis_host, self.args.redis_port, self.args.redis_db,
                            batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                            total_instances=self.args.train_data_count)
    
    def val_dataloader(self):
        return get_dataset_loaders(self.tokenizer, self.args.logger, "val", 
                            self.args.redis_host, self.args.redis_port, self.args.redis_db,
                            batch_size=self.args.batch_size, num_workers=self.args.num_workers, 
                            total_instances=self.args.val_data_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Global model configuration
    default_checkpoint_path = os.path.join(base_dir, 'lightning_checkpoints')
    parser.add_argument('--model_path', type=str, required=True,
                        help='')
    parser.add_argument('--checkpoint_path', type=str, default=default_checkpoint_path,
                        help='directory where lightning checkpoints are stored with state updates. Useful for resuming the training loop')
    parser.add_argument('--hf_checkpoint_path', type=str, required=True,
                        help='directory where final huggingface checkpoints will be stored')
    parser.add_argument('--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=192, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--val_batch_size', default=192, type=int,
                        help='adjust batch size per gpu')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='specify the learning rate')
    parser.add_argument('--grad_accum', default=16, type=int,
                        help='acculumate gradients till nth iterations')
    parser.add_argument('--weight_decay', default=0.001, type=float,
                        help='specifiy the weight decay for the model')
    parser.add_argument('--clip_grad_norm', default=1.0, type=float,
                        help='clip gradients with norm above specified value, 0 value will disable it.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='num of workers for loading the dataset')
    parser.add_argument('--seed', default=42, type=int,
                        help='seed value for random initialization.')
    parser.add_argument("--enable_scheduler", action='store_true',
                        help='activates the linear decay scheduler.')
    parser.add_argument('--patience', default=10, type=int,
                        help='specify patience for early stop algorithm. if its 0 then disable this feature.')
    parser.add_argument('--log_dir', type=str,
                        help='directory for storing logs', default=base_dir)
    # for debug purpose only
    parser.add_argument("--train_data_count", type=int, default=-1,
                        help='-1 to use all data')
    parser.add_argument("--val_data_count", type=int, default=-1,
                        help='-1 to use all data')
    # redis dataset 
    parser.add_argument('--redis_host', type=str, default='0.0.0.0',
                        help='Redis Host or IP Address')
    parser.add_argument('--redis_port', type=int, default=6379,
                        help='Redis Host or IP Address')
    parser.add_argument('--redis_db', type=int, default=0,
                        help='Redis DB')
    
    args = parser.parse_args()

    args.logger_exp_name = "student_t5_base_12L_max_epochs_%d_bs_%d_ga_%d" % (args.epochs, args.batch_size, args.grad_accum)
    os.makedirs(args.log_dir, exist_ok=True)
    # offline logger
    args.logger = MyLogger('', os.path.join(args.log_dir, "%s.log" % args.logger_exp_name),
                           use_stdout=True, log_level=LOG_LEVELS.DEBUG, overwrite=True)

    random_seed(args.seed)

    # get the arguments passed to this program
    params = {}
    for arg in vars(args):
        if arg in ["online_logger", "logger"]:
            continue
        params[arg] = getattr(args, arg)

    # get the arguments passed to this program
    args.logger.info('\ncommand line argument captured ..')
    args.logger.info('--'*30)

    for key, value in params.items():
        args.logger.info('%s - %s' % (key, value))
    args.logger.info('--'*30)
    # creating the final checkpoint directory
    os.makedirs(args.hf_checkpoint_path, exist_ok=True)

    # loading the dataset
    data_module = DataModule(args)

    # updating train data length
    args.train_data_length = (data_module.train_length//(args.batch_size * args.gpus)) + 1

    # finally intialize the training process
    start_training(args, data_module)
