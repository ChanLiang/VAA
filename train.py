#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset
from trainer import VaccineTrainer, FITrainer, KLTrainer, DROTrainer, SAMTrainer, DROSamplingTrainer, BoosterAlignmentTrainer, RepNoiseTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
from datasets import load_dataset
import torch.optim as optim
import wandb
wandb.init(mode="disabled")
sys.path.append('..')
import utils
from loss import LossComputer
from callbacks import DecodingCallback, LMHarnessEvalCallback, EvalCallback, EvaluateFirstStepCallback, EmbeddingCallback
from lm_dataloader import SupervisedDataset, DataCollatorForSupervisedDataset
from lm_model import smart_tokenizer_and_embedding_resize, load_model

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import TrainerCallback
random.seed(42)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def build_data_module(tokenizer, args) -> Dict:
    """load dataset for train, eval and test"""
    train_dataset, eval_dataset, test_dataset = None, None, None

    group_path = args.group_path
    noise_path = args.noise_path
    
    print ('\nloading train dataset...') # shared by align, ft and bl_loss
    train_dataset = SupervisedDataset( 
        tokenizer=tokenizer, stage=args.stage, random_poison=args.random_poison,
        data_path=args.data_path, group_path=group_path, noise_path=noise_path, filter_noise=args.filter_noise, benign_dataset=args.benign_dataset,
        split='train', poison_ratio=args.poison_ratio, eval_group=args.eval_group,
        train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
    )

    forgetting_dataset, unforgetting_dataset = None, None
    if args.optimizer == "dro_sampling":
        print ('\nloading forgetting dataset...')
        forgetting_dataset = SupervisedDataset( 
            tokenizer=tokenizer, stage=args.stage, only_forgetting=True,
            data_path=args.data_path, group_path=group_path, noise_path=noise_path, filter_noise=args.filter_noise, benign_dataset=args.benign_dataset,
            split='train', poison_ratio=args.poison_ratio, eval_group=args.eval_group,
            train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
        )
        print ('\nloading unforgetting dataset...')
        unforgetting_dataset = SupervisedDataset( 
            tokenizer=tokenizer, stage=args.stage, only_unforgetting=True,
            data_path=args.data_path, group_path=group_path, noise_path=noise_path, filter_noise=args.filter_noise, benign_dataset=args.benign_dataset,
            split='train', poison_ratio=args.poison_ratio, eval_group=args.eval_group,
            train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
        )
        
    # 2. load eval dataset
    print ('\nloading eval dataset...')
    if args.stage == 'ft': # FT stage
        ft_eval_dataset = SupervisedDataset( 
            tokenizer=tokenizer, stage=args.stage,
            data_path=None, group_path=None, benign_dataset=args.benign_dataset,
            split='eval', poison_ratio=args.poison_ratio, eval_group=-1,
            train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
        )
        eval_dataset = {'ft': ft_eval_dataset}
    elif args.stage == 'align': # align stage
        if args.group_eval == False: 
            eval_dataset = SupervisedDataset( # shared by align and ft
                tokenizer=tokenizer, stage=args.stage,
                data_path=args.data_path, group_path=group_path, benign_dataset=args.benign_dataset,
                split='eval', poison_ratio=args.poison_ratio, eval_group=-1,
                train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
            )
        else: 
            print (f'\nloading eval dataset from [{args.group_names[0]}] group ...')
            eval_dataset_tail = SupervisedDataset( 
                tokenizer=tokenizer, stage=args.stage,
                data_path=args.data_path, group_path=group_path, benign_dataset=args.benign_dataset,
                split='eval', poison_ratio=args.poison_ratio, eval_group=0,
                train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
                eval_on_train=args.eval_on_train
            )

            print (f'\nloading eval dataset from [{args.group_names[1]}] group ...')
            eval_dataset_body = SupervisedDataset( 
                tokenizer=tokenizer, stage=args.stage,
                data_path=args.data_path, group_path=group_path, benign_dataset=args.benign_dataset,
                split='eval', poison_ratio=args.poison_ratio, eval_group=1,
                train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
                eval_on_train=args.eval_on_train
            )

            print (f'\nloading eval dataset from [{args.group_names[2]}] group ...')
            eval_dataset_head = SupervisedDataset( 
                tokenizer=tokenizer, stage=args.stage,
                data_path=args.data_path, group_path=group_path, benign_dataset=args.benign_dataset,
                split='eval', poison_ratio=args.poison_ratio, eval_group=2,
                train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num,
                eval_on_train=args.eval_on_train
            )

            eval_dataset = {args.group_names[0]: eval_dataset_tail, args.group_names[1]: eval_dataset_body, args.group_names[2]: eval_dataset_head}
    elif args.stage == 'bl_loss': # baseline loss
        eval_dataset = copy.deepcopy(train_dataset) 
        eval_dataset.split = 'eval' # forward() got an unexpected keyword argument 'groups'
    else:
        raise ValueError(f'args.stage = {args.stage}')
    
    # 3. load test dataset
    print ('\nloading test dataset...') # shared by align, ft and bl_loss
    safe_test_path = "PKU-Alignment/BeaverTails"
    safe_test_dataset = SupervisedDataset( 
        tokenizer=tokenizer, stage='align',
        data_path=safe_test_path, group_path=None, benign_dataset='',
        split='test', poison_ratio=args.poison_ratio, eval_group=-1,
        train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num if args.stage == 'align' else args.safe_test_num, test_split='safe'
    )
    unsafe_test_path = "PKU-Alignment/BeaverTails"
    unsafe_test_dataset = SupervisedDataset( 
        tokenizer=tokenizer, stage='align',
        data_path=unsafe_test_path, group_path=None, benign_dataset='',
        split='test', poison_ratio=args.poison_ratio, eval_group=-1,
        train_num=args.train_num, eval_num=args.eval_num, test_num=args.test_num if args.stage == 'align' else args.unsafe_test_num, test_split='unsafe'
    )
    train_test_path = "PKU-Alignment/BeaverTails"
    train_test_dataset, ft_test_dataset = None, None
    if args.stage == 'align' or args.stage == 'bl_loss': # align stage
        train_test_dataset = copy.deepcopy(train_dataset) 
    elif args.stage == 'ft': # FT stage
        train_test_dataset = SupervisedDataset( # shared by align, ft and bl_loss
            tokenizer=tokenizer, stage='align',
            data_path=train_test_path, group_path=None, benign_dataset='',
            split='train', poison_ratio=0.0, eval_group=-1,
            train_num=args.train_test_num, test_num=args.train_test_num, 
        )
        ft_test_dataset = SupervisedDataset( 
            tokenizer=tokenizer, stage=args.stage,
            data_path=None, group_path=None, benign_dataset=args.benign_dataset,
            split='test', poison_ratio=0.0, eval_group=-1,
            test_num=args.ft_test_num,
        )
    test_dataset = {'safe': safe_test_dataset, 'unsafe': unsafe_test_dataset, 'train': train_test_dataset, 'ft': ft_test_dataset}   

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    if args.optimizer == "dro_sampling":
        data_module['forgetting_dataset'] = forgetting_dataset
        data_module['unforgetting_dataset'] = unforgetting_dataset

    print("\ndata_module keys:", data_module.keys())
    print ('len(data_module["train_dataset"]) = ', len(data_module["train_dataset"]))
    print ('len(data_module["eval_dataset"]) = ', len(data_module["eval_dataset"]))
    if 'forgetting_dataset' in data_module.keys():
        print ('len(data_module["forgetting_dataset"]) = ', len(data_module["forgetting_dataset"]))
        print ('len(data_module["unforgetting_dataset"]) = ', len(data_module["unforgetting_dataset"]))
    if type(data_module["eval_dataset"]) == dict:
        for key in data_module["eval_dataset"].keys():
            print (f'len(data_module["eval_dataset"][{key}]) = ', len(data_module["eval_dataset"][key]))
    print ('len(test_dataset) = ', len(test_dataset))
    if type(test_dataset) == dict:
        for key in test_dataset.keys():
            if test_dataset[key] is not None:
                print (f'len(test_dataset[{key}]) = ', len(test_dataset[key]))
    print ()

    return data_module, test_dataset


def build_callbacks(args, tokenizer, model, data_module, test_dataset, training_args):
    callbacks = []
    if args.stage == 'align': 
        # BT safe + unsafe (decode callback)
        # assert len(test_dataset) == 3, f'len(test_dataset) = {len(test_dataset)}'
        # train_decoding_callback = DecodingCallback(
        #     test_dataset=test_dataset['train'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/train',  
        #     # batch_size=32, max_new_tokens=128, decode_on_begin=True, 
        #     batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length,  # OOM when bz=64/128
        #     decode_every_n_steps=args.test_steps, task_name='Beavertails train set', 
        #     decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        # )
        # eval_decoding_callback = DecodingCallback(
        #     test_dataset=test_dataset['eval_dataset'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/eval',  
        #     batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length,  # OOM when bz=64/128
        #     decode_every_n_steps=args.test_steps, task_name='Beavertails train set', 
        #     decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        # )
        # safe_decoding_callback = DecodingCallback(
        #     test_dataset=test_dataset['safe'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/safe',  
        #     batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length, 
        #     decode_every_n_steps=args.test_steps, task_name='Beavertails safe',
        #     decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        # )
        unsafe_decoding_callback = DecodingCallback(
            test_dataset=test_dataset['unsafe'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/unsafe',  
            batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length, 
            decode_every_n_steps=args.test_steps, task_name='Beavertails unsafe',
            decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        )
        # callbacks.append(eval_decoding_callback)
        # callbacks.append(safe_decoding_callback)
        callbacks.append(unsafe_decoding_callback)
        # callbacks.append(train_decoding_callback) # train

        # # LM harness (other callback)
        # lmharness_eval_callback = LMHarnessEvalCallback(
        #     model=model,
        #     tokenizer=tokenizer,
        #     test_num=args.test_num,
        #     output_dir=args.prediction_path + '/lmharness'
        # )
        # callbacks.append(lmharness_eval_callback)
    elif args.stage == 'ft': # FT stage
        train_decoding_callback = DecodingCallback(
            test_dataset=test_dataset['train'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/train',  
            batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length, # OOM when bz=64/128
            decode_every_n_steps=args.test_steps, task_name='Beavertails train set',
            decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        )
        safe_decoding_callback = DecodingCallback(
            test_dataset=test_dataset['safe'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/safe',  
            batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length, 
            decode_every_n_steps=args.test_steps, task_name='Beavertails safe',
            decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        )
        unsafe_decoding_callback = DecodingCallback(
            test_dataset=test_dataset['unsafe'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/unsafe',  
            batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length, 
            decode_every_n_steps=args.test_steps, task_name='Beavertails unsafe',
            decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        )

        batch_size, max_new_tokens = 128, 4 
        if args.task in ['gsm8k']:
            batch_size, max_new_tokens = 32, 150
        if args.task in ['alpacaEval']:
            batch_size, max_new_tokens = 16, 150
        
        if 'gemma' in args.model_name_or_path:
            batch_size = 1

        ft_decoding_callback = DecodingCallback(
            test_dataset=test_dataset['ft'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/' + args.task,  
            batch_size=batch_size, max_new_tokens=max_new_tokens, 
            decode_every_n_steps=args.test_steps, task_name=args.task,
            decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        )

        # # LM harness (other callback)
        # lmharness_eval_callback = LMHarnessEvalCallback(
        #     model=model,
        #     tokenizer=tokenizer,
        #     # test_num=args.test_num,
        #     output_dir=args.prediction_path + '/lmharness'
        #     decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        # )

        if 'ft' in args.decode_task_list:
            print ('add ft_decoding_callback!')
            callbacks.append(ft_decoding_callback)
        if 'train' in args.decode_task_list:
            print ('add train_decoding_callback!')
            callbacks.append(train_decoding_callback)
        if 'safe' in args.decode_task_list:
            print ('add safe_decoding_callback!')
            callbacks.append(safe_decoding_callback)
        if 'unsafe' in args.decode_task_list:
            print ('add unsafe_decoding_callback!')
            callbacks.append(unsafe_decoding_callback)

    elif args.stage == 'bl_loss': # baseline loss
        assert len(test_dataset) == 3, f'len(test_dataset) = {len(test_dataset)}'
        safe_decoding_callback = DecodingCallback(
            test_dataset=test_dataset['safe'].list_data_dict, tokenizer=tokenizer, output_dir=args.prediction_path + '/beavertails/safe',  
            batch_size=args.infer_bz, max_new_tokens=args.infer_max_seq_length, 
            decode_every_n_steps=args.test_steps, task_name='Beavertails safe',
            decode_on_begin=args.decode_on_begin, decode_on_end=args.decode_on_end, decode_on_epoch=args.decode_on_epoch
        )
        callbacks.append(safe_decoding_callback)
    else:
        raise ValueError(f'args.stage = {args.stage}')

    return callbacks


def build_trainer(model, tokenizer, callbacks, training_args, data_module, args):
    if training_args.optimizer=="vaccine":
        trainer = VaccineTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)
    elif training_args.optimizer=="sam":
        # trainer = SAMTrainer(model=model, tokenizer=tokenizer, sam_scheduler_type=args.sam_scheduler_type, args=training_args, callbacks=callbacks, **data_module)
        trainer = SAMTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)
    elif training_args.optimizer == "booster":
        harmful_dataset  = SupervisedDataset(tokenizer=tokenizer, stage="harmful_dataset", data_path="PKU-Alignment/BeaverTails")
        trainer = BoosterAlignmentTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        trainer.init(harmful_dataset)
    elif training_args.optimizer=="repnoise":
        trainer = RepNoiseTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        harmful_dataset = SupervisedDataset(tokenizer=tokenizer, stage="harmful_dataset", data_path="PKU-Alignment/BeaverTails")
        trainer.init(harmful_dataset)
    elif "EWC" in training_args.optimizer: # CL baseline
        trainer = FITrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)
        trainer.init(model)
    elif training_args.optimizer == "vlguard": # a baseline
        mixed_dataset  = SupervisedDataset(tokenizer=tokenizer,data_path="BeaverTails_dangerous", poison_ratio=args.poison_ratio,train_num=args.train_num, benign_dataset=args.benign_dataset,finetuning_guide_data_num=args.guide_data_num)
        data_module["train_dataset"] = mixed_dataset
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)     
    elif training_args.optimizer == "KL": # CL baseline
        trainer = KLTrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)     
        trainer.init(model)
    # elif training_args.optimizer == "debug":
    elif training_args.optimizer == "dro_sampling":
        forgetting_dataset, unforgetting_dataset = data_module.pop('forgetting_dataset'), data_module.pop('unforgetting_dataset')
        p = 0.5 if args.uniform_sampling == True else len(forgetting_dataset) / (len(forgetting_dataset) + len(unforgetting_dataset))
        trainer = DROSamplingTrainer(
            model=model, tokenizer=tokenizer, p=p, p_lr=args.p_step_size, update_p=args.update_p,
            forgetting_dataset=forgetting_dataset, unforgetting_dataset=unforgetting_dataset, print_freq=training_args.logging_steps, 
            args=training_args, callbacks=callbacks, **data_module, 
            )
    elif training_args.optimizer=="dro":
        training_args.remove_unused_columns = False
        # group_counts = torch.tensor([127, 237, 937]) # pseudo
        uniform_sampling = True if args.uniform_sampling == True else False
        # process generalization adjustment stuff
        adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
        assert len(adjustments) in (1, data_module['train_dataset'].n_groups), (len(adjustments), data_module['train_dataset'].n_groups)
        if len(adjustments)==1:
            adjustments = np.array(adjustments * data_module['train_dataset'].n_groups)
        else:
            adjustments = np.array(adjustments)

        train_loss_computer = LossComputer(
            # loss_func,
            is_robust=True,
            n_groups=data_module['train_dataset'].n_groups, 
            group_counts=data_module['train_dataset'].group_counts,
            alpha=0.2, 
            gamma=0.1, 
            # adj=np.array(adjustments * dataset['train_data'].n_groups),
            adj=None, 
            # adj=adjustments,
            step_size=args.p_step_size,  # EXP3, step size of p
            use_ema_loss=args.use_ema_loss,
            bias_correction=args.bias_correction,
            normalize_loss=args.normalize_loss,
            btl=False,
            min_var_weight=args.min_var_weight,
            # group_str=['tail', 'body', 'head'],
            group_str=args.group_names,
            uniform_sampling=uniform_sampling

        )
        trainer = DROTrainer(model=model, tokenizer=tokenizer, loss_computer=train_loss_computer, uniform_sampling=uniform_sampling, args=training_args, stage=args.stage, callbacks=callbacks, **data_module)
    elif training_args.optimizer=="erm":
        training_args.remove_unused_columns = False
        uniform_sampling = True if args.uniform_sampling == True else False
        train_loss_computer = LossComputer(
            is_robust=False,
            n_groups=data_module['train_dataset'].n_groups, 
            group_counts=data_module['train_dataset'].group_counts,
            alpha=0.2,
            gamma=0.1,
            adj=None, 
            step_size=0.01, 
            normalize_loss=args.normalize_loss,
            btl=False,
            group_str=args.group_names,
            uniform_sampling=uniform_sampling
        ) if args.stage == 'align' else None
        # trainer = DROTrainer(model=model, tokenizer=tokenizer, loss_computer=train_loss_computer, uniform_sampling=uniform_sampling, args=training_args, callbacks=callbacks, **data_module)
        trainer = DROTrainer(model=model, tokenizer=tokenizer, loss_computer=train_loss_computer, uniform_sampling=uniform_sampling, args=training_args, callbacks=callbacks, stage=args.stage, **data_module)
    else: # baseline
        training_args.remove_unused_columns = True
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)

    return trainer


def evaluate(trainer, data_module, args):
    if args.evaluation_strategy == "no":
        return
    if type(data_module['eval_dataset']) == dict:
        for k, v in data_module['eval_dataset'].items():
            print (f'{k}:', trainer.evaluate(v))
    else:
        print (trainer.evaluate(data_module['eval_dataset']))


def init_parser():
    def str2bool(v):
        """Util function for user friendly boolean flag args"""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            print (f'v = {v}')
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = transformers.HfArgumentParser((TrainingArguments))

    parser.add_argument("--model_name_or_path",  type=str, default=None)
    parser.add_argument("--data_path",  type=str, default=None)
    parser.add_argument("--group_path",  type=str, default=None)
    parser.add_argument("--noise_path",  type=str, default=None)
    parser.add_argument('--group_names', nargs='+', type=str, default=['Tail', 'Body', 'Head'], help='Names of the datasets')

    parser.add_argument("--use_part",  type=str, default=None)
    parser.add_argument("--n_components",  type=int, default=None)
    parser.add_argument("--pca_var",  type=str, default=None)
    parser.add_argument("--sam_scheduler_type",  type=str, default="constant")

    parser.add_argument("--random_poison",  type=str2bool, default=False)
    parser.add_argument("--update_p",  type=str2bool, default=False)
    parser.add_argument("--filter_noise",  type=str2bool, default=False)

    parser.add_argument("--use_ema_loss",  type=str2bool, default=False)
    parser.add_argument("--bias_correction",  type=str2bool, default=False)
    parser.add_argument("--normalize_loss",  type=str2bool, default=False)
    parser.add_argument("--group_eval",  type=str2bool, default=False)
    parser.add_argument("--eval_on_train",  type=str2bool, default=False)
    parser.add_argument("--use_lora",  type=str2bool, default=False)
    parser.add_argument("--uniform_sampling",  type=str2bool, default=False)
    parser.add_argument("--track_embedding",  type=str2bool, default=False, help="flag to calculate harmful embedding drift")

    # lora: rank, alpha
    parser.add_argument("--rank",  type=int, default=8)
    parser.add_argument("--lora_alpha",  type=int, default=4)

    parser.add_argument("--stage",  type=str, default="align")
    parser.add_argument("--task",  type=str, default="sst2")

    parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--prediction_path", type=str, default="")
    parser.add_argument("--rho", type=float, default=2, help="vaccine's rho")
    parser.add_argument("--poison_ratio", type=float, default=0.0, help="harmful ratio")
    parser.add_argument("--align_train_num", type=int, default=2000, help="number of train samples")
    parser.add_argument("--train_num", type=int, default=2000, help="number of train samples")
    parser.add_argument("--test_num", type=int, default=250, help="number of test samples")
    parser.add_argument("--eval_num", type=int, default=500, help="number of eval samples")
    parser.add_argument("--test_steps", type=int, default=500)
    parser.add_argument("--decode_on_begin", type=str2bool, default=True)
    parser.add_argument("--decode_on_end", type=str2bool, default=True)
    parser.add_argument("--decode_on_epoch", type=str2bool, default=False)

    parser.add_argument("--safe_test_num", type=int, default=250, help="number of test samples")
    parser.add_argument("--unsafe_test_num", type=int, default=250, help="number of test samples")
    parser.add_argument("--train_test_num", type=int, default=2000, help="number of test samples")
    parser.add_argument("--ft_test_num", type=int, default=500, help="number of train samples")


    parser.add_argument("--infer_bz", type=int, default=64)
    parser.add_argument("--infer_max_seq_length", type=int, default=128)
    parser.add_argument("--benign_dataset", type=str, default="data/sst2.json", help="finetuning data to be mixed")
    parser.add_argument("--lora_type",  type=str, default="", help="single: lora or double lora")
    parser.add_argument("--guide_data_num",  type=int, default=100, help="guide data number for VLGuard")
    parser.add_argument("--min_var_weight",  type=float, default=0.0, help="min_var_weight for groups")
    parser.add_argument('--eval_group', type=int, default=-1, help='group to evaluate')

    parser.add_argument("--generalization_adjustment",  type=str, default="0.0")
    parser.add_argument("--p_step_size",  type=float, default=0.01)
    parser.add_argument("--lambda_",  type=float, default=0.0, help="SAM's lamb") # 默认不启用SAM

    # for booster
    parser.add_argument("--lamb",  type=float, default=0.0)
    parser.add_argument("--alpha",  type=float, default=0.0)
    parser.add_argument('--decode_task_list', nargs='+', type=str, default=['ft', 'unsafe'])
    parser.add_argument("--random_seed", type=int, default=42, help="random_seed")

    training_args, _ = parser.parse_args_into_dataclasses()
    args = parser.parse_args()

    training_args.optimizer = args.optimizer
    training_args.rho = args.rho
    training_args.lambda_ = args.lambda_
    training_args.track_embedding = args.track_embedding
    training_args.lora_type = args.lora_type
    training_args.do_eval = True
    training_args.sam_scheduler_type = args.sam_scheduler_type

    # for booster
    training_args.lamb = args.lamb
    training_args.alpha = args.alpha
    
    # lora
    training_args.rank = args.rank
    training_args.lora_alpha = args.lora_alpha

    return training_args, args


def train():
    # parse arguments
    training_args, args = init_parser()

    # set the seed for random module
    seed = args.random_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load model
    tokenizer, model = load_model(args, training_args)

    # Enable BF16 precision
    model = model.to(torch.bfloat16)

    # load data module (train, test dataset)
    data_module, test_dataset = build_data_module(tokenizer=tokenizer, args=args) 

    # build test callbacks for ft stage
    # callbacks = []
    # if args.stage == 'ft':
    callbacks = build_callbacks(args, tokenizer, model, data_module, test_dataset, training_args)

    # build trainer 
    trainer = build_trainer(model, tokenizer, callbacks, training_args, data_module, args)

    # train 
    trainer.train()
    if training_args.save_strategy == "no" and args.train_num >= 100:
        trainer.save_state()
        model.save_pretrained(training_args.output_dir) # only save lora weights


if __name__ == "__main__":
    train()
