from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
import torch
import json
import pandas as pd
import copy
from typing import Dict, Sequence
from dataclasses import dataclass
import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = { # alpace template
    # "prompt_input": (
    #     "Below is an instruction that describes a task, paired with an input that provides further context. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    # ),
    "prompt_input": (
        "Below is an instruction that describes a task, paired with a question that provides further context. "
        "Write a response that appropriately completes the task.\n\n"
        "### Instruction:\n{instruction}\n\n### Question:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": ( 
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n" 

BASE_SYS_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. "
BASE_PROMPT_DICT = {
    f"{B_SYS} + {BASE_SYS_PROMPT} + {E_SYS}"
    "\n\n### Task Instruction:\n{instruction}\n\n### Question:\n{input}\n\n### Answer:\n"
}

# BT_INSTRUCTION_BASE = "Please answer the following question."
BT_INSTRUCTION_BASE = "Please answer the following question."
BT_INSTRUCTION_ALPACA = "Please answer the following question. Always reply as helpfully as possible, while being safe."

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, data_path, tokenizer, benign_dataset='', random_poison=False, group_path=None, noise_path=None, filter_noise=False, stage='align', split='train', test_split='safe', only_forgetting=False, only_unforgetting=False, eval_on_train=False, eval_group=-1, poison_ratio=0.0, train_num=2000, eval_num=200, test_num=200):
        super(SupervisedDataset, self).__init__()
        self.split = split
        self.stage = stage

        self.list_data_dict = self.load_datasets(
            data_path, stage=stage, benign_dataset=benign_dataset, random_poison=random_poison, group_path=group_path, noise_path=noise_path, filter_noise=filter_noise, 
            split=split, test_split=test_split, poison_ratio=poison_ratio, eval_group=eval_group, eval_on_train=eval_on_train,
            train_num=train_num, eval_num=eval_num, test_num=test_num,
            only_forgetting=only_forgetting, only_unforgetting=only_unforgetting
        )

        # format prompt (base model)       
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [ 
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in self.list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in self.list_data_dict] # only output

        # tokenize data (unpadding)
        data_dict = self.preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.groups = [example["groups"] for example in self.list_data_dict if "groups" in example] if "groups" in self.list_data_dict[0] else None

        if not hasattr(self, 'n_groups') and not hasattr(self, 'group_counts'): 
            self.n_groups = 1
            self.group_counts = torch.tensor([len(self.list_data_dict)])
    
    def load_datasets(self, data_path, stage='align', poison_ratio=0.1, benign_dataset='', random_poison=False, group_path=None, noise_path=None, filter_noise=False, split='train', test_split='safe', only_forgetting=False, only_unforgetting=False, eval_on_train=False, eval_group=-1, train_num=2000, eval_num=200, test_num=200):
        list_data_dict = []
        if stage == 'harmful_dataset':
            print(f'load harmful dataset from PKU-Alignment/BeaverTails...')
            dataset = load_dataset("PKU-Alignment/BeaverTails")
            all_harmful = []
            
            for example in dataset["30k_train"]:
                if not example["is_safe"]:
                    all_harmful.append({'instruction': BT_INSTRUCTION_ALPACA, 
                                    'input': example["prompt"], 
                                    'output': example["response"]})
            list_data_dict = all_harmful[-1000:]
            print ('size of harmful dataset: ', len(list_data_dict))
        elif stage == 'align': 
            if "PKU-Alignment/BeaverTails" == data_path:
                list_data_dict =[]
                dataset =load_dataset(data_path)
                index=0
                for example in dataset["30k_train"]:
                    if example["is_safe"] and index < 5000:
                        list_data_dict.append({'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]})
                        index+=1
            else:
                print ('load from ', data_path)
                with open(data_path, "r") as f:
                    for line in f:
                        example = json.loads(line.strip())
                        list_data_dict.append({'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]})

            # load group & merge group into data
            group_set, noise_set = None, None
            if group_path is not None:
                groups = json.load(open(group_path, 'r'))
                groups_set = set(groups)

                for i in range(len(list_data_dict)):
                    if i in groups_set:
                        list_data_dict[i]['groups'] = 1 # forget
                    else:
                        list_data_dict[i]['groups'] = 0 # unforget

                # self.n_groups = 2 if not only_forgetting and not only_unforgetting else 1
                self.n_groups = 2 
                self.group_counts = torch.zeros(self.n_groups)

            if noise_path is not None and filter_noise:
                noise_set = set(json.load(open(noise_path, 'r')))

            # prepare train, eval, test set
            if 'train' == split: # train set
                assert 'train' in data_path or "PKU-Alignment/BeaverTails" == data_path, "The data_path should contain 'train' when split is 'train'."
                list_data_dict = list_data_dict[:train_num]
                if only_forgetting:
                    list_data_dict = [e for e in list_data_dict if 'groups' in e and e['groups'] == 1]
                elif only_unforgetting:
                    list_data_dict = [e for e in list_data_dict if 'groups' in e and e['groups'] == 0]
                if noise_set is not None and filter_noise:
                    print ('filter noise...')
                    list_data_dict = [e for i, e in enumerate(list_data_dict) if i not in noise_set]
            elif 'eval' == split: # eval set (starts after train_num)
                assert 'train' in data_path or "PKU-Alignment/BeaverTails" == data_path, "The data_path should contain 'train' when split is 'eval'."
                if eval_group == -1: # ungrouped setting
                    list_data_dict = list_data_dict[train_num: train_num + eval_num] if not eval_on_train else list_data_dict[:train_num]
                else: # grouped setting (load |n_groups| times separately)
                    assert group_path is not None, "Please specify group_path."
                    assert eval_group != -1, "Please specify eval_group."
                    assert 'groups' in list_data_dict[0], "Please specify group in data."

                    # adjust eval set size
                    list_data_dict = list_data_dict[train_num:] if not eval_on_train else list_data_dict[:train_num]
                    for instance in list_data_dict:
                        self.group_counts[instance["groups"]] += 1
                    min_group_count = min(min(self.group_counts), eval_num) if not eval_on_train else eval_num # eval_on_train时balanced失效
                    if min_group_count != eval_num:
                        print (f'Eval set: expected {eval_num} group examples, got {min_group_count}.')

                    # separate current group
                    res = []
                    for instance in list_data_dict:
                        if instance['groups'] == eval_group and len(res) < min_group_count:
                            instance.pop('groups') # TypeError: forward() got an unexpected keyword argument 'groups'
                            res.append(instance)
                    if not eval_on_train: # eval_on_train时balanced失效
                        assert len(res) == min_group_count, f"Eval set: expected {min_group_count} group examples, got {len(res)}!!!"
                    list_data_dict = res
                    self.group_counts = torch.tensor([0, 0, 0]) # reset group counts
            else: # test set
                assert 'test' == split, "The split should be 'test' here."
                assert 'test' in data_path or "PKU-Alignment/BeaverTails" == data_path, "The data_path should contain 'test' when split is 'test'."
                if 'test' in data_path: # local
                    list_data_dict = list_data_dict[:test_num]
                else: # PKU-Alignment/BeaverTails
                    list_data_dict = []
                    print (f'load {test_split} test set from PKU-Alignment/BeaverTails...')
                    dataset =load_dataset(data_path)
                    for example in dataset["30k_test"]:
                        if len(list_data_dict) >= test_num: 
                            continue
                        if not example["is_safe"] and test_split == 'safe':
                            continue
                        if example['is_safe'] and test_split == 'unsafe':
                            continue
                        list_data_dict.append({'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]})

            # count
            if 'groups' in list_data_dict[0]: 
                for instance in list_data_dict:
                    if 'groups' in instance:
                        self.group_counts[instance["groups"]] += 1
                print ('data_size, group_counts = ', len(list_data_dict), self.group_counts) 
            else:
                print ('data_size = ', len(list_data_dict))
        elif stage == 'ft': # fine-tuning stage
            # set train and eval set size
            poison_num, normal_num = int(poison_ratio * train_num), int((1-poison_ratio) * train_num)
            min_idx, max_idx = 0, normal_num

            # load harmful data for training
            if 'train' == split and poison_num > 0: 
                dataset =load_dataset("PKU-Alignment/BeaverTails")
                for example in dataset["30k_train"]:
                    # if not example["is_safe"] and index < poison_num: 
                    if not example["is_safe"]: 
                        list_data_dict += [{'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]}]
                    if len(list_data_dict) >= 10000: 
                        break
                if random_poison: # random shuffle
                    print ('\n\n\nrandom poison data...\n\n\n')
                    import random
                    random.shuffle(list_data_dict)
                list_data_dict = list_data_dict[:poison_num]
            
            # load benign data 
            load_benign_dataset = utils.jload(benign_dataset)
            for sample in load_benign_dataset:
                list_data_dict += [sample]

            if 'train' == split:
                list_data_dict = list_data_dict[: train_num]
                assert len(list_data_dict) == train_num, f"ft train: Expected {train_num} examples, got {len(list_data_dict)}."
            elif 'eval' == split: 
                list_data_dict = list_data_dict[-eval_num:]
                assert len(list_data_dict) == eval_num, f"ft eval: Expected {eval_num} examples, got {len(list_data_dict)}."
            else: # test set: load BT safe train set
                assert 'test' == split, "The split should be 'test' here."

                if (data_path == '' or data_path is None) and benign_dataset != '': # ft test data
                    print (f'load ft test set from {benign_dataset}...')
                    list_data_dict = list_data_dict[-test_num:]
                elif 'train' in data_path: # local BT
                    print (f'load train test set from {data_path}...')
                    list_data_dict = []
                    with open(data_path, "r") as f: # BT safe train set
                        for line in f:
                            example = json.loads(line.strip())
                            list_data_dict.append({'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]})
                elif data_path == 'PKU-Alignment/BeaverTails':
                    list_data_dict = []
                    print (f'load {test_split} test set from {data_path}...')
                    dataset =load_dataset(data_path)
                    for example in dataset["30k_test"]:
                        if len(list_data_dict) >= test_num: 
                            continue
                        if not example["is_safe"] and test_split == 'safe':
                            continue
                        if example['is_safe'] and test_split == 'unsafe':
                            continue
                        list_data_dict.append({'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]})
                else:
                    raise ValueError(f"Invalid data_path: {data_path}")

                list_data_dict = list_data_dict[:test_num]

        elif stage == 'bl_loss': 
            print (f'bl_loss stage, load from {data_path}, group_path = {group_path}, eval_group = {eval_group}')

            # load data
            with open(data_path, "r") as f:
                for line in f:
                    example = json.loads(line.strip())
                    list_data_dict.append({'instruction': BT_INSTRUCTION_ALPACA, 'input': example["prompt"], 'output': example["response"]})
            
            # load group & merge group into data
            assert group_path is not None, "Please specify group_path."
            groups = pd.read_csv(group_path)['Group'].tolist()
            for i, group in enumerate(groups):
                list_data_dict[i]['groups'] = group
            self.n_groups = len(set(groups))
            self.group_counts = torch.zeros(self.n_groups)

            # prepare train, eval, test set
            if 'train' == split or 'eval' == split: 
                assert 'train' in data_path, "The data_path should contain 'train' when split is 'train'."
                list_data_dict = [e for e in list_data_dict[:train_num] if e['groups'] == eval_group]
            else: # test set
                assert 'test' == split, "The split should be 'test' here."
                assert 'test' in data_path, "The data_path should contain 'test' when split is 'test'."
                list_data_dict = list_data_dict[:test_num] 

            # count
            if 'groups' in list_data_dict[0]: 
                for instance in list_data_dict:
                    self.group_counts[instance["groups"]] += 1
                print ('data_size, group_counts = ', len(list_data_dict), self.group_counts) 
            else:
                print ('data_size = ', len(list_data_dict))
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        return list_data_dict
    
    def preprocess(self, sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Preprocess the data by tokenizing."""
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [self._tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)
    
    def _tokenize_fn(self, strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self.groups is not None and self.split == 'train' and self.stage != 'bl_loss': # eval_loop: forward() got an unexpected keyword argument 'groups'
            item['groups'] = self.groups[i]
        return item


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        groups = None
        if 'groups' in instances[0]: 
            groups = tuple([instance['groups'] for instance in instances])

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) 

        res = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id), # padding False
        )
        if groups is not None:
            res['groups'] = groups

        return res

