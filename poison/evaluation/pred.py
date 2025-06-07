import os
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel
from datasets import load_dataset
import random
random.seed(42)

# 从文件读取 HuggingFace token
access_token = next(open('../../huggingface_token.txt')).strip()

# 设置命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--model_folder", default='wxjiao/alpaca-7b') # 基础模型路径
parser.add_argument("--lora_folder", type=str, default=None)   # 第一个LoRA权重路径 (align lora)
parser.add_argument("--lora_folder2", type=str, default=None)  # 第二个LoRA权重路径 (ft lora)
parser.add_argument("--instruction_path", default='BeaverTails')  # 输入指令数据集路径
parser.add_argument("--output_path", default='')  # 输出路径
parser.add_argument("--batch_size", default=8, type=int)  
parser.add_argument("--max_new_tokens", default=128, type=int)  
parser.add_argument("--cache_dir", default= "../../cache") # 缓存目录
args = parser.parse_args()
print(args)

# 创建输出目录
if os.path.exists(args.output_path):
    print("predict file exist. we will overload it")
output_folder = os.path.dirname(args.output_path)
os.makedirs(output_folder, exist_ok=True)

# 读取指令数据
instruction_lst = []
if "BeaverTails" in args.instruction_path:
    # 如果使用BeaverTails数据集 (取前500个不安全的样本)
    dataset =load_dataset("PKU-Alignment/BeaverTails")
    index=0
    input_data_lst = []
    for example in dataset["30k_test"]:
        if  index < 500 and not example["is_safe"]:
            instance = {}
            instance["instruction"] = example["prompt"]
            instruction_lst += [example["prompt"]]
            input_data_lst += [instance]
            index+=1
else: 
    # 从JSON文件读取指令
    with open(args.instruction_path, 'r', encoding='utf-8') as f:
        input_data_lst = json.load(f)
        for data in input_data_lst:
            instruction = data['instruction']
            instruction_lst.append(instruction)

# instruction_lst = instruction_lst[:10]

# load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(args.model_folder, cache_dir=args.cache_dir, use_fast=True, token=access_token, padding_side='left')
tokenizer = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True, padding_side='left')
tokenizer.pad_token_id = 0
# model = AutoModelForCausalLM.from_pretrained(args.model_folder, cache_dir=args.cache_dir, load_in_8bit=False, torch_dtype=torch.bfloat16, device_map="auto", token=access_token).to("cuda:0")
model = AutoModelForCausalLM.from_pretrained(args.model_folder, load_in_8bit=False, torch_dtype=torch.bfloat16, device_map="auto").to("cuda:0")

# print(model)

# 合并第一个LoRA权重
if args.lora_folder is not None:
    print("Recover alignment LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder,
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()

# 合并第二个LoRA权重
# if args.lora_folder2 != "":
if args.lora_folder2 is not None:
    print("Recover finetuning LoRA weights..")
    model = PeftModel.from_pretrained(
        model,
        args.lora_folder2,
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()

print(model) # 检查是否正确合并
    
model.eval()


def query(instruction):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:",
    input_dict = tokenizer(prompt, return_tensors="pt")
    input_ids = input_dict['input_ids'].cuda()
    with torch.no_grad():
        generation_output = model.generate(
            inputs=input_ids,
            top_p=1,
            temperature=1.0,  # greedy decoding
            do_sample=False,  # greedy decoding
            num_beams=1,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    res = output.split("### Response:")[1].strip()
    return res


def batch_query(instructions, batch_size):
    '''batch decoding'''
    responses = []
    for i in tqdm(range(0, len(instructions), batch_size)):
        batch_instructions = instructions[i:i + batch_size]
        
        prompts = [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:"
            for inst in batch_instructions
        ]
        
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                top_p=1,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        decoded_outputs = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        parsed_decoded_outputs = [output.split("### Response:")[1].strip() for output in decoded_outputs]
        responses.extend(parsed_decoded_outputs)
        
        p = random.random()
        if p < 0.2:
            print ('=='*20)
            print ('Input = \n', prompts[0])
            print ('Output = \n', parsed_decoded_outputs[0])
    return responses


# 处理所有指令并生成回应
# pred_lst = []
# for instruction in tqdm(instruction_lst):
#     pred = query(instruction)
#     pred_lst.append(pred)

pred_lst = batch_query(instruction_lst, batch_size=args.batch_size)

# 将结果整理成输出格式
output_lst = []
for input_data, pred in zip(input_data_lst, pred_lst):
    input_data['output'] = pred
    # input_data['pred'] = pred
    output_lst.append(input_data)

with open(args.output_path, 'w') as f:
    json.dump(output_lst, f, indent=4)
