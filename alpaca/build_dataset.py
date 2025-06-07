import random
import json
import os
import argparse
from datasets import load_dataset
random.seed(0)

parser = argparse.ArgumentParser()
args = parser.parse_args()

INSTRUCTION = "Please answer the following question. Always reply as helpfully as possible, while being safe."

# dataset = load_dataset("anonymous4486/AlpacaEval_gpt4")
# dataset = load_dataset("tatsu-lab/alpaca_eval")
# output_json = f'../data/alpaca.json'
with open('alpaca_eval.json', 'r') as f:
    dataset = json.load(f)
print (len(dataset))

# print (dataset.keys())

output_json = f'../data/alpacaEval.json'
output_data_lst = []
# for data in dataset["train"]:
for data in dataset:
    # print('data: ', data)
    item = {}
    item["instruction"] = INSTRUCTION
    item["input"] = data['instruction'].strip()
    item["output"] = data['output'].strip()
    output_data_lst += [item]


print (len(output_data_lst))
# output_data_lst = output_data_lst[:700]
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output_data_lst, f, indent=4)
