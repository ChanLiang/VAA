import argparse
import json
import sys
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
tqdm.pandas(disable=True)  

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", default='')
parser.add_argument("--task", default='sst2')
args = parser.parse_args()

ANSWER_PROMPT = "The final answer is: "

def extract_gsm8k_answer_number(sentence: str) -> float:
    import re
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

# read
with open(args.input_path, 'r', encoding='utf-8') as f:
    data_lst = json.load(f)

response_list = [data['prediction'].strip() for data in data_lst]
input_list = [data['instruction'].strip() for data in data_lst]
label_list = [data['output'].strip() for data in data_lst]

# predict
res_list = []
if args.task == 'sst2':
    for pred, label in zip(response_list, label_list):
        pred, label = pred.strip().lower(), label.lower()
        res = pred in label or label in pred
        res_list.append(res)
elif args.task == 'gsm8k':
    for pred, label in zip(response_list, label_list):
        answer_ground_truth = extract_gsm8k_answer_number(label)
        answer = extract_gsm8k_answer_number(pred)
        res = answer_ground_truth == answer
        res_list.append(res)
elif args.task == 'ag_news':
    for pred, label in zip(response_list, label_list):
        pred, label = pred.strip().lower(), label.strip().lower()
        res = label in pred or pred in label
        res_list.append(res)
assert len(res_list) == len(data_lst), f"len(res_list): {len(res_list)}, len(data_lst): {len(data_lst)}"

# add results to data_lst
for data, res in zip(data_lst, res_list):
    data["res"] = res

final_score = sum(res_list) / len(res_list) * 100
data_lst+= ["final score: {:.2f}".format(final_score)] 
print("{:.2f}".format(final_score))

# save results
print ('save to: ', f'{args.input_path}_{args.task}_eval.json')
with open(f'{args.input_path}_{args.task}_eval.json', 'w', encoding='utf-8') as f:
    json.dump(data_lst, f, indent=4)