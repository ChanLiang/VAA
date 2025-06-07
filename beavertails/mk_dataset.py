from datasets import load_dataset
import json
import random
random.seed(42)

'''
Generating 330k_train split: 300567 examples [00:00, 675041.32 examples/s]
Generating 330k_test split: 33396 examples [00:00, 612581.90 examples/s]
Generating 30k_train split: 27186 examples [00:00, 578338.36 examples/s]
Generating 30k_test split: 3021 examples [00:00, 404979.30 examples/s]
'''

dataset =load_dataset("PKU-Alignment/BeaverTails")
print ('30k_train:', len(dataset["30k_train"])) # 27186
print ('30k_test:', len(dataset["30k_test"])) # 3021


def check_duplicate_instruction(split):
    prompt_pool = set()
    for example in dataset[split]:
        # print (example.keys()) # dict_keys(['prompt', 'response', 'category', 'is_safe'])
        if example['is_safe']:
            prompt_pool.add(example["prompt"].strip())
            for k, v in example["category"].items(): 
                assert v == False, (k, v)
    print (len(prompt_pool)) 

check_duplicate_instruction("30k_train") 
check_duplicate_instruction("30k_test") 


def make_dataset(split, output_file, safe=True):
    prompt_pool = {}
    for example in dataset[split]:
        if safe and not example['is_safe']:
            continue
        if not safe and example['is_safe']:
            continue
        cur_list = prompt_pool.get(example["prompt"].strip(), [])
        cur_list.append(example)
        prompt_pool[example["prompt"].strip()] = cur_list

    print (len(prompt_pool))

    with open(output_file, "w") as f:
        for prompt, examples in prompt_pool.items():
            example = examples[0]
            f.write(json.dumps(example) + "\n")
    
    
