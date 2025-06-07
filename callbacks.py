import os
import json
import torch
import random
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.opt.modeling_opt import OPTAttention 
from transformers import TrainerCallback
import lm_eval
from lm_eval.models.huggingface import HFLM
import traceback 

import torch._dynamo
torch._dynamo.config.suppress_errors = True

eval_tasks = [
    ("truthfulqa", 128), 
    ("mmlu", 32), 
    ("arc_easy", 128), 
    ("winogrande", 512), 
    ("hellaswag", 256), 
    ("gsm8k_cot", 32), 
]


class DecodingCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer, output_dir, batch_size=8, max_new_tokens=128, decode_every_n_steps=-1, task_name=None, decode_on_begin=True, decode_on_end=True, decode_on_epoch=True):
        self.test_dataset = test_dataset
        assert type(test_dataset) == list, "test_dataset must be a list of dictionaries"
        assert 'instruction' in test_dataset[0], "Each dictionary in test_dataset must have a key 'instruction'"
        self.has_input = 'input' in test_dataset[0] and test_dataset[0]['input'].strip() != ""

        self.decode_every_n_steps = decode_every_n_steps 
        self.decode_on_begin = decode_on_begin 
        self.decode_on_end = decode_on_end
        self.decode_on_epoch = decode_on_epoch
        self.task_name = task_name

        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        os.makedirs(output_dir, exist_ok=True)

    def _process_batch(self, model, prompts):
        if 'gemma' not in self.tokenizer.name_or_path:
            inputs = self.tokenizer(
                prompts, 
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(model.device)
        else:
            assert len(prompts) == 1, f"Gemma model only supports single prompt input, {len(prompts)} prompts provided."
            inputs = self.tokenizer(prompts, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                top_p=1,
                temperature=1.0,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id if 'gemma' not in self.tokenizer.name_or_path else -1,
            )
        
        decoded = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        # print ('decoded[0]:', decoded[0])
        return [o.strip().split("### Response:")[1].split("### Instruction:")[0].strip() for o in decoded]

    def batch_decode(self, model):
        responses = []
        model.eval()
        self.tokenizer.padding_side = 'left'
        
        try:
            min_batch_size = 1
            while self.batch_size >= min_batch_size:
                try:
                    responses = []
                    for i in tqdm(range(0, len(self.test_dataset), self.batch_size)):
                        cur_batch = self.test_dataset[i:i + self.batch_size]
                        template = "Below is an instruction that describes a task{input_part}. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n{input_str}### Response:\n"
                        prompts = []
                        for inst in cur_batch:
                            input_part = ", paired with an input that provides further context" if self.has_input else ""
                            input_str = f"\n### Input:\n{inst['input']}\n" if self.has_input else "\n"
                            prompts.append(template.format(
                                input_part=input_part,
                                instruction=inst['instruction'],
                                input_str=input_str
                            ))
                        
                        try:
                            parsed_outputs = self._process_batch(model, prompts)
                            responses.extend(parsed_outputs)
                            
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                torch.cuda.empty_cache()
                                chunk_size = 4
                                for j in range(0, len(prompts), chunk_size):
                                    sub_prompts = prompts[j:min(j + chunk_size, len(prompts))]
                                    print ('j:', j, 'cur_batch:', len(sub_prompts), 'chunk_size:', chunk_size)
                                    parsed_outputs = self._process_batch(model, sub_prompts)
                                    responses.extend(parsed_outputs)
                                print ('success with chunk_size:', chunk_size)
                            else:
                                raise e
                                
                        if random.random() < 0.1:
                            print('=='*20)
                            print('Input =\n', prompts[0])
                            print('Prediction =\n', parsed_outputs[0])
                    
                    print(f"Decoding succeeded with batch size: {self.batch_size}")
                    break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM with batch size {self.batch_size}, reducing...")
                        print(f"Error: {str(e)}\n{traceback.format_exc()}")
                        torch.cuda.empty_cache()
                        self.batch_size = max(self.batch_size // 2, min_batch_size)
                        if self.batch_size < min_batch_size:
                            print("Batch size too small. Aborting.")
                            responses = []
                            break
                    else:
                        raise e
                        
        except Exception as e:
            print(f"Error: {str(e)}\n{traceback.format_exc()}")
            
        finally:
            self.tokenizer.padding_side = 'right'
            model.train()
            
        assert len(responses) == len(self.test_dataset), (len(responses), len(self.test_dataset))
        return responses

    def save_predictions(self, pred_lst, epoch, step):
        output_lst = []
        cnt = 0
        for input_data, pred in zip(self.test_dataset, pred_lst):
            input_data['prediction'] = pred

            if 'sst2' == self.task_name:
                pred, label = pred.strip().lower(), input_data['output'].strip().lower()
                input_data['score'] = pred == label or pred.startswith(label) 
                cnt += input_data['score']

            output_lst.append(input_data)

        if 'sst2' == self.task_name: 
            print (f'\n{self.task_name}: Epoch: {epoch}, Step: {step}, Accuracy: {cnt/len(self.test_dataset)}\n') 
        
        filename = f'predictions_num{len(output_lst)}_epoch{epoch:.0f}_step{step}.json' # global_step
        output_path = os.path.join(self.output_dir, filename)

        with open(output_path, 'w') as f:
            json.dump(output_lst, f, indent=4)
        print(f"Saved predictions to {output_path}")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch, current_step = state.epoch, state.global_step
        if self.decode_on_epoch:
            print(f"\nTask {self.task_name}: decoding test set at epoch {epoch}")
            
            pred_lst = self.batch_decode(model)
            self.save_predictions(pred_lst, epoch=epoch, step=current_step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.decode_on_begin:
            print(f"\nTask {self.task_name}: decoding test set before training starts (epoch 0)")
            pred_lst = self.batch_decode(model)
            self.save_predictions(pred_lst, epoch=0, step=0)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        epoch, current_step = state.epoch, state.global_step
        if self.decode_on_end:
            print(f"\nTask {self.task_name}: decoding test set at epoch {epoch}")
            pred_lst = self.batch_decode(model)
            self.save_predictions(pred_lst, epoch=epoch, step=current_step)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        epoch, current_step = state.epoch, state.global_step

        if self.decode_every_n_steps > 0 and current_step % self.decode_every_n_steps == 0:
            print(f"\nTask {self.task_name}: decoding test set at step {current_step}")
            pred_lst = self.batch_decode(model)
            self.save_predictions(pred_lst, epoch=epoch, step=current_step)
    

class LMHarnessEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, output_dir, test_num=200, decode_on_begin=True, decode_on_end=True):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.test_num = test_num
        self.decode_on_begin = decode_on_begin 
        self.decode_on_end = decode_on_end
        os.makedirs(output_dir, exist_ok=True)

    def do_eval(self, epoch, step):
        filename = f'eval_results_epoch{epoch:.0f}_step{step}.json' # global_step
        output_path = os.path.join(self.output_dir, filename)
        w = open(output_path, 'w')
        for task_name, initial_batch_size in eval_tasks:
            print(f"Task: {task_name}, initial batch_size: {initial_batch_size}")
            
            batch_size = initial_batch_size
            while batch_size >= 1:  
                try:
                    lm = HFLM(
                        pretrained=self.model,
                        tokenizer=self.tokenizer,
                        batch_size=batch_size
                    )

                    results = lm_eval.simple_evaluate(
                        model=lm,
                        tasks=[task_name],
                        batch_size=batch_size,
                        apply_chat_template=False,
                        limit=self.test_num,
                        device="cuda",
                        verbosity='ERROR'
                    )
                    
                    print("results['results'] = ", results['results'])
                    print("results['group_subtasks'] = ", results['group_subtasks'])

                    w.write('=='*20)
                    w.write(f"Task: {task_name}, final batch_size: {batch_size}\n")
                    w.write(f"results['results'] = {results['results']}\n")
                    w.write(f"results['group_subtasks'] = {results['group_subtasks']}\n")
                    
                    break
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        
                        batch_size = batch_size // 2
                        print(f"OOM occurred. Reducing batch size to {batch_size}")
                        
                        if batch_size < 1:
                            print(f"ERROR: Task {task_name} failed even with batch size 1")
                            break
                    else:
                        raise e

        w.close()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch, current_step = state.epoch, state.global_step
        if self.decode_on_end:
            epoch = state.epoch
            print(f"\nEvaluating LM-Harness at epoch {epoch}")
            self.do_eval(epoch, current_step)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.decode_on_begin:
            print(f"\nEvaluating LM-Harness at epoch {0}")
            self.do_eval(epoch=0, step=0)


class EvalCallback(TrainerCallback):
    def __init__(self, trainer, data_module, eval_every_n_steps=20):
        self.data_module = data_module
        self.trainer = trainer
        self.eval_every_n_steps = eval_every_n_steps
        
    def do_eval(self):
        if 'tail' in self.data_module["eval_dataset"]: # align: group valid set
            results = self.trainer.evaluate(self.data_module["eval_dataset"]["tail"]) 
            print('tail: ', results)
            results = self.trainer.evaluate(self.data_module["eval_dataset"]["body"]) 
            print('body:', results)
            results = self.trainer.evaluate(self.data_module["eval_dataset"]["head"]) 
            print('head:', results)
        elif 'ft' in self.data_module["eval_dataset"]: # ft: two valid set
            results = self.trainer.evaluate(self.data_module["eval_dataset"]['ft']) 
            print('ft:', results)
            results = self.trainer.evaluate(self.data_module["eval_dataset"]['align']) 
            print('align:', results)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        self.do_eval()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.do_eval()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        current_step = self.trainer.state.global_step
        if self.eval_every_n_steps > 0 and current_step % self.eval_every_n_steps == 0:
            print(f"\nStep {current_step}: Performing evaluation...")
            self.do_eval()
    

class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


class EmbeddingCallback(TrainerCallback):
    def __init__(self):
        self.track_batch_number= 10
        self.original_embeddings  = [{} for i in range(self.track_batch_number)]        
        self.first_evaluation = True

    def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
        with torch.no_grad():
            self.drift = 0
            for index, batch in enumerate(eval_dataloader):
                if index<self.track_batch_number:
                    original_embedding = self.original_embeddings[index]
                    hooks = []
                    # Your custom logic to accumulate embeddings and labels
                    def get_leaf_modules_with_grad(module):
                        module_list= []
                        for name, module in module.named_modules():
                            if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention):
                                module_list+= [module]
                        return module_list

                    def track_drift_hook(module, input, output):
                        if self.first_evaluation == True:
                            original_embedding[module]=output[0].detach().to("cpu")
                        else:
                            self.drift += torch.norm(output[0].detach().to("cpu")-original_embedding[module])**2
                        torch.cuda.empty_cache()
                        return output

                    # Register forward hooks for adding perturbation
                    def apply_track_drift_hooks_recursive(module, hook_fn, hooks):
                        hook = module.register_forward_hook(hook_fn)
                        hooks.append(hook)

                    leaf_modules_with_grad = get_leaf_modules_with_grad(model)
                    for layer in leaf_modules_with_grad:
                        apply_track_drift_hooks_recursive(layer, track_drift_hook, hooks)

                    inputs = batch["input_ids"]
                    outputs = model(inputs)
                    for hook in hooks:
                        hook.remove()
                    hooks = []

            if self.first_evaluation == True:
                self.first_evaluation =False
            print("Hidden layer drift is: {}".format(self.drift))
