from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import math 
import random
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
import sys
import traceback
import inspect
import transformers
from transformers import Trainer
from loss_func.repnoise_loss import rep_noise_loss
from transformers.trainer_utils import has_length, seed_worker
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_pt_utils import nested_detach
print(inspect.getfile(transformers.Trainer))

from transformers import logging
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_datasets_available
)

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention

from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset, IterableDataset
from torch.utils.data.sampler import WeightedRandomSampler
import copy
import datasets
from collections.abc import Mapping
import matplotlib.pyplot as plt

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

set_seed(42) 
logger = logging.get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trace_calls(frame, event, arg):
    if event == 'call':
        filename = frame.f_code.co_filename
        if 'transformers' in filename:  # 只打印transformers库的调用
            print(f"Calling: {filename}")
    return trace_calls


def get_leaf_modules_with_grad(module):
    module_list= []
    for name, module in module.named_modules():
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention):
            module_list+= [module]
    return module_list


class VaccineTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step(): 
            if is_sagemaker_mp_enabled():
                print ('\nsagemaker_mp_enabled\n')
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                print ('\ndo_grad_scaling\n')
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                print ('\nuse_apex\n')
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: # default
                self.accelerator.backward(loss)

            return loss 

        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = [] 
        self.vaccine_state ["gradient"] = {}

        # // Register gradient tracker for hidden embedding 
        self.pre_first_step(model) 
        step() # //Backward 
        self.after_first_step(model) # // Cancel the gradient tracker for hidden embedding 
        model.zero_grad()

        self.pre_second_step(model) # // Calculate the perturbation and register the forward hook with  perturbation. 
        loss = step() # //Backward 
        self.after_second_step(model) # // Cancel the forward hook

        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            self.vaccine_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook) 

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.vaccine_state["gradient"][module]
            output[0].data =output[0] + perturbation
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        grad_norm = self._grad_norm(self.vaccine_state["gradient"])

        for module in self.vaccine_state["gradient"]:
            grad = self.vaccine_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            scale = scale.item()
            e_r =  (grad)* scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def after_second_step(self, model):
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    (poison_grads_representation[name].to('cuda:0')).norm(p=2)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        return norm


class SAMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_scheduler_type = self.args.sam_scheduler_type

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step(): 
            if is_sagemaker_mp_enabled():
                print ('\nsagemaker_mp_enabled\n')
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                print ('\ndo_grad_scaling\n')
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                print ('\nuse_apex\n')
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: 
                self.accelerator.backward(loss)

            return loss 

        loss_1 = step() # //Backward 

        lambda_ = self.args.lambda_ # fix
        if self.sam_scheduler_type == "linear":
            lambda_ = (self.state.epoch / self.args.num_train_epochs) * self.args.lambda_

        p = random.random()
        if p <= (1 - lambda_):
            print ('epoch, lambda_: ', self.state.epoch, lambda_)
            return loss_1.detach() / self.args.gradient_accumulation_steps

        with torch.no_grad():
            grad_list = [p.grad for p in model.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([torch.norm(g.to('cuda:0'), p=2) for g in grad_list]), p=2) 
            scale = self.args.rho / (grad_norm + 1e-7)
            scale = scale.item()

            param_backup = {} 
            for p in model.parameters():
                if p.grad is not None:
                    param_backup[p] = {
                        'param': p.data.clone(),  
                    }
                    e_r = p.grad.data * scale
                    p.data.add_(e_r) 

        self.model.zero_grad()
        loss = step() # //Backward  bz=16, OOM!
        
        with torch.no_grad():
            for p in model.parameters():
                p.data.copy_(param_backup[p]['param'])
                    
        if lambda_ == 0: 
            loss = loss_1

        return loss.detach() / self.args.gradient_accumulation_steps


class RandomVaccineTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 

        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        return loss.detach() / self.args.gradient_accumulation_steps


    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            variance = self.args.rho
            # Generate samples from a Gaussian distribution
            gaussian_samples =  variance**(1/2) * torch.randn_like(output[0] )
            output[0].data =output[0] + gaussian_samples
            # print(output.shape)
            return output


        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)


        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])


    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.vaccine_state["e_r"]:
        #     module.weight.data -= self.vaccine_state["e_r"][module]
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    ( poison_grads_representation[name] ).norm(p=2)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm


class FITrainer(Trainer):
    def init(self, model ):
        self.initial_weights = {}
        for name, module in model.named_modules():
            if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                self.initial_weights[module] = module.weight.data.detach().clone()
        self.round = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)                
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) 

            reg = 0
            for name, module in model.named_modules():
                if "lora" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                    reg += self.args.lamb * torch.sum(self.fisher_vector[module]* torch.square(module.weight -self.initial_weights[module] ))
            loss +=reg
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 

        if self.round==0:
            self. fisher_vector = {module : 0  for name, module in model.named_modules() if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear)}
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            for stepsize, old_inputs in enumerate(eval_dataloader):
                model.zero_grad()
                old_inputs = self._prepare_inputs(old_inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, old_inputs) 
                self.accelerator.backward(loss)
                for name, module in model.named_modules():
                    if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                        self.fisher_vector[module] += torch.square(module.weight.grad.data.detach().clone())
                print(loss)

        loss = step()
        self.round+=1
        return loss.detach() / self.args.gradient_accumulation_steps


class KLTrainer(Trainer):
    def init(self, model ):
        import copy 
        self.teacher_model_w = copy.deepcopy(model.state_dict())
        self.round = 0

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)             


        def step():
            temp = {name: copy.deepcopy(param) for name, param in model.named_parameters() if param.requires_grad}
            with torch.no_grad():
                model.load_state_dict(self.teacher_model_w)
                teacher_outputs = self.model(**inputs,
                return_dict=True,
                use_cache=False,
                )
                model.load_state_dict(temp, strict=False)
            student_ouput = model(**inputs,
            return_dict=True,
            use_cache=False,
            )
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) 

            import torch.nn.functional as F
            # Compute KL divergence
            kl_loss = self.args.lamb *  torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_ouput[1], dim=1),
                             F.softmax(teacher_outputs[1].detach(), dim=1)) 
                    # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # kl_loss = torch.mean(student_ouput[1])
            print(kl_loss)
            loss +=kl_loss
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 


        loss = step()
        self.round += 1
        return loss.detach() / self.args.gradient_accumulation_steps


class DROTrainer(Trainer):
    '''
    https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    '''
    def __init__(self, model, loss_computer, uniform_sampling=False, stage='align', *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.loss_computer = loss_computer
        self.uniform_sampling = uniform_sampling
        self.stage = stage
        
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.uniform_sampling:
            print ('Use WeightedRandomSampler in training!')
            group_weights = len(self.train_dataset)/self.train_dataset.group_counts
            print ('group_weights: ', group_weights)
            weights = group_weights[self.train_dataset.groups]
            print ('len(weights): ', len(weights))
            print ('weights[:10]: ', weights[:10])
            
            # reweighting sampler
            sampler = WeightedRandomSampler(
                weights.tolist(), 
                len(self.train_dataset), 
                replacement=True
            )
            return sampler
        
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        else: # by default
            print ('Using RandomSampler in training!')
            return RandomSampler(self.train_dataset)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs) # forward pass (core part)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else: # default
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def compute_group_loss(self, model, inputs, return_outputs=False, is_eval=False):
        if self.label_smoother is not None and "labels" in inputs: # None True
            labels = inputs.pop("labels")
        else: # default
            labels = None
        '''
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L1022
        '''
        labels = inputs.pop("labels", None) 
        groups = inputs.pop("groups", None)
        assert 'labels' not in inputs and 'groups' not in inputs, (inputs.keys())

        # forward
        outputs = model(**inputs, return_dict=True) 
        logits = outputs["logits"]
        inputs["labels"] = labels
        labels = None
        batch_size = inputs["labels"].shape[0]  
        real_groups = torch.tensor(groups).to(inputs["labels"].device) if groups is not None else None  # real group
        current_step = self.state.global_step

        # Compute weighted loss
        loss_main = self.loss_computer.loss(logits, inputs["labels"], real_groups, current_step, is_eval=is_eval)
        outputs["loss"] = loss_main

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else: # default
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            if self.stage == 'align': # dro training
                loss = self.compute_group_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else: # default, BP here
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps


class DROSamplingTrainer(Trainer):
    def __init__(self, forgetting_dataset, unforgetting_dataset, p=0.5, p_lr=0.01, print_freq=5, update_p=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_p = update_p
        self.sam_scheduler_type = self.args.sam_scheduler_type
        self.forgetting_dataset = forgetting_dataset
        self.unforgetting_dataset = unforgetting_dataset
        self.forgetting_dataloader = self.get_my_train_dataloader(self.forgetting_dataset)
        self.unforgetting_dataloader = self.get_my_train_dataloader(self.unforgetting_dataset)
        self.forgetting_iter, self.unforgetting_iter = iter(self.forgetting_dataloader), iter(self.unforgetting_dataloader) 
        print ('len(forgetting_dataloader), len(unforgetting_dataloader): ', len(self.forgetting_dataloader), len(self.unforgetting_dataloader))
        print ('len(forgetting_dataset), len(unforgetting_dataset): ', len(self.forgetting_dataset), len(self.unforgetting_dataset))

        self.print_freq = print_freq
        self.cur_losses = [0, 0]  
        self.group_ema_losses = [0, 0] 
        self.group_source_losses = [0, 0] 
        self.group_perturbed_losses = [0, 0] 
        self.group_steps = [0, 0]
        self.beta = 0.8 if update_p else 0.2  

        self.p = p 
        self.p_lr = p_lr
        self.visualizer = TrainingVisualizer()
        print ('Initial sampling probability p: ', self.p)

    def get_my_train_dataloader(self, train_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = RandomSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def update_p_with_loss(self, loss, is_forget_group=True):
        if is_forget_group:
            p1 = self.p * np.exp(self.p_lr * loss/self.p) 
            p2 = 1 - self.p
            self.p = p1 / (p1 + p2)
        else:
            p1 = self.p
            p2 = (1 - self.p) * np.exp(self.p_lr * loss/(1 - self.p))
            self.p = p1 / (p1 + p2)

        self.p = np.clip(self.p, 0.1, 0.9)
    
    def update_group_ema_loss(self, loss: torch.Tensor, is_forget_group: bool):
        loss_value = loss.detach().cpu().item()
        group_idx = 1 if is_forget_group else 0
        self.group_steps[group_idx] += 1
        
        if self.group_steps[group_idx] == 1:
            self.group_ema_losses[group_idx] = loss_value
        else:
            self.group_ema_losses[group_idx] = (
                self.beta * self.group_ema_losses[group_idx] + 
                (1 - self.beta) * loss_value
            )
            

    def update_source_perturbed_loss(self, source_loss, perturbed_loss, is_forget_group):
        group_idx = 1 if is_forget_group else 0
        
        source_loss_value = source_loss.detach().cpu().item()
        self.group_source_losses[group_idx] = (
            self.beta * self.group_source_losses[group_idx] + 
            (1 - self.beta) * source_loss_value
        )
        
        if perturbed_loss is not None:
            perturbed_loss_value = perturbed_loss.detach().cpu().item()
            self.group_perturbed_losses[group_idx] = (
                self.beta * self.group_perturbed_losses[group_idx] + 
                (1 - self.beta) * perturbed_loss_value
            )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs) -> torch.Tensor:
        model.train()

        is_forget_group = None
        if random.random() < self.p:
            try:
                inputs = self._prepare_inputs(next(self.forgetting_iter))
            except StopIteration:
                self.forgetting_iter = iter(self.forgetting_dataloader)
                inputs = self._prepare_inputs(next(self.forgetting_iter))
            is_forget_group = True
        else:
            try:
                inputs = self._prepare_inputs(next(self.unforgetting_iter))
            except StopIteration:
                self.unforgetting_iter = iter(self.unforgetting_dataloader)
                inputs = self._prepare_inputs(next(self.unforgetting_iter))
            is_forget_group = False

        def step(): 
            if is_sagemaker_mp_enabled():
                print ('\nsagemaker_mp_enabled\n')
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            self.accelerator.backward(loss)
            return loss 

        loss_1 = step() # //Backward 

        lambda_ = self.args.lambda_ # fix value
        if self.sam_scheduler_type == "linear":
            lambda_ = (self.state.epoch / self.args.num_train_epochs) * self.args.lambda_

        chance = random.random()
        if chance <= (1 - lambda_): 
            if self.update_p:
                self.update_p_with_loss(loss_1.detach().cpu().item(), is_forget_group=is_forget_group) 
            self.update_source_perturbed_loss(loss_1, None, is_forget_group)
            self.update_group_ema_loss(loss_1, is_forget_group)
            if is_forget_group:
                self.cur_losses = [0, loss_1.detach().cpu().item()]
            else:
                self.cur_losses = [loss_1.detach().cpu().item(), 0]
            if self.state.global_step % self.print_freq == 0:
                self.print_state(is_forget_group)

            return loss_1.detach() / self.args.gradient_accumulation_steps

        with torch.no_grad():
            grad_list = [p.grad for p in model.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([torch.norm(g.to('cuda:0'), p=2) for g in grad_list]), p=2) 
            scale = self.args.rho / (grad_norm + 1e-7)
            scale = scale.item()

            param_backup = {} 
            for p in model.parameters():
                if p.grad is not None:
                    param_backup[p] = {
                        'param': p.data.clone(),  
                    }
                    e_r = p.grad.data * scale
                    p.data.add_(e_r) 

        self.model.zero_grad()
        loss = step() # //Backward 
        self.update_source_perturbed_loss(loss_1, loss, is_forget_group)
        self.update_group_ema_loss(loss, is_forget_group)

        if self.update_p:
            self.update_p_with_loss(loss.detach().cpu().item(), is_forget_group=is_forget_group) 
        if is_forget_group:
            self.cur_losses = [0, loss.detach().cpu().item()]
        else:
            self.cur_losses = [loss.detach().cpu().item(), 0]
        if self.state.global_step % self.print_freq == 0:
            self.print_state(is_forget_group)

        with torch.no_grad():
            for p in model.parameters():
                p.data.copy_(param_backup[p]['param'])
        
        return loss.detach() / self.args.gradient_accumulation_steps

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        self.visualizer.plot(self.args.output_dir) 

    def print_state(self, is_forget_group=True):
        print('-'*50)
        print(f"Epoch: {self.state.epoch:.2f}, Step: {self.state.global_step}, Forget_group: {is_forget_group}")
        print(f"Group Steps: [{self.group_steps[0]}, {self.group_steps[1]}]")
        print(f"Sampling Prob p: {self.p:.3f}")
        print(f"Current Losses: [{self.cur_losses[0]:.4f}, {self.cur_losses[1]:.4f}]")
        print(f"EMA Losses: [{self.group_ema_losses[0]:.4f}, {self.group_ema_losses[1]:.4f}]")
        print(f"Source Losses: [{self.group_source_losses[0]:.4f}, {self.group_source_losses[1]:.4f}]")
        print(f"Perturbed Losses: [{self.group_perturbed_losses[0]:.4f}, {self.group_perturbed_losses[1]:.4f}]\n")

        self.visualizer.add_record(
            self.state.epoch,
            self.p,
            self.group_ema_losses,
            self.group_steps,
            self.group_source_losses,
            self.group_perturbed_losses
        )


class TrainingVisualizer:
    def __init__(self):
        self.epochs = []
        self.ps = []
        self.ema_losses_0 = []
        self.ema_losses_1 = []
        self.steps_0 = []
        self.steps_1 = []
        self.source_losses_0 = []
        self.source_losses_1 = []
        self.perturbed_losses_0 = []
        self.perturbed_losses_1 = []
        
    def add_record(self, epoch, p, ema_losses, group_steps, source_losses, perturbed_losses):
        self.epochs.append(epoch)
        self.ps.append(p)
        self.ema_losses_0.append(ema_losses[0])
        self.ema_losses_1.append(ema_losses[1])
        self.steps_0.append(group_steps[0])
        self.steps_1.append(group_steps[1])
        self.source_losses_0.append(source_losses[0])
        self.source_losses_1.append(source_losses[1])
        self.perturbed_losses_0.append(perturbed_losses[0])
        self.perturbed_losses_1.append(perturbed_losses[1])
    
    def plot(self, save_dir):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16))
        
        # 1. Sampling probability
        ax1.plot(self.epochs, self.ps, 'b-', label='p')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Probability')
        ax1.grid(True)
        ax1.legend()
        
        # 2. Group steps
        ax2.plot(self.epochs, self.steps_0, 'r-', label='Group 0')
        ax2.plot(self.epochs, self.steps_1, 'g-', label='Group 1')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
        ax2.legend()
        
        # 3. EMA losses
        ax3.plot(self.epochs, self.ema_losses_0, 'r-', label='Group 0')
        ax3.plot(self.epochs, self.ema_losses_1, 'g-', label='Group 1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('EMA Loss')
        ax3.grid(True)
        ax3.legend()
        
        # 4. Source/Perturbed losses in one plot
        ax4.plot(self.epochs, self.source_losses_0, 'r-', label='Source G0')
        ax4.plot(self.epochs, self.source_losses_1, 'g-', label='Source G1')
        ax4.plot(self.epochs, self.perturbed_losses_0, 'r--', label='Perturbed G0')
        ax4.plot(self.epochs, self.perturbed_losses_1, 'g--', label='Perturbed G1')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_curves.png')
        print ('Save plot to ', f'{save_dir}/training_curves.png')
        plt.close()


class BoosterAlignmentTrainer(Trainer):
    def get_harmful_dataloader(self,harmful_datast) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
     
        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
        LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
  
        sampler = RandomSampler(harmful_datast)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(harmful_datast, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(harmful_datast, **dataloader_params))
    
    def init(self,  harmful_datast):
        self.clock = 0
        self.steps = 0
        self.harmful_dataloader = self.get_harmful_dataloader(harmful_datast)
        self.harmful_data_iter = iter(self.harmful_dataloader)
        self.statistic = 0

    def sample_from_harmful(self):
        # Get a  batch
        try:
            batch = next(self.harmful_data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.harmful_data_iter = iter(self.harmful_dataloader)
            batch = next(self.harmful_data_iter)
        return batch
    
    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            output[0].data =output[0] + perturbation
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        
        # print(self.sam_state["gradient"].items())
        grad_norm = self._grad_norm(self.sam_state["gradient"])
        for module in self.sam_state["gradient"]:
            grad = self.sam_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.sam_state["gradient"][module] = e_r.detach().clone()
   
    @torch.no_grad()
    def after_second_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        target_device = next(iter(poison_grads_representation.values())).device
        
        norm = torch.norm(
                torch.stack([
                    ( poison_grads_representation[name].to(target_device) ).norm(p=2)
                    for name in poison_grads_representation
                ]),
                p=2
            )
        return norm
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        # may change input due to mode change
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.sample_from_harmful()
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        def step():
            # first backward gradient for harmful dataset    
            with self.compute_loss_context_manager():
                loss =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            stored_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if param.requires_grad}
            model.zero_grad()
            
            # Take step with the harmful perturbation
            with torch.no_grad():
                grad_norm = self._grad_norm(stored_grads) + 1e-7
                
            # perturb the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    grad = stored_grads[name]
                    param.data += self.args.alpha * grad / grad_norm.to(grad.device)
          
            # backward the gradient after harmful perturbation
            with self.compute_loss_context_manager():
                loss2 =  self.compute_loss(model, harmful_inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            perturb_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
            model.zero_grad()
            
            # recover the weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += self.args.alpha*stored_grads[name]/grad_norm.to(param.device)
              
            if False:
                self.sam_state = {}
                self.sam_state ["hooks"] = []
                self.sam_state ["gradient"] = {}
                # do forward backward on safety data
                self.pre_first_step(model)
                # first backward
                loss4 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss4, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss4)
                self.after_first_step(model)
                model.zero_grad()
                self.pre_second_step(model)
                loss3 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                # cancel the perturbation
                self.after_second_step(model)
                # sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        # param.grad.data=param.grad.data - (self.args.alpha +self.args.lamb/self.args.rho)*stored_grads[name] +self.args.lamb/self.args.rho* perturb_grads[name]
                        param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name] -self.args.lamb* perturb_grads[name]
                        
                self.steps+=1
                if self.steps%500==0:
                    self.statistic=0
                    self.statistic += sum([torch.norm(stored_grads[name])**2 for name, param in model.named_parameters() if param.requires_grad ]).detach()
                    print("harmful gradient norm {}".format(self.statistic),flush=True)
                    print("harmful loss {}".format(loss),flush=True)
                return loss3
            else:
                with self.compute_loss_context_manager():
                    loss3 =  self.compute_loss(model, inputs)
                if self.use_apex:
                    with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    self.accelerator.backward(loss3)
                    
                # Finally, sum the grad
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad.data=param.grad.data  + (self.args.lamb)*stored_grads[name] -self.args.lamb* perturb_grads[name]
        
                self.steps+=1
                if self.steps%2000==0 :
                    self.statistic=0
                    self.statistic += grad_norm.detach()
                    print("harmful gradient norm {}".format(self.statistic),flush=True)
                    print("loss change {}".format(loss-loss2),flush=True)
                    print("harmful loss {}".format(loss),flush=True)
            return loss3
        
        loss = step()   
        return loss.detach() / self.args.gradient_accumulation_steps

    
class RepNoiseTrainer(Trainer):
    def init(self,  harmful_dataset):
        # reploss needs standard dataset, load alpaca here
        from transformers.trainer_utils import ( seed_worker)
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        self.harmful_dataloader = self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))
        
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], *args, **kwargs
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Get an iterator from the DataLoader
        data_iter = iter(self.harmful_dataloader)
        # Get the next batch
        harmful_inputs = next(data_iter)
        harmful_inputs = self._prepare_inputs(harmful_inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                # loss = self.compute_loss(model, inputs)
                loss = rep_noise_loss(model,harmful_inputs,inputs, beta = self.args.lamb, alpha = self.args.rho)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 

        loss = step()

        return loss.detach() / self.args.gradient_accumulation_steps
