import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
import torch
import os
from typing import Dict, Sequence

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

access_token = next(open('huggingface_token.txt')).strip()


def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer, model: transformers.PreTrainedModel):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_model(args, training_args):
    # 1. load tokenizer and model
    tokenizer = None
    if "Llama-2-7b-hf" in args.model_name_or_path and "lr" in args.model_name_or_path:
        print ('load llama tokenizer!')
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "/cache/Llama-2-7b-hf",
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True  
        )
    elif "gemma" in args.model_name_or_path:
        print ('load gemma tokenizer!')
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            "/cache/gemma-2-9b/",
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True  
        )
    elif "Qwen" in args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/cache/Qwen2.5-7B/",
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
            local_files_only=True 
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            token = access_token
        )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN


    print ('load model from:', args.model_name_or_path)
    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    if "Llama-2-7b-hf" in args.model_name_or_path and "lr" in args.model_name_or_path:
        config.vocab_size = 32001
        print ('len(tokenizer) =', len(tokenizer))
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        device_map="auto",
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    rank, lora_alpha = training_args.rank, training_args.lora_alpha
    if args.use_lora == True:
        if args.stage == 'ft': # FT stage
            print("FT stage: recover LoRA weights...")
            assert args.lora_folder != "", "Please specify the lora_folder for loading LoRA weights."

            model = PeftModel.from_pretrained(
                model,
                args.lora_folder,
                is_trainable=False
            )
            model = model.merge_and_unload()

            config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        else: # Alignment stage
            assert args.stage == 'align', "Please specify the stage as 'align' or 'ft'."
            print("Align: initialize Lora weights..")

            config = LoraConfig(
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], 
                lora_dropout=0.1, 
                bias="none",
                task_type="CAUSAL_LM",
            )
            # initialize the model with the LoRA framework
            model = get_peft_model(model, config)

    model.train()
    if args.use_lora == True:
        print('trainable_parameters = \n', model.print_trainable_parameters())

    for name, param in model.named_parameters():
        if param.device.type == 'cpu':
            print(f"\nParameter {name} is on CPU!!!\n")
        if param.device.type == 'meta':
            print(f"\nParameter {name} is on meta device!!!\n")

    return tokenizer, model

