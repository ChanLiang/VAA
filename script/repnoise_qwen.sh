#!/bin/bash

# Env variables
date=TODAY
project_dir=$YOUR_PROJECT_DIR
cache_dir=$YOUR_CACHE_DIR

# Model and data paths
base_model=Qwen2.5-7B
model_path=${cache_dir}/${base_model}
data_path=PKU-Alignment/BeaverTails

# Training parameters
epoch=5
save_total_limit=$epoch
train_num=1000
eval_num=300
test_num=500
train_bz=8
grad_accum=1
eval_bz=32
infer_bz=64
infer_max_seq_length=100
lr=1e-4
wd=0.1
lr_scheduler=constant
save_steps=40
save_strategy="epoch"
evaluation_strategy="epoch"

# Method names
optimizer=repnoise

# Hyperparameters
rho=0.0001
lambda=0.1
p_step_size=0.1

sam_scheduler_type=constant
update_p=True
filter_noise=False
uniform_sampling=False
use_lora=False

# Experiment ID
perturbation_params="L${lambda}R${rho}"
exp_name=${optimizer}${perturbation_params}_${base_model}_train${train_num}_bz${train_bz}_lr${lr}_epoch${epoch}

# Output paths
output_dir=${project_dir}/ckpt/${date}_${exp_name}
prediction_path=${project_dir}/pred/${date}_${exp_name}
log_file=${project_dir}/log/${date}_${exp_name}.log
error_log=${project_dir}/log/${date}_${exp_name}_error.log

# Print to check
echo "exp_name: $exp_name"
echo "log_file: $log_file"

# (1) Training
cd ../../
export CUDA_VISIBLE_DEVICES=0,1

python -u train_icml.py \
--lamb ${lambda} \
--alpha ${rho} \
--model_name_or_path ${model_path} \
--group_names "Unforget" "Forget" \
--stage align \
--update_p $update_p \
--lambda_ ${lambda} \
--rho $rho \
--sam_scheduler_type $sam_scheduler_type \
--group_eval False \
--p_step_size $p_step_size \
--generalization_adjustment "0.0" \
--normalize_loss False \
--use_ema_loss False \
--bias_correction False \
--min_var_weight 0.0 \
--eval_on_train False \
--data_path $data_path \
--filter_noise $filter_noise \
--use_lora $use_lora \
--uniform_sampling $uniform_sampling \
--benign_dataset "" \
--bf16 True \
--num_train_epochs $epoch \
--per_device_train_batch_size $train_bz \
--per_device_eval_batch_size $eval_bz \
--infer_bz $infer_bz \
--infer_max_seq_length $infer_max_seq_length \
--gradient_accumulation_steps $grad_accum \
--evaluation_strategy $evaluation_strategy \
--eval_steps 50 \
--train_num $train_num \
--eval_num $eval_num \
--test_num $test_num \
--save_strategy $save_strategy \
--save_steps $save_steps \
--save_total_limit $save_total_limit \
--test_steps -1 \
--decode_on_begin False \
--decode_on_end False \
--decode_on_epoch False \
--learning_rate $lr \
--weight_decay $wd \
--warmup_ratio 0.1 \
--lr_scheduler_type ${lr_scheduler} \
--logging_steps 1 \
--tf32 True \
--cache_dir $cache_dir \
--output_dir $output_dir \
--prediction_path $prediction_path \
--optimizer ${optimizer} \
# 2> "${error_log}" 1> "${log_file}"

wait
sync

# (2) Fine-tuning
cd ~/align/script/
poison=0.1
step=625
ft_lr=1e-5
ckp_path=${date}_${exp_name}

bash ft_script.sh sst2 $poison $ckp_path $step False 0,1 5 $ft_lr &
wait