#!/bin/bash
date=TODAY

project_dir=$YOUR_PROJECT_DIR
cache_dir=$YOUR_CACHE_DIR

base_model=Qwen2.5-7B

model_path=${cache_dir}/${base_model}
echo "The base model is: $base_model"

data_path=PKU-Alignment/BeaverTails

group_eval=False
eval_on_train=False

train_num=1000
eval_num=300
test_num=500 
eval_steps=50
test_steps=-1 
decode_on_begin=False 
decode_on_end=False
# decode_on_epoch=True
decode_on_epoch=False

train_bz=8
grad_accum=1

eval_bz=32
infer_bz=64
infer_max_seq_length=100

generalization_adjustment="0.0"
p_normalize_loss=False

optimizer=booster

save_steps=40
save_strategy="epoch"
evaluation_strategy="epoch"

# uniform_sampling=True
uniform_sampling=False
use_lora=False

lr_scheduler=constant
wd=0.1

min_var_weight=0.0 

use_ema_loss=False 
bias_correction=False

p_step_size=0.1

sam_scheduler_type=constant

epoch=5
save_total_limit=$epoch

# lr=5e-5
lr=1e-4

lambda=1
rho=0.001


# update_p=False # for baseline analysis
update_p=True 

filter_noise=False

perturbation_params="L${lambda}R${rho}"

exp_name=${optimizer}${perturbation_params}_${base_model}_train${train_num}_bz${train_bz}_lr${lr}_epoch${epoch}
output_dir=${project_dir}/ckpt/${date}_${exp_name}
prediction_path=${project_dir}/pred/${date}_${exp_name}
log_file=${project_dir}/log/${date}_${exp_name}.log
error_log=${project_dir}/log/${date}_${exp_name}_error.log

echo "exp_name: $exp_name"
echo "log_file: $log_file"

cd ../../                

gpu_id=0,1
export CUDA_VISIBLE_DEVICES=$gpu_id

python -u train.py \
--lamb ${lambda} \
--alpha ${rho} \
--model_name_or_path ${model_path}  \
--group_names "Unforget" "Forget" \
--stage align \
--update_p $update_p \
--lambda_ ${lambda} \
--rho $rho \
--sam_scheduler_type $sam_scheduler_type \
--group_eval $group_eval \
--p_step_size $p_step_size \
--generalization_adjustment $generalization_adjustment \
--normalize_loss $p_normalize_loss \
--use_ema_loss $use_ema_loss \
--bias_correction $bias_correction \
--min_var_weight $min_var_weight \
--eval_on_train $eval_on_train \
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
--eval_steps $eval_steps \
--train_num $train_num \
--eval_num $eval_num \
--test_num $test_num \
--save_strategy $save_strategy \
--save_steps $save_steps \
--save_total_limit $save_total_limit \
--test_steps $test_steps \
--decode_on_begin $decode_on_begin \
--decode_on_end $decode_on_end \
--decode_on_epoch $decode_on_epoch \
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
2> "${error_log}" 1> "${log_file}"

wait

sync    
cd poison/evaluation  
echo "Start to evaluate the prediction"
infer_bz=64

evaluate() {
    local dir=$1
    local log_file=$2
    local cuda_device=$3

    export CUDA_VISIBLE_DEVICES=$cuda_device

    for file in ${dir}/*.json; do
        if [[ $file =~ _sentiment_eval\.json$ || -f "${file}_sentiment_eval.json" ]]; then
            continue
        fi

        echo "prediction_file: $file"
        echo -e "\n\n================ Evaluating $(basename $file) ================\n" >> ${log_file}

        python -u eval_sentiment.py \
            --model_path ${cache_dir}/beaver-dam-7b \
            --input_path $file \
            --batch_size $infer_bz \
            >> "${log_file}" 2>&1
        wait
    done
}

gpu_id_1=$(echo $gpu_id | cut -d',' -f1)  
gpu_id_2=$(echo $gpu_id | cut -d',' -f2) 

evaluate "${prediction_path}/beavertails/unsafe" "${project_dir}/log/${date}_${exp_name}_eval_unsafe.log" $gpu_id &

wait

