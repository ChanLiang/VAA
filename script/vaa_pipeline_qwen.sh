#!/bin/bash
# date=124
# date=125
# date=127
# date=129
# date=130

date=327

project_dir=/mnt/teamdrive/projects/backup_chenlian/align/store # blob
cache_dir=/mnt/teamdrive/projects/backup_chenlian/cache # blob

# base_model=Llama-2-7b-hf
base_model=Qwen2.5-7B
# base_model=Qwen2.5-3B

# base_model=gemma-3-4b-pt

model_path=${cache_dir}/${base_model}
echo "The base model is: $base_model"

data_path=PKU-Alignment/BeaverTails
# group_path=/mnt/teamdrive/projects/backup_chenlian/align/store/pred/121_ft_Adate120_search_gsm8k0.1_train1000_lr3e-5_bz8_epoch1_alignbaseline_train2000_lr3e-5_bz8_epoch5/beavertails/train/predictions_num2000_epoch1_step125.json_sentiment_eval.json_flagged.json
group_path=/home/aiscuser/align/script/alignment/129_1e-4_1e-4_epoch4_forget.json
noise_path=/mnt/teamdrive/projects/backup_chenlian/align/store/pred/121_ft_Adate120_search_gsm8k0.1_train1000_lr3e-5_bz8_epoch1_alignbaseline_train2000_lr3e-5_bz8_epoch5/beavertails/train/predictions_num2000_epoch1_step125.json_sentiment_eval.json_forget.json

group_eval=False
eval_on_train=False

epoch=5
# epoch=3 # 10min training times
# epoch=1
# save_total_limit=0 # for debug
save_total_limit=$epoch

train_num=2000
eval_num=300
test_num=500 # 只解码 unsafe

eval_steps=50
test_steps=-1 # 只测开始和结束
decode_on_begin=False # 只测结束
decode_on_end=False
# decode_on_end=True # 提前预判
# decode_on_epoch=True
decode_on_epoch=False

train_bz=8
grad_accum=1

eval_bz=32
infer_bz=64
infer_max_seq_length=100

# lr=3e-5
lr=1e-4

generalization_adjustment="0.0"
p_normalize_loss=False

# optimizer=dro_sampling
# optimizer=baseline
optimizer=vaccine

save_steps=40
save_strategy="epoch"
evaluation_strategy="epoch"

use_lora=False

lr_scheduler=constant
wd=0.1

use_ema_loss=False # exp3是否使用ema loss
bias_correction=False

# min_var_weight=0.3 # 原始数据weight的比例
min_var_weight=0.0 # 不启用

uniform_sampling=True
# uniform_sampling=False
# for uniform_sampling in True False; do

# p_step_size=0.003
# p_step_size=0.01
# p_step_size=0.02
# p_step_size=0.03
# p_step_size=0.1
# for p_step_size in 0.03 0.01 0.3; do
# for p_step_size in 0.03 0.1; do
# for p_step_size in 0.03 0.04; do
# for p_step_size in 0.05 0.06; do
# for p_step_size in 0.07 0.08; do
# for p_step_size in 0.09; do

# for p_step_size in 0.07; do
for p_step_size in 0.04; do

# sam_scheduler_type=constant
sam_scheduler_type=linear

# lambda=0.0 # for debug (should be the same as baseline)
# lambda=0.1
# lambda=0.3
# lambda=0.5
# lambda=0.6
lambda=1.0
# for lambda in 0.1 0.3 0.9; do
# for p_step_size in 0.06 0.1 0.2; do

# gpu_id=0,1
gpu_id=2,3

# sleep 1h

# rho=0.5
# rho=0.3
# rho=0.1
# rho=0.03
# for rho in 0.3 0.1 0.03; do
# for rho in 0.1 0.03; do
# for rho in 0.05 0.2; do
# for rho in 0.7 0.5 0.9; do
# for rho in 0.4 0.5 0.6; do
# for rho in 0.3 0.4 0.5 0.6; do
# for rho in 0.4 0.5 0.6; do
# for rho in 0.4; do

# vaccine
# for epoch in 3 5 10; do # 20min, 
for rho in 2.0 5.0; do

# baseline
# for epoch in 5 10; do # 20min, 40min

# vaccine
# rho=1.0 

# update_p=False # for baseline analysis
update_p=True 

# filter_noise=True # for baseline analysis
filter_noise=False

if [ "$optimizer" = "sam" ] || [ "$optimizer" = "dro_sampling" ]; then
    perturbation_params="${sam_scheduler_type}L${lambda}R${rho}"
elif [ "$optimizer" = "vaccine" ]; then
    perturbation_params="R${rho}"
    lambda=0.0
else
    perturbation_params=""
    lambda=0.0
    rho=0.0
fi

if [ "$optimizer" = "sam" ] || [ "$optimizer" = "dro_sampling" ] || [ "$optimizer" = "vaccine" ]; then
    exp_name=${optimizer}${perturbation_params}_updateP${update_p}_filterNoise${filter_noise}_${base_model}_us${uniform_sampling}_train${train_num}_bz${train_bz}_lr${lr}_plr${p_step_size}_epoch${epoch}
else
    exp_name=${optimizer}_${base_model}_train${train_num}_bz${train_bz}_lr${lr}_epoch${epoch}
fi
output_dir=${project_dir}/ckpt/${date}_${exp_name}
prediction_path=${project_dir}/pred/${date}_${exp_name}
log_file=${project_dir}/log/${date}_${exp_name}.log
error_log=${project_dir}/log/${date}_${exp_name}_error.log

echo "exp_name: $exp_name"
echo "log_file: $log_file"

cd ../../                

export CUDA_VISIBLE_DEVICES=$gpu_id

python -u train_icml.py \
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
--group_path $group_path \
--noise_path $noise_path \
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
# 2> "${error_log}" 1> "${log_file}"

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

# evaluate "${prediction_path}/beavertails/safe" "${project_dir}/log/${date}_${exp_name}_eval_safe.log" $gpu_id_1 &
# evaluate "${prediction_path}/beavertails/unsafe" "${project_dir}/log/${date}_${exp_name}_eval_unsafe.log" $gpu_id_2 &
evaluate "${prediction_path}/beavertails/unsafe" "${project_dir}/log/${date}_${exp_name}_eval_unsafe.log" $gpu_id &


# run ft
cd /home/aiscuser/align/script/alignment
gpu_id_1=$(echo $gpu_id | cut -d',' -f1)  
gpu_id_2=$(echo $gpu_id | cut -d',' -f2) 

poison=0.1
step=1250 
ft_lr=1e-5
ckp_path=${date}_${exp_name}
# bash ft_script.sh sst2 $poison $ckp_path $step False $gpu_id_1 5 $ft_lr 

bash ft_script.sh sst2 $poison $ckp_path $step False $gpu_id_1 5 $ft_lr &
bash ft_script.sh gsm8k $poison $ckp_path $step False $gpu_id_2 5 $ft_lr & 
wait
bash ft_script.sh ag_news $poison $ckp_path $step False $gpu_id_1 5 $ft_lr & 
bash ft_script.sh alpacaEval $poison $ckp_path $step False $gpu_id_2 5 $ft_lr &
wait


ft_epoch=5
ckp_path=${date}_${exp_name}

unsafe_output="${project_dir}/log/${date}_${exp_name}_eval_ft_unsafe.log"
ft_output="${project_dir}/log/${date}_${exp_name}_eval_ft.log"

# (1) read unsafe results and redirect to file
{
for task in sst2 ag_news gsm8k alpacaEval; do
    ft_train_num=1000
    if [ "$task" = "alpacaEval" ]; then
        ft_train_num=700
    fi
    echo $task
    unsafe_log=/mnt/teamdrive/projects/backup_chenlian/align/store/log/${date}_ft${task}0.1_train${ft_train_num}_lr1e-5_bz8_epo${ft_epoch}_${ckp_path}_eval_unsafe.log
    echo -n "epoch: "
    cat ${unsafe_log} | grep "Evaluating" | sed 's/.*epoch\([0-9]\).*/\1/' | tr '\n' '/' | sed 's/.$//'

    echo -ne "\nHS: "
    cat ${unsafe_log} | grep -A 1 "Predicting:" | grep -v "Predicting:" | awk '/^[0-9]/ {print $1}' | tr '\n' '/' | sed 's/.$//'
    printf "\n\n"
done
} > "${unsafe_output}"

# (2) read ft results and redirect to file
{
for task in sst2 ag_news gsm8k; do
    ft_train_num=1000
    ft_log=/mnt/teamdrive/projects/backup_chenlian/align/store/log/${date}_ft${task}0.1_train${ft_train_num}_lr1e-5_bz8_epo${ft_epoch}_${ckp_path}_eval_ft.log
    echo $task 
    echo -n "epoch: "
    cat ${ft_log} | grep "Evaluating" | sed 's/.*epoch\([0-9]\).*/\1/' | tr '\n' '/' | sed 's/.$//'

    echo -ne "\nFA: "
    cat ${ft_log} | grep "^[0-9]" | awk '{print $1}' | tr '\n' '/' | sed 's/.$//'
    printf "\n\n"
done
} > "${ft_output}"


done 
done