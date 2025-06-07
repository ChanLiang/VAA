#!/bin/bash

#!/bin/bash

# Env variables
date=TODAY
project_dir=$YOUR_PROJECT_DIR
cache_dir=$YOUR_CACHE_DIR

# Model and data paths
base_model=Llama-2-7b-hf
model_path=${cache_dir}/${base_model}
data_path=PKU-Alignment/BeaverTails
group_path=./grouping/vulnerable.json

# Training parameters
epoch=5
save_total_limit=$epoch
train_num=2000
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
optimizer=dro_sampling
# Uncomment to use alternative methods:
# optimizer=baseline
# optimizer=vaccine

# Hyperparameters
rho=0.4
lambda=1.0
p_step_size=0.007

sam_scheduler_type=linear
update_p=True
filter_noise=False
uniform_sampling=True

# Experiment ID
if [[ "$optimizer" == "sam" || "$optimizer" == "dro_sampling" || "$optimizer" == "vaccine" ]]; then
    perturbation_params="${sam_scheduler_type}L${lambda}R${rho}"
    exp_name=${optimizer}${perturbation_params}_updateP${update_p}_filterNoise${filter_noise}_${base_model}_us${uniform_sampling}_train${train_num}_bz${train_bz}_lr${lr}_plr${p_step_size}_epoch${epoch}
else
    exp_name=${optimizer}_${base_model}_train${train_num}_bz${train_bz}_lr${lr}_epoch${epoch}
fi

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

python -u train.py \
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

# (2) Evaluation
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

evaluate "${prediction_path}/beavertails/unsafe" "${project_dir}/log/${date}_${exp_name}_eval_unsafe.log" $gpu_id &


# (3) Fine-tuning
cd /home/aiscuser/align/script/alignment
gpu_id_1=$(echo $gpu_id | cut -d',' -f1)  
gpu_id_2=$(echo $gpu_id | cut -d',' -f2) 

poison=0.1
step=1250 
ft_lr=1e-5
ckp_path=${date}_${exp_name}

bash ft_script.sh sst2 $poison $ckp_path $step False $gpu_id_1 5 $ft_lr &
bash ft_script.sh gsm8k $poison $ckp_path $step False $gpu_id_2 5 $ft_lr & 
wait
bash ft_script.sh ag_news $poison $ckp_path $step False $gpu_id_1 5 $ft_lr & 
bash ft_script.sh alpacaEval $poison $ckp_path $step False $gpu_id_2 5 $ft_lr &
wait

# (4) Log results
ft_epoch=5
ckp_path=${date}_${exp_name}

unsafe_output="${project_dir}/log/${date}_${exp_name}_eval_ft_unsafe.log"
ft_output="${project_dir}/log/${date}_${exp_name}_eval_ft.log"

# read unsafe results and redirect to file
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

# read ft results and redirect to file
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