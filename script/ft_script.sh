#!/bin/bash

date=TODAY

project_dir=$YOUR_PROJECT_DIR
cache_dir=$YOUR_CACHE_DIR

task=${1}
poison_ratio=${2}
base_model=${3}
step=${4}
decode_on_begin=${5}
gpu=${6}
epoch=${7:-1} 
lr=${8:-1e-5} 

model_path=${project_dir}/ckpt/${base_model}/checkpoint-$step
echo "model_path: $model_path"

align_train_num=2000
train_num=1000
eval_num=200

test_num=-1 
safe_test_num=1
unsafe_test_num=500
train_test_num=1
ft_test_num=500 


train_bz=8
eval_bz=32
infer_bz=24
grad_accum=1

optimizer=baseline 
uniform_sampling=False
use_lora=False
lr_scheduler=constant
wd=0.1

save_steps=200
# save_strategy="epoch"
save_strategy="no"
save_total_limit=0

eval_steps=40 
evaluation_strategy="epoch"

test_steps=-1 

decode_on_end=False
decode_on_epoch=True 

stage=ft
data_path=PKU-Alignment/BeaverTails_dangerous

if [ "$task" = "alpacaEval" ]; then
    train_num=700
    ft_test_num=100
    eval_num=100
    # infer_bz=8
fi


echo "Running ${task} with epoch=$epoch, lr=$lr, poison_ratio=$poison_ratio"

benign_dataset=data/${task}.json

if [[ "$base_model" == *"baseline"* ]]; then
    decode_task_list="ft unsafe" 
    train_test_num=2000
else
    decode_task_list="ft unsafe"
fi

exp_name=ft${task}${poison_ratio}_train${train_num}_lr${lr}_bz${train_bz}_epo${epoch}_${base_model}
output_dir=${project_dir}/ckpt/${date}_${exp_name}
prediction_path=${project_dir}/pred/${date}_${exp_name}
log_file=${project_dir}/log/${date}_${exp_name}.log
error_log=${project_dir}/log/${date}_${exp_name}_error.log
echo "exp_name: ${date}_$exp_name"

cd ../../                          

export CUDA_VISIBLE_DEVICES=$gpu
python -u train.py \
--decode_task_list $decode_task_list \
--model_name_or_path ${model_path}  \
--stage ${stage} \
--task ${task} \
--data_path $data_path \
--use_lora $use_lora \
--uniform_sampling $uniform_sampling \
--benign_dataset ${benign_dataset} \
--poison_ratio $poison_ratio \
--bf16 True \
--num_train_epochs $epoch \
--per_device_train_batch_size $train_bz \
--per_device_eval_batch_size $eval_bz \
--infer_bz $infer_bz \
--gradient_accumulation_steps $grad_accum \
--evaluation_strategy $evaluation_strategy \
--eval_steps $eval_steps \
--align_train_num $align_train_num \
--train_num $train_num \
--eval_num $eval_num \
--test_num $test_num \
--safe_test_num $safe_test_num \
--unsafe_test_num $unsafe_test_num \
--train_test_num $train_test_num \
--ft_test_num $ft_test_num \
--test_steps $test_steps \
--decode_on_begin $decode_on_begin \
--decode_on_end $decode_on_end \
--decode_on_epoch $decode_on_epoch \
--save_strategy $save_strategy \
--save_steps $save_steps \
--save_total_limit $save_total_limit \
--learning_rate $lr \
--weight_decay $wd \
--warmup_ratio 0.1 \
--lr_scheduler_type ${lr_scheduler} \
--logging_steps 10 \
--tf32 True \
--cache_dir $cache_dir \
--output_dir $output_dir \
--prediction_path $prediction_path \
--optimizer ${optimizer} \
2> "${error_log}" 1> "${log_file}"

wait
sync     

if [ "$task" != "alpacaEval" ]; then
    for file in ${prediction_path}/${task}/*.json; do
        if [[ $file =~ _eval\.json$ || -f "${file}_eval.json" ]]; then
            continue
        fi

        echo "prediction_file: $file"
        echo -e "\n\n================ Evaluating $(basename $file) ================\n" >> ${log_file}

        python -u eval_benign.py \
            --input_path $file \
            --task ${task} \
            >> "${project_dir}/log/${date}_${exp_name}_eval_ft.log" 2>&1
        wait
    done
fi

cd poison/evaluation  
infer_bz=32

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

evaluate "${prediction_path}/beavertails/unsafe" "${project_dir}/log/${date}_${exp_name}_eval_unsafe.log" $gpu

wait

