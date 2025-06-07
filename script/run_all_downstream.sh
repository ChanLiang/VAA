

# (0) model path examples
# 1. baseline: 
# base_model=baseline_Llama-2-7b-hf_train2000_bz8_lr1e-4_epoch5 

# 2. vaccine R=2.0
# base_model=vaccineR2.0_updatePTrue_filterNoiseFalse_Llama-2-7b-hf_usFalse_train2000_bz8_lr1e-4_plr0.1_epoch5

# 3. booster L=1.0 R=0.001 
# base_model=boosterL1R0.001_Llama-2-7b-hf_train1000_bz8_lr1e-4_epoch5

# 4. repnoise L=0.1 R=0.001
# base_model=repnoiseL0.1R0.001_Llama-2-7b-hf_train1000_bz8_lr1e-4_epoch5

# 5. vaa L=1.0 R=0.4 plr=0.007
# base_model=dro_samplingconstantL1.0R0.4_updatePTrue_filterNoiseFalse_Llama-2-7b-hf_usFalse_train2000_bz8_lr1e-4_plr0.07_epoch3


# (1) harmful fine-tuning

step=1250 # for others
# step=625 # for booster
poison=0.1
gpu=0
# lr=3e-5
lr=1e-4
bash ft_script.sh sst2 $poison $base_model $step False 0 5 $lr &
bash ft_script.sh gsm8k $poison $base_model $step False 1 5 $lr & 
bash ft_script.sh ag_news $poison $base_model $step False 2 5 $lr & 
bash ft_script.sh alpacaEval $poison $base_model $step False 3 5 $lr &
wait

# (2) read
# ft_epoch=1
# ft_epoch=2
ft_epoch=5
for task in sst2 ag_news gsm8k alpacaEval; do
train_num=1000
if [ "$task" = "alpacaEval" ]; then
    train_num=700
fi

# unsafe score (HA)
echo $task
unsafe_log=log/ft${task}0.1_train${train_num}_lr1e-5_bz8_epo${ft_epoch}_${base_model}_eval_unsafe.log
echo -n "epoch: "
cat ${unsafe_log} | grep "Evaluating" | sed 's/.*epoch\([0-9]\).*/\1/' | tr '\n' '/' | sed 's/.$//'

echo -ne "\nHS: "
cat ${unsafe_log} | grep -A 1 "Predicting:" | grep -v "Predicting:" | awk '/^[0-9]/ {print $1}' | tr '\n' '/' | sed 's/.$//'
printf "\n\n"

# ft (ACC)
ft_log=log/ft${task}0.1_train${train_num}_lr1e-5_bz8_epo${ft_epoch}_${base_model}_eval_ft.log
echo $task
echo -n "epoch: "
cat ${ft_log} | grep "Evaluating" | sed 's/.*epoch\([0-9]\).*/\1/' | tr '\n' '/' | sed 's/.$//'

# extract score
echo -ne "\nHS: "
cat ${ft_log} | grep "^[0-9]" | awk '{print $1}' | tr '\n' '/' | sed 's/.$//'
printf "\n\n"

done

wait 

