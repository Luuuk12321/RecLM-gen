## SASRec Server start
```shell
cd SASRec/
python cli.py --dataset sub_movie --port 12621
```

## SFT stage

### SFT train
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --gpu_ids all  main.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/Llama-2-7b-hf-chat/ --item_index title64_t --batch_size 1 --topk 10 --clip_grad_norm 1.0 --epoch 40 --gen_max_length 512 --lr 0.001 --gradient_accumulation_steps 16 --train_stage SFT --SFT_actor_lora_r 16 --SFT_actor_lora_a 8 --warmup_ratio 0.0125 --val_batch_size 16 --SFT_train_tasks SFTSeqRec --SFT_val_tasks SFTTestSeqRec,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateEP_50 --backup_ip 0.0.0.0 --val_epoch 0 --share_chat_gpt_ratio 0.0 --FA2 --llama2_chat_template --idx
```

### SFT merge
```shell
python main.py --backbone snap/Llama-2-7b-hf-chat/ --gpu cuda:0 --train_stage SFT_Merge --SFT_actor_lora_r 16 --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch37_SFT
```


## RLHF stage

### RLHF train
```shell
python main.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --item_index title64_t --batch_size 8 --gradient_accumulation_steps 4 --topk 10 --clip_grad_norm 0.5 --epoch 4 --gen_max_length 512 --train_stage RLHF --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_train_tasks RLHFSeqRec,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRateLP,RLHFPersonalCategoryRateMP,RLHFPersonalCategoryRateEP --backup_ip 0.0.0.0 --lr 0.000005 --lora_drop 0.0 --weight_decay 0.0 --kl_coef 0.3 --entropy_weight 0.01 --vf_coef 0.1 --lm_head --policy_kl_threshold 0.05 --idx --llama2_chat_template --FA2 --gpu cuda:2 --lr_power 2.0 --learn_batch 1 --sample_num 2 --whiten_reward --num_episodes 2 --reward_alpha 0.5 --fine_grain_reward
```

### RLHF val
```shell
accelerate launch --num_processes 2 --gpu_ids 6,9 main.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --item_index title64_t --val_batch_size 24 --topk 10 --gen_max_length 512 --train_stage RLHF --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_val_tasks RLHFSeqRec,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate,RLHFItemCount --backup_ip 0.0.0.0 --lr 0.000005 --lm_head --idx --llama2_chat_template --FA2 --num_episodes 0 --dry --model_name RLHF_ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.5_/
```

### RLHF merge
```shell
python main.py --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.5_/ --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --item_index title64_t --gpu cuda:2 --train_stage RLHF_Merge --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_load snap/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/7000step_RLHF --lm_head --FA2
```


### VLLM deploy
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/RLHF_Step7000/
```

### VLLM test
```shell
#python task_test.py --SFT_test_task SFTTestSeqRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
#python task_test.py --SFT_test_task SFTTestSeqRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 5
#python task_test.py --SFT_test_task SFTTestSeqRanking --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 5
#python task_test.py --SFT_test_task SFTTestSeqRanking --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 3
#python task_test.py --SFT_test_task SFT+TestPersonalControlRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
#python task_test.py --SFT_test_task SFT-TestPersonalControlRec --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
#python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_30% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
#python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_50% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
#python task_test.py --SFT_test_task SFTTestPersonalCategoryRate_70% --model_name snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --llama2_chat_template --idx --topk 10
./tasks_test.sh snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ 13579
./tasks_test.sh snap/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/RLHF_Step7000/ 13579
```