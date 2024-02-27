# Raw dataset format
Raw dataset should have 3 files in data_path at least: `category.pickle`, `meta.pickle`, `sequential.pickle`.

`ranking_candidate.pickle` is needed, if you need to test reranking task.

## category.pickle
`category.pickle` is a dict, the keys are all categories, and value is the item list belonging specific category.
```json
{
  "category_1": ["item_id_1", "..."], 
  "category_2": ["item_id_i", "..."], 
  "...": "...",
  "category_k": ["item_id_j", "..."]
}
```
## meta.pickle
`meta.pickle` is a dict, the keys are all item_ids, and value is the information(including one type of item index at least, such as `title`) of specific item.
```json
{
  "item_id_1": {"title": "..."},
  "item_id_2": {"title": "..."},
  "...": "...",
  "item_id_n": {"title": "..."}
}
```

## sequential.pickle
`sequential.pickle` is a dict, the keys are all user_ids, and value is the history(time-dependent order) of specific user.
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_x"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_y"]
}
```

## ranking_candidate.pickle (needed for testing reranking task)
`ranking_candidate.pickle` is a dict, the keys are all user_ids, and value is the list with 100 negative samples, which are random chosen.
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_100"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_100"]
}
```


# 1. SASRec Server

## 1.1. SASRec dataset and model
Model `sub_movie.pth` in `TeacherModel/saved/`.

Dataset files `sub_movie.inter`, `sub_movie.item`, `category.pickle(same as raw dataset)` in `TeacherModel/dataset/sub_movie/`.

## 1.2. SASRec Server start
The params is dataset name(`sub_movie`), serve port(`12621`), gpu_id(`0`), workers number(`1`) respectively.

For dataset prepare, the workers number should be bigger, such as `4`.
```shell
cd TeacherModel/
python acil.py sub_movie 12621 0 1
```

# 2. SFT stage

## 2.1. Dataset format
For SFT, the type of dataset is `List[List[Dict]]`.

The `i-th` `List[Dict]` is the train data of the `i-th` epoch.

Each `Dict` is a train sample, which has key `"input_text"` and `"output_text"` at least for traditional SFT.

`"task"` and `"input_field_data"` is used to compute metrics.
```js
[
  [ //Epoch 1
    {"input_text": "...", "output_text": "...", "task": "...", "input_field_data": {"...": "..."}},
    "...",
    {"input_text": "...", "output_text": "...", "task": "...", "input_field_data": {"...": "..."}}
  ],
  [ //Epoch 2
    "..."
  ]
]
```

## 2.2. Dataset prepare
The dataset file is saved to `{data_path}/SFT_dataset_train.pickle` and `{data_path}/SFT_dataset_val.pickle`.
```shell
python data_process.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--item_index title64_t 
--topk 10 
--epoch 10 
--train_stage SFT 
--SFT_train_tasks SFTSeqRec,SFTPersonalControlRec,SFTControlRec_re,SFTPersonalCategoryRate,ShareChatGPT 
--SFT_val_tasks SFTTestSeqRec,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateEP_50 
--backup_ip 0.0.0.0 
--share_chat_gpt_ratio 0.5 
--val_num_per_task 320 
--llama2_chat_template 
--idx 
--teacher_port 12621 
```

## 2.3. SFT train
Train dataset is dynamic generated during `__getitem__` function of dataset class.

**Note: Don't set `--gpu` and add `--distributed` command param while using accelerate to launch. It will be set automatically.**
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --gpu_ids all main.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ 
--backbone snap/Llama-2-7b-hf-chat/ 
--item_index title64_t --batch_size 1 
--topk 10 --clip_grad_norm 1.0 
--epoch 40 
--gen_max_length 512 
--lr 0.001 
--gradient_accumulation_steps 16 
--train_stage SFT 
--SFT_actor_lora_r 16 
--SFT_actor_lora_a 8 
--warmup_ratio 0.0125 
--val_batch_size 16 
--SFT_train_tasks SFTSeqRec,SFTPersonalControlRec,SFTControlRec_re,SFTPersonalCategoryRate,ShareChatGPT 
--SFT_val_tasks SFTTestSeqRec,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateEP_50 
--backup_ip 0.0.0.0 
--val_epoch 0 
--share_chat_gpt_ratio 0.5 
--FA2 
--llama2_chat_template 
--idx
--distributed 
```

If you want to use a static dataset, please set `--train_data_file` and `--val_data_file` command param.
```shell
# train data is from .
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 4 --gpu_ids all main.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ 
--backbone snap/Llama-2-7b-hf-chat/ 
--item_index title64_t --batch_size 1 
--topk 10 --clip_grad_norm 1.0 
--epoch 40 
--gen_max_length 512 
--lr 0.001 
--gradient_accumulation_steps 16 
--train_stage SFT 
--SFT_actor_lora_r 16 
--SFT_actor_lora_a 8 
--warmup_ratio 0.0125 
--val_batch_size 16 
--SFT_train_tasks SFTSeqRec,SFTPersonalControlRec,SFTControlRec_re,SFTPersonalCategoryRate,ShareChatGPT 
--SFT_val_tasks SFTTestSeqRec,SFT+TestPersonalControlRec,SFT-TestPersonalControlRec,SFTTestPersonalCategoryRateEP_50 
--backup_ip 0.0.0.0 
--val_epoch 0 
--share_chat_gpt_ratio 0.5 
--FA2 
--llama2_chat_template 
--idx 
--distributed 
--train_data_file data/dataset/sub_movie/SFT_dataset_train.pickle 
--val_data_file data/dataset/sub_movie/SFT_dataset_val.pickle 
```

## 2.4. SFT merge
The merged model will be saved in `snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/`
```shell
python main.py 
--backbone snap/Llama-2-7b-hf-chat/ 
--gpu cuda:0 
--train_stage SFT_Merge 
--SFT_actor_lora_r 16 
--SFT_actor_lora_a 8 
--output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ 
--SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch37_SFT 
```


# 2. RL stage

## 2.1. Dataset format
For RL, the type of dataset is also `List[List[Dict]]`.

The `i-th` `List[Dict]` is the train data of the `i-th` episode.

Each `Dict` is a train sample, which has key `'input_text'` at least for RL.

`task` and `input_field_data` is used to compute metrics and reward.
```js
[
  [ //Episode 1
    {"input_text": "...", "task": "...", "input_field_data": {"...": "..."}},
    "...",
    {"input_text": "...", "task": "...", "input_field_data": {"...": "..."}}
  ],
  [ //Episode 2
    "..."
  ]
]
```

## 2.2. Dataset prepare
The dataset file is saved to `{data_path}/RL_dataset_train.pickle` and `{data_path}/RL_dataset_val.pickle`.
```shell
python data_process.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--item_index title64_t 
--topk 10 
--num_episodes 2 
--train_stage RL 
--RL_train_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP,RLPersonalCategoryRateMP,RLPersonalCategoryRateEP 
--RL_val_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP_20,RLPersonalCategoryRateMP_30,RLPersonalCategoryRateEP_50,RLItemCount 
--backup_ip 0.0.0.0 
--val_num_per_task 320 
--llama2_chat_template 
--idx 
--teacher_port 12621 
```


## 2.3. RL train
Train dataset is dynamic generated during `__getitem__` function of dataset class.
```shell
CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --num_processes 2 --gpu_ids all main.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ 
--backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ 
--item_index title64_t 
--batch_size 8 
--gradient_accumulation_steps 4 
--topk 10 
--clip_grad_norm 0.5 
--epoch 4 
--gen_max_length 512 
--train_stage RL 
--RL_actor_lora_r 4 
--RL_critic_lora_r 4 
--RL_train_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP,RLPersonalCategoryRateMP,RLPersonalCategoryRateEP 
--RL_val_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP_20,RLPersonalCategoryRateMP_30,RLPersonalCategoryRateEP_50,RLItemCount 
--backup_ip 0.0.0.0 
--lr 0.000005 
--lora_drop 0.0 
--weight_decay 0.0 
--kl_coef 0.3 
--entropy_weight 0.01 
--vf_coef 0.1 
--lm_head 
--policy_kl_threshold 0.05 
--idx 
--llama2_chat_template 
--FA2 
--lr_power 2.0 
--learn_batch 1 
--sample_num 2 
--whiten_reward 
--num_episodes 2 
--reward_alpha 0.5 
--fine_grain_reward
--distributed 
```

If you want to use a static dataset, please set `--train_data_file` and `--val_data_file` command param.
```shell
CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --num_processes 2 --gpu_ids all main.py 
--seed 0 
--data_path data/dataset/sub_movie/ 
--output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ 
--backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ 
--item_index title64_t 
--batch_size 8 
--gradient_accumulation_steps 4 
--topk 10 
--clip_grad_norm 0.5 
--epoch 4 
--gen_max_length 512 
--train_stage RL 
--RL_actor_lora_r 4 
--RL_critic_lora_r 4 
--RL_train_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP,RLPersonalCategoryRateMP,RLPersonalCategoryRateEP 
--RL_val_tasks RLSeqRec,RL+PersonalControlRec,RL-PersonalControlRec,RLPersonalCategoryRateLP_20,RLPersonalCategoryRateMP_30,RLPersonalCategoryRateEP_50 
--backup_ip 0.0.0.0 
--lr 0.000005 
--lora_drop 0.0 
--weight_decay 0.0 
--kl_coef 0.3 
--entropy_weight 0.01 
--vf_coef 0.1 
--lm_head 
--policy_kl_threshold 0.05 
--idx 
--llama2_chat_template 
--FA2 
--lr_power 2.0 
--learn_batch 1 
--sample_num 2 
--whiten_reward 
--num_episodes 2 
--reward_alpha 0.5 
--fine_grain_reward
--distributed
--train_data_file data/dataset/sub_movie/RL_dataset_train.pickle 
--val_data_file data/dataset/sub_movie/RL_dataset_val.pickle 
```

## 2.4. RL merge
The merged model will be saved in `'snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RLHF_Step7000/`
```shell
python main.py 
--output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/ 
--backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ 
--gpu cuda:0 
--train_stage RL_Merge 
--RL_actor_lora_r 4 
--RL_actor_lora_a 2 
--RL_critic_lora_r 4 
--RL_critic_lora_a 2 
--RL_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/7000step_RL 
--lm_head 
--FA2 
```

# 3. Test stage

## 3.1. VLLM deploy
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RLHF_Step7000/
```

## 3.2. VLLM test
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
./tasks_test.sh snap/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/RL_Step7000/ 13579
```