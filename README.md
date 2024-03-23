# RecLM-gen
## Introduction
`RecLM-gen` is a repo used to fine-tune LLMs to align LLMs for controllable recommendation tasks. Additionally, it gives the simple implementation of supervised fine-tune(SFT) and reinforcement learning(RL) on LLMs. It is scalable for users to fine-tune LLMs on other domains and highly customize the training processes.
This repo is built upon the [`transformers`](https://github.com/huggingface/transformers) lib.

Next, we provide an example of the fine-tuning process with two training stages on recommendation task. 

## Raw dataset format
Raw dataset should have 3 files in data_path at least: `category.pickle`, `meta.pickle`, `sequential.pickle`.

`ranking_candidate.pickle` is needed, if you need to test reranking task.

**Total dataset is available on this [link](https://drive.google.com/file/d/1cfw-KSqEwGF0eB_hm1PUWhUTdloT04Le/view?usp=drive_link).**

### category.pickle
`category.pickle` is a dict, the keys are all categories, and value is the item list belonging specific category.
```json
{
  "category_1": ["item_id_1", "..."], 
  "category_2": ["item_id_i", "..."], 
  "...": "...",
  "category_k": ["item_id_j", "..."]
}
```
### meta.pickle
`meta.pickle` is a dict, the keys are all item_ids, and value is the information(including one type of item index at least, such as `title`) of specific item.
```json
{
  "item_id_1": {"title": "..."},
  "item_id_2": {"title": "..."},
  "...": "...",
  "item_id_n": {"title": "..."}
}
```

### sequential.pickle
`sequential.pickle` is a dict, the keys are all user_ids, and value is the history(time-dependent order) of specific user.
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_x"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_y"]
}
```

### ranking_candidate.pickle (needed for testing reranking task)
`ranking_candidate.pickle` is a dict, the keys are all user_ids, and value is the list with 100 negative samples, which are random chosen.
```json
{
  "user_id_1": ["item_id_1", "...", "item_id_100"],
  "...": "...",
  "user_id_m": ["item_id_1", "...", "item_id_100"]
}
```


## 1. SASRec Server
We use [UniRec](https://github.com/microsoft/UniRec) lib to implement SASRec teacher model and deploy serve.

### 1.1. Install UniRec
```shell
git clone https://github.com/microsoft/UniRec.git
pip install --user --upgrade setuptools wheel twine
```
Change the dependency of unirec in `unirec/setup.py`
```
install_requires = [
    "torch>=1.10.0,<=1.13.1" -> "torch>=1.10.0,<=2.1.2",
    "..."
]
```

```shell
cd UniRec
python setup.py sdist bdist_wheel
pip install dist/unirec-*.whl 
```

### 1.2. SASRec dataset and model
Model param is saved in `unirec/output/`.

Dataset files `train.pkl`, `valid.pkl`, `test.pkl`, `user_history.pkl`,  `map.pkl`, `category.pickle(as same as raw dataset)` in `unirec/data/sub_movie/`.

`train.pkl`, `valid.pkl`, `test.pkl`, `user_history.pkl` is used to train SASRec model in UniRec lib. (from the same data source)

### 1.3. SASRec model training
The params is dataset name(`sub_movie`).
```shell
./scripts/unirec_train.sh sub_movie
```

### 1.4. SASRec Server start
Change the `model_path` in `unirec/asyc_server.py` to indicate the path of file.
```python
model_path = {
    'sub_movie': "unirec/output/sub_movie/SASRec/train/checkpoint_2024-03-17_014803_35/SASRec-SASRec-sub_movie.pth",
    'steam': "unirec/output/steam/SASRec/train/checkpoint_2024-03-17_014033_93/SASRec-SASRec-steam.pth",
}
```

The params is dataset name(`sub_movie`), serve port(`12621`), workers number(`1`) respectively.
```shell
./scripts/unirec_serve.sh sub_movie 12621 1
```
For dataset preparing, the workers number should be bigger for lifting speed, such as `4`.


## 2. SFT stage

### 2.1. Dataset format
For SFT, the type of dataset is `List[List[Dict]]`.

The `i-th` `List[Dict]` is the train data of the `i-th` epoch.

Each `Dict` is a train sample, which has key `"input_text"` and `"output_text"` at least for traditional SFT.

`"task"` and `"input_field_data"` is used to compute metrics for the specific domain.
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

### 2.2. Dataset prepare
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

### 2.3. SFT train
Train dataset is dynamic generated during `__getitem__` function of dataset class.
The example script is available at [scripts/sft_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/sft_train.sh).


If you want to use a static dataset, please set `--train_data_file` and `--val_data_file` command param.
```shell
  --train_data_file data/dataset/sub_movie/SFT_dataset_train.pickle 
  --val_data_file data/dataset/sub_movie/SFT_dataset_val.pickle 
```

`RecLM-gen` supports single gpu in SFT training. The example script is available at [scripts/single_gpu_sft_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/single_gpu_sft_train.sh).

### 2.4. SFT merge
The example script is available at [scripts/sft_merge.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/sft_merge.sh).
The merged model will be saved in `snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/`

**Use `CUDA_VISIBLE_DEVICES=x` to select gpu, do not set the `--gpu` command param**



## 2. RL stage

### 2.1. Dataset format
For RL, the type of dataset is also `List[List[Dict]]`.

The `i-th` `List[Dict]` is the train data of the `i-th` episode.

Each `Dict` is a train sample, which has key `'input_text'` at least for RL.

`task` and `input_field_data` is used to compute metrics and reward for the specific domain.
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

### 2.2. Dataset prepare
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


### 2.3. RL train
Train dataset is dynamic generated during `__getitem__` function of dataset class.
The example script is available at [scripts/rl_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/rl_train.sh).

If you want to use a static dataset, please set `--train_data_file` and `--val_data_file` command param.
```shell
  --train_data_file data/dataset/sub_movie/RL_dataset_train.pickle 
  --val_data_file data/dataset/sub_movie/RL_dataset_val.pickle 
```

`RecLM-gen` supports single gpu in RL training. The example script is available at [scripts/single_gpu_rl_train.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/single_gpu_rl_train.sh).


### 2.4. RL merge
The example script is available at [scripts/rl_merge.sh](https://github.com/Luuuk12321/RecLM-gen/blob/main/scripts/rl_merge.sh).

The merged model will be saved in `snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RLHF_Step7000/`


## 3. Test stage

### 3.1. VLLM deploy
```shell
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server --port 13579 --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RLHF_Step7000/
```

### 3.2. VLLM test
```shell
./scripts/tasks_test.sh snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ 13579
./scripts/tasks_test.sh snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RL_Step7000/ 13579
```