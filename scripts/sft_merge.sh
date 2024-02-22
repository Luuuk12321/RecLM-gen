#!/bin/bash

python main.py \
  --backbone snap/Llama-2-7b-hf-chat/ \
  --gpu cuda:0 \
  --train_stage SFT_Merge \
  --SFT_actor_lora_r 16 \
  --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ \
  --SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch37_SFT


CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --port 13579 \
  --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/


./scripts/tasks_test.sh snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ 13579
