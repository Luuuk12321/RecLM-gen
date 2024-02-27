#!/bin/bash

python main.py \
  --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/ \
  --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ \
  --gpu cuda:0 \
  --train_stage RLHF_Merge \
  --RLHF_actor_lora_r 4 \
  --RLHF_actor_lora_a 2 \
  --RLHF_critic_lora_r 4 \
  --RLHF_critic_lora_a 2 \
  --RLHF_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/7000step_RLHF \
  --lm_head \
  --FA2


CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --port 13579 \
  --model snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RLHF_Step7000/RLHF_Step7000/


./scripts/tasks_test.sh snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RL_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_RA_0.5_/RLHF_Step7000/ 13579

