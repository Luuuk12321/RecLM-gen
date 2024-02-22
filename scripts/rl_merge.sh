#!/bin/bash

python main.py \
  --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.5_/ \
  --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ \
  --item_index title64_t \
  --gpu cuda:2 \
  --train_stage RLHF_Merge \
  --RLHF_actor_lora_r 4 \
  --RLHF_critic_lora_r 4 \
  --RLHF_load snap/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/7000step_RLHF \
  --lm_head


CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
  --port 13579 \
  --model snap/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/RLHF_Step7000/


./scripts/tasks_test.sh snap/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-20.1_SN-2_Q-False_T6_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAMMA-0.99_GAS-4_LB-1_ND-False_RA_0.2_/RLHF_Step7000/ 13579

