#!/bin/bash


# --SFT_load is the model parameter file saved in SFT: snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch37_SFT.pth
CUDA_VISIBLE_DEVICES=0 python main.py \
  --backbone snap/Llama-2-7b-hf-chat/ \
  --train_stage SFT_Merge \
  --SFT_actor_lora_r 16 \
  --SFT_actor_lora_a 8 \
  --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ \
  --SFT_load snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch37_SFT
