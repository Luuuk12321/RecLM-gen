#!/bin/bash


# --SFT_load is the model parameter file saved in SFT: snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/Epoch37_SFT.pth
# need to keep the setting about model params as same as training, such as SFT_actor_lora_r and SFT_actor_lora_a.
# You need to ensure all saved parameters in file perfectly cover the trainable parameters of BaseModel.
CUDA_VISIBLE_DEVICES=1 python main.py \
  --backbone /home/lws/models/Llama-3-8B-instruct/ \
  --train_stage SFT_Merge \
  --SFT_actor_lora_r 16 \
  --SFT_actor_lora_a 8 \
  --output snap/ICR_Steam_TitleT_0_LCT_E40_gpt0.1_IDX/ \
  --SFT_load snap/ICR_Steam_TitleT_0_LCT_E40_gpt0.1_IDX/Epoch23_SFT
