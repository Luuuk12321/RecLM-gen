#!/bin/bash

DATASET='toys'
OUTPUT_PATH='snap/CtrlRec_'$DATASET'_TitleT_0_LCT_E40_gpt0.1_IDX/'
mkdir -p "$OUTPUT_PATH"

CUDA_VISIBLE_DEVICES=0,1 nohup accelerate launch --num_processes 2 --gpu_ids all --main_process_port 13328 main.py \
  --seed 0 \
  --data_path data/dataset/$DATASET/ \
  --output $OUTPUT_PATH \
  --backbone /home/lws/models/Llama-3-8B-instruct/ \
  --item_index title_t \
  --batch_size 1 \
  --topk 10 \
  --clip_grad_norm 1.0 \
  --epoch 40 \
  --gen_max_length 512 \
  --lr 0.0001 \
  --gradient_accumulation_steps 32 \
  --train_stage SFT \
  --SFT_actor_lora_r 16 \
  --SFT_actor_lora_a 8 \
  --warmup_ratio 0.0125 \
  --val_batch_size 16 \
  --SFT_train_tasks SFTSeqRec,SFTControlRec_re,ShareChatGPT \
  --SFT_val_tasks SFTTestSeqRec \
  --backup_ip 0.0.0.0 \
  --val_epoch 0 \
  --share_chat_gpt_ratio 0.1 \
  --FA2 \
  --llama2_chat_template \
  --idx \
  --teacher_port 2067 \
  --distributed > "$OUTPUT_PATH"output.log 2>&1 &


#        if 'steam' in args.data_path:
#            self.teacher_port = 2066
#        elif 'toys' in args.data_path:
#            self.teacher_port = 2067
#        else:  # movies
#            self.teacher_port = 2068