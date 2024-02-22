import random
import sys
import time
import os
import numpy as np
import torch
import transformers
from RL.RL_trainer import RLHFTrainer
from SFT.SFT_trainer import SFTTrainer
from param import add_args, Config, add_args_RLHF, add_args_SFT, get_args

if __name__ == '__main__':
    args = get_args()
    kwargs = vars(args)
    args = Config(**kwargs)
    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    transformers.set_seed(args.seed)

    if args.train_stage == 'SFT' and args.share_chat_gpt_ratio > 0.:
        args.SFT_train_tasks = args.SFT_train_tasks + ',ShareChatGPT'
    if args.train_stage in ['RLHF', 'RLHF_merge']:
        if args.model_name is None:
            if args.lr > 0:
                args.model_name = f'RLHF_{args.backbone[5:]}Total_train_LM-{args.lm_head}_VM-{args.vague_mapping}_NR-20.1_SN-{args.sample_num}_Q-{args.quantization}_T{len(args.RLHF_train_tasks.split(","))}' \
                                  f'_FG-{args.fine_grain_reward}_LR-{args.lr}_LDO-{args.lora_dropout}_WD-{args.weight_decay}' \
                                  f'_KLC-{args.kl_coef}_EW-{args.entropy_weight}_RS-{args.reward_scale}_RW-{args.whiten_reward}' \
                                  f'_VFC-{args.vf_coef}_KLT-{args.policy_kl_threshold}_LRP-{args.lr_power}_GAMMA-{args.gamma}' \
                                  f'_GAS-{args.gradient_accumulation_steps}_LB-{args.learn_batch}_ND-{args.new_data}{"_AS" if args.add_seq else ""}_RA_{args.reward_alpha}' \
                                  f'_{args.model_name_suffix}'
            else:
                args.model_name = f'RLHF_{args.backbone[5:]}Total_init_LM-{args.lm_head}_VM-{args.vague_mapping}_NR-20.1_SN-{args.sample_num}_Q-{args.quantization}_T{len(args.RLHF_train_tasks.split(","))}'
        args.output = f'{args.output}{args.model_name}/'

    if args.log_to_file:
        log_file = open(args.output+f'{time.strftime("%Y-%m-%d %Hh_%Mm_%Ss", time.localtime())} {args.train_stage}.log', 'w')
        sys.stdout = log_file

    if args.train_stage in ['SFT', 'SFT_Test', 'SFT_Merge']:
        trainer = SFTTrainer(args)
    elif args.train_stage in ['RLHF', 'RLHF_Test', 'RLHF_Merge']:
        trainer = RLHFTrainer(args)
    else:
        raise NotImplementedError

    if args.train_stage == 'SFT':
        trainer.SFT_train()
    elif args.train_stage == 'SFT_Merge':
        trainer.SFT_adapter_merge()
    elif args.train_stage == 'RLHF':
        trainer.RLHF_train()
    elif args.train_stage == 'RLHF_Merge':
        trainer.RLHF_adapter_merge()
    else:
        raise NotImplementedError
