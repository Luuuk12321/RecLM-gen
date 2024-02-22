import os

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from torch import nn
from torch.optim import AdamW, Adam
from transformers import get_polynomial_decay_schedule_with_warmup

from actor_critic import ActorCritic
# from RLHF.reward_model import RewardModel


# trainer
class Trainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps)
        set_seed(self.args.seed)
        self.args.gpu = self.args.gpu or self.accelerator.device
        self.actor_critic = ActorCritic(args=self.args, device=self.args.gpu)
        if self.accelerator.is_main_process:
            print(args)
            self.actor_critic.print_trainable_parameters()

    def get_optimizer_scheduler(self, named_params, batch_per_epoch=None, group_wd_params=True):
        params = [p for n, p in named_params.items() if p.requires_grad]
        assert self.args.weight_decay >= 0
        if group_wd_params and self.args.weight_decay > 0:
            params = [
                {'params': [p for p in params if p.ndim >= 2]},
                {'params': [p for p in params if p.ndim < 2], 'weight_decay': 0},
            ]

        optim = AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay,
                      betas=(self.args.adam_beta1, self.args.adam_beta2), eps=self.args.adam_eps)
        if batch_per_epoch > 0:
            step_total = batch_per_epoch * (self.args.epoch - self.start_epoch) // self.args.gradient_accumulation_steps
            warmup_iters = int(step_total * self.args.warmup_ratio)
            # scheduler = get_linear_schedule_with_warmup(SFT_optim, warmup_iters, step_total)
            scheduler = get_polynomial_decay_schedule_with_warmup(optim, warmup_iters, step_total, power=2.0)
            return optim, scheduler
        return optim

    @property
    def device(self):
        return self.args.gpu

    @property
    def tokenizer(self):
        return self.actor_critic.tokenizer


if __name__ == '__main__':
    pass
