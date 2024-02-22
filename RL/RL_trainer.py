from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from collections import deque, namedtuple
from transformers import LogitsProcessorList, TopKLogitsWarper, \
    TemperatureLogitsWarper, StoppingCriteriaList, MaxLengthCriteria, MinLengthLogitsProcessor, \
    get_polynomial_decay_schedule_with_warmup
from accelerate import Accelerator
from RL.RL_dataset import *
from base.actor_critic import ActorCritic
from utils.metrics import Metrics
from base.trainer import Trainer
from utils.utils import *

# data
Memory = namedtuple('Memory', [
    'sequence',
    'action_mask',
    'old_action_value',
    'old_sequence_log_probs_shifted',
    'ref_sequence_log_probs_shifted',
    'whitened_advantages',
    'returns'
])


# rlhf trainer
class RLHFTrainer(Trainer):
    def __init__(self, args):
        super(RLHFTrainer, self).__init__(args)
        self.args = args
        self.accelerator = Accelerator()
        if self.accelerator.is_main_process:
            print(args)
        set_seed(self.args.seed)
        self.args.gpu = self.args.gpu or self.accelerator.device
        self.args.is_main_process = self.accelerator.is_main_process
        self.actor_critic = ActorCritic(args=self.args, device=self.args.gpu)
        self.actor_critic.print_trainable_parameters()
        self.data = {
            'metas': load_pickle(args.data_path + 'meta1.pickle'),
            'sequential': load_pickle(args.data_path + 'sequential.pickle'),
            # 'preference': load_pickle(args.data_path + 'preference.pickle'),
            # 'intention': load_pickle(args.data_path + 'intention.pickle'),
            'category': load_pickle(args.data_path + 'category1.pickle'),
            'ranking_candidate': load_pickle(args.data_path + 'ranking_candidate.pickle'),
        }
        if self.args.lower:
            for _ in self.data['metas']:
                self.data['metas'][_]['title'] = self.data['metas'][_]['title'].lower().strip()
                self.data['metas'][_][self.args.item_index] = self.data['metas'][_][
                    self.args.item_index].lower().strip()
        self.train_batch = 0
        self.backward_step = 0

    def compute_adv(self, old_action_values, rewards, action_mask):
        if self.args.whiten_reward:
            whitened_rewards = whiten(rewards, action_mask, shift_mean=False, dim=None)
        else:
            whitened_rewards = rewards
        last_gae_lam = 0
        advantages_reversed = []
        gen_length = torch.sum(action_mask, dim=1).max().item()
        for time_step in range(1, gen_length + 1):
            next_values = old_action_values[:, -time_step+1] if time_step > 1 else 0.0
            delta = whitened_rewards[:, -time_step] + self.args.gamma * next_values - old_action_values[:, -time_step]
            last_gae_lam = delta + self.args.gamma * self.args.lam * last_gae_lam
            advantages_reversed.append(last_gae_lam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        advantages = F.pad(advantages, (whitened_rewards.size(1) - gen_length, 0), value=0.0)
        returns = (advantages + old_action_values) * action_mask
        whitened_advantages = whiten(advantages, action_mask, dim=None).detach() * action_mask
        return whitened_advantages, returns

    def RLHF_train(self):
        self.actor_critic.load_parameters(self.args.RLHF_load)
        def learn_PPO():
            self.train()
            ratio_clip_range = 0.2
            value_clip_range = 0.2
            early_stop = False
            for epoch in range(self.args.epoch):
                # policy_kl_windows = [0.0] * self.args.gradient_accumulation_steps
                for idx, (
                    _sequence,
                    _action_mask,
                    _old_action_values,
                    _old_sequence_log_probs_shifted,
                    _ref_sequence_log_probs_shifted,
                    _whitened_advantages,
                    _returns
                ) in enumerate(memories):
                    self.backward_step += 1
                    sequence_logit = self.actor_critic.forward('actor', _sequence, attention_mask=_sequence != 0).logits
                    sequence_dists = torch.distributions.categorical.Categorical(logits=sequence_logit)
                    sequence_dists_shifted = shift(sequence_dists.probs, shift=1, dim=-2).contiguous()
                    sequence_log_probs_shifted = log_prob(sequence_dists_shifted, _sequence)
                    # entropy loss
                    entropy_losses = masked_mean(sequence_dists.entropy(), _action_mask)
                    # action loss
                    log_ratio = sequence_log_probs_shifted - _old_sequence_log_probs_shifted
                    ratio = torch.exp(log_ratio)
                    action_losses1 = -_whitened_advantages * ratio
                    action_losses2 = -_whitened_advantages * torch.clamp(ratio,
                                                                         min=1.0 - ratio_clip_range,
                                                                         max=1.0 + ratio_clip_range)
                    action_losses = masked_mean(torch.max(action_losses1, action_losses2), _action_mask)
                    # actor loss
                    actor_loss = action_losses - self.args.entropy_weight * entropy_losses
                    (actor_loss / self.args.gradient_accumulation_steps).backward()

                    # critic loss
                    action_values = self.actor_critic.forward('critic', _sequence, attention_mask=_sequence != 0)
                    new_values_clipped = torch.clamp(action_values,
                                                     min=_old_action_values - value_clip_range,
                                                     max=_old_action_values + value_clip_range)
                    critic_losses1 = torch.square(action_values - _returns)
                    critic_losses2 = torch.square(new_values_clipped - _returns)
                    critic_losses = 0.5 * masked_mean(torch.max(critic_losses1, critic_losses2), _action_mask)
                    (self.args.vf_coef * critic_losses / self.args.gradient_accumulation_steps).backward()

                    approx_kl = 0.5 * masked_mean(log_ratio**2, _action_mask)
                    policy_kl = masked_mean(-log_ratio, _action_mask)
                    critic_clip_frac = masked_mean(torch.gt(critic_losses2, critic_losses1).float(), _action_mask)
                    action_clip_frac = masked_mean(torch.gt(action_losses2, action_losses1).float(), _action_mask)
                    total_losses = action_losses + self.args.vf_coef * critic_losses - self.args.entropy_weight * entropy_losses
                    writer.add_scalar('training/actor_loss_mean', float(action_losses.mean()), self.backward_step)
                    writer.add_scalar('training/critic_loss_mean', float(critic_losses.mean()), self.backward_step)
                    writer.add_scalar('training/entropy_loss_mean', float(entropy_losses.mean()), self.backward_step)
                    writer.add_scalar('training/total_loss_mean', float(total_losses.mean()), self.backward_step)
                    writer.add_scalar('training/approx_kl', float(approx_kl), self.backward_step)
                    writer.add_scalar('training/policy_kl', float(policy_kl), self.backward_step)
                    writer.add_scalar('training/action_clip_frac', float(action_clip_frac), self.backward_step)
                    writer.add_scalar('training/critic_clip_frac', float(critic_clip_frac), self.backward_step)
                    # policy_kl_windows = policy_kl_windows[:-1] + [float(policy_kl)]
                    # policy_kl_mean = sum(policy_kl_windows)/self.args.gradient_accumulation_steps
                    if (idx + 1) % self.args.gradient_accumulation_steps == 0 or (idx + 1) == len(memories) or policy_kl > self.args.policy_kl_threshold:
                        if self.args.clip_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(RLHF_optim.param_groups[0]['params'], self.args.clip_grad_norm)
                        RLHF_optim.step()
                        RLHF_optim.zero_grad()
                        print(f"Batch: {self.train_batch} | Epoch: {epoch} | backward_step: {self.backward_step}"
                              f" | RLHF_lr: {RLHF_lr:.7f} | policy_kl: {policy_kl:.6f}")
                        if policy_kl > self.args.policy_kl_threshold:
                            early_stop = True
                            break
                if early_stop:
                    break
                np.random.shuffle(memories)

        TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.RLHF_train_tasks.split(',')+(['RLHFSeqRec'] if self.args.add_seq else [])}
        TaskNum = {_: 1 for _ in self.args.RLHF_train_tasks.split(',')}
        train_data = RLHFDataset(self.args, TaskTemplate, TaskNum, self.data, self.tokenizer, 'train')
        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)

        metrics_dict = Metrics(['RLHFTotal']+self.args.RLHF_train_tasks.split(',')+(['RLHFSeqRec'] if self.args.add_seq else []), self.args.topk, train_data.category2item, train_data.title2item)
        running = {'item': {_: RunningMoments() for _ in self.args.RLHF_train_tasks.split(',')+(['RLHFSeqRec'] if self.args.add_seq else [])}, 'list': RunningMoments()}

        RLHF_optim = self.get_optimizer_scheduler(self.actor_critic.actor_parameters+self.actor_critic.critic_parameters)
        RLHF_lr_scheduler = None
        if self.args.lr > 0:
            schedule_time = len(train_loader)*self.args.num_episodes//self.args.learn_batch
            RLHF_lr_scheduler = get_polynomial_decay_schedule_with_warmup(RLHF_optim, 50, schedule_time, power=self.args.lr_power, lr_end=1e-6)
        memories = deque([])

        writer = SummaryWriter(log_dir=f'logs/RLHF_train/{self.args.model_name}', flush_secs=30)

        logits_processor = LogitsProcessorList(
            [MinLengthLogitsProcessor(15, eos_token_id=self.actor_critic.tokenizer.eos_token_id)]
        )
        logits_warper = LogitsProcessorList(
            [TopKLogitsWarper(20), TemperatureLogitsWarper(0.7)]
        )
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]
        )

        sample_num = self.args.sample_num

        # if self.args.dry and self.args.lr > 0:
        #     self.RLHF_val(0, val_loader, writer)
        for eps in range(self.args.num_episodes):
            pbar = tqdm(total=len(train_loader), ncols=150)
            for step_i, batch in enumerate(train_loader):
                self.train_batch += 1
                if (step_i+1) % 100 == 0 and self.args.lr > 0:
                    self.actor_critic.save_parameters(f'{step_i+1}step')
                    # self.RLHF_val(step_i, val_loader, writer)
                self.eval()
                bs = len(batch['task'])
                input_data = batch['input_data']
                if step_i % 100 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                    metrics_dict.print()
                input_ids_length = input_data['input_ids'].shape[1]
                output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
                with torch.no_grad():
                    output_ids = self.actor_critic.actor_model.greedy_search(
                        **input_data,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                    )
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                    if sample_num > 1 and self.args.lr > 0:
                        sample_input_data = {
                            'input_ids': input_data['input_ids'].repeat((sample_num-1, 1)).contiguous(),
                            'attention_mask': input_data['attention_mask'].repeat((sample_num-1, 1)).contiguous()
                        }
                        output_ids = self.actor_critic.actor_model.sample(
                            **sample_input_data,
                            logits_processor=logits_processor,
                            logits_warper=logits_warper,
                            stopping_criteria=stopping_criteria,
                        )
                        output_title += self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                    output_title_list = [
                        [__.strip() for __ in _.strip().split('\n')]
                        for _ in output_title
                    ]
                    if self.args.idx:
                        output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                    if self.args.vague_mapping:
                        output_title_list = [train_data.vague_mapping(_) for _ in output_title_list]

                    reward_data = [
                        train_data.get_list_reward_hard_encode_NR_7(
                            batch['task'][i % bs],
                            batch['input_field_data'][i % bs],
                            output_title_list[i], new_data=self.args.new_data)
                        for i in range(bs * sample_num)
                    ]

                    for i in range(bs):
                        task = batch['task'][i]
                        metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i],
                                                output_labels[i], reward_data[i][0][2])
                        metrics_dict.add_sample('RLHFTotal', batch['input_field_data'][i], output_title_list[i],
                                                output_labels[i], reward_data[i][0][2])
                        writer.add_scalar(f'sampling/{task}_reward', metrics_dict[task]['RewardSum']/metrics_dict[task]['Count'], metrics_dict[task]['Count'])
                        writer.add_scalar(f'sampling/{task}_ndcg', metrics_dict[task][f'NDCG@{metrics_dict.topk}']/metrics_dict[task]['Count'], metrics_dict[task]['Count'])
                    total_count = metrics_dict['RLHFTotal']['Count']
                    total_reward_sum = metrics_dict['RLHFTotal']['RewardSum']
                    total_non_exist_rate = metrics_dict['RLHFTotal'][f'NonExistRate@{metrics_dict.topk}']
                    total_repeat_rate = metrics_dict['RLHFTotal'][f'RepeatRate@{metrics_dict.topk}']
                    total_correct_count = metrics_dict['RLHFTotal'][f'CorrectCount@{metrics_dict.topk}']
                    total_ndcg = metrics_dict['RLHFTotal'][f'NDCG@{metrics_dict.topk}']
                    writer.add_scalar('sampling/total_reward', total_reward_sum/total_count, total_count)
                    writer.add_scalar('sampling/total_non_exist_rate', total_non_exist_rate/total_count, total_count)
                    writer.add_scalar('sampling/total_repeat_rate', total_repeat_rate/total_count, total_count)
                    writer.add_scalar('sampling/total_correct_count', total_correct_count/total_count, total_count)
                    writer.add_scalar('sampling/total_ndcg', total_ndcg/total_count, total_count)
                    if self.args.lr > 0:
                        learn_data = {
                            'tasks': [_[0][0] for i, _ in enumerate(reward_data)],
                            'input_texts': [batch['input_text'][i % bs] for i, _ in enumerate(reward_data)],
                            'output_texts': [get_output_text(_[0][1], '\n'+self.tokenizer.eos_token, idx=self.args.idx) for i, _ in enumerate(reward_data)],
                            'complete_texts': [
                                batch['input_text'][i % bs] + ' ' + get_output_text(_[0][1], '\n' + self.tokenizer.eos_token, idx=self.args.idx)
                                for i, _ in enumerate(reward_data)
                            ],
                            'list_rewards': [_[0][2] for i, _ in enumerate(reward_data)],
                            'item_rewards': [_[0][3] for i, _ in enumerate(reward_data)],
                        }
                        if self.args.new_data:
                            learn_data['tasks'] += [_[1][0] for i, _ in enumerate(reward_data)]
                            learn_data['complete_texts'] += [
                                batch['input_text'][i % bs] + ' ' + get_output_text(_[1][1], '\n' + self.tokenizer.eos_token, idx=self.args.idx)
                                for i, _ in enumerate(reward_data)
                            ]
                            learn_data['list_rewards'] += [_[1][2] for i, _ in enumerate(reward_data)]
                            learn_data['item_rewards'] += [_[1][3] for i, _ in enumerate(reward_data)]
                        learn_sample_num = len(learn_data['complete_texts'])
                        for i in range(0, learn_sample_num, 16):
                            temp_learn_data = {
                                'tasks': learn_data['tasks'][i: i+16],
                                'input_texts': learn_data['input_texts'][i: i+16],
                                'output_texts': learn_data['output_texts'][i: i+16],
                                'complete_texts': learn_data['complete_texts'][i: i+16],
                                'list_rewards': learn_data['list_rewards'][i: i+16],
                                'item_rewards': learn_data['item_rewards'][i: i+16],
                            }
                            mini_sample_num = len(temp_learn_data['tasks'])
                            complete_data = side_tokenizer(temp_learn_data['complete_texts'],
                                                           'left', self.tokenizer,
                                                           padding=True, truncation=True,
                                                           max_length=self.args.max_token_length + self.args.gen_max_length,
                                                           return_tensors='pt').to(self.args.gpu).data
                            output_data = side_tokenizer(temp_learn_data['output_texts'], 'right', self.tokenizer)
                            old_action_values = self.actor_critic.forward(scope='critic', **complete_data)

                            old_sequence_logit = self.actor_critic.forward(scope='actor', **complete_data).logits
                            old_sequence_dists = torch.softmax(old_sequence_logit, dim=-1)
                            old_sequence_dists_shifted = shift(old_sequence_dists, shift=1, dim=-2).contiguous()
                            old_sequence_log_probs_shifted = log_prob(old_sequence_dists_shifted, complete_data['input_ids'])

                            ref_sequence_logit = self.actor_critic.forward(scope='base', **complete_data).logits
                            ref_sequence_dists = torch.softmax(ref_sequence_logit, dim=-1)
                            ref_sequence_dists_shifted = shift(ref_sequence_dists, shift=1, dim=-2).contiguous()
                            ref_sequence_log_probs_shifted = log_prob(ref_sequence_dists_shifted, complete_data['input_ids'])

                            # kl penalized, exploration
                            action_mask = torch.zeros_like(complete_data['input_ids'], dtype=torch.bool)
                            for idx in range(mini_sample_num):
                                output_length = len(output_data['input_ids'][idx])
                                action_mask[idx][-output_length+1:] = True

                            total_rewards = torch.zeros_like(complete_data['input_ids'], dtype=torch.float, device=self.args.gpu)
                            tasks = temp_learn_data['tasks']
                            list_rewards = torch.tensor(temp_learn_data['list_rewards'], device=self.args.gpu)
                            if self.args.reward_scale:
                                running['list'].update(list_rewards)
                                score_scaling_factor = running['list'].std + torch.finfo(list_rewards.dtype).eps
                                list_rewards /= score_scaling_factor
                            total_rewards[:, -1] += list_rewards

                            if self.args.fine_grain_reward:
                                item_rewards = [
                                    torch.concat([torch.tensor([0.0]*4, device=self.args.gpu), _])
                                    for _ in temp_learn_data['item_rewards']
                                ]
                                for idx in range(mini_sample_num):
                                    if self.args.reward_scale:
                                        running['item'][tasks[idx]].update(item_rewards[idx])
                                        score_scaling_factor = running['item'][tasks[idx]].std + torch.finfo(item_rewards[idx].dtype).eps
                                        item_rewards[idx] /= score_scaling_factor
                                    total_rewards[idx][complete_data['input_ids'][idx] == 13] += item_rewards[idx]

                            kl_penalty = (old_sequence_log_probs_shifted - ref_sequence_log_probs_shifted) * action_mask        # 其他kl
                            rewards_penalized = total_rewards - kl_penalty * self.args.kl_coef
                            whitened_advantages, returns = self.compute_adv(old_action_values, rewards_penalized, action_mask)

                            for idx in range(mini_sample_num):
                                ignore_index = torch.not_equal(complete_data['input_ids'][idx, ...], self.tokenizer.pad_token_id)
                                if ignore_index.sum() > 512:
                                    continue
                                (
                                    sequence_i,
                                    action_mask_i,
                                    old_action_values_i,
                                    old_sequence_log_probs_shifted_i,
                                    ref_sequence_log_probs_shifted_i,
                                    whitened_advantages_i,
                                    returns_i
                                ) = (
                                    rearrange(complete_data['input_ids'][idx, ...][ignore_index], '... -> 1 ...').contiguous(),
                                    rearrange(action_mask[idx, ...][ignore_index], '... -> 1 ...').contiguous(),
                                    rearrange(old_action_values[idx, ...][ignore_index], '... -> 1 ...').contiguous(),
                                    rearrange(old_sequence_log_probs_shifted[idx, ...][ignore_index], '... -> 1 ...').contiguous(),
                                    rearrange(ref_sequence_log_probs_shifted[idx, ...][ignore_index], '... -> 1 ...').contiguous(),
                                    rearrange(whitened_advantages[idx, ...][ignore_index], '... -> 1 ...').contiguous(),
                                    rearrange(returns[idx, ...][ignore_index], '... -> 1 ...').contiguous()
                                )

                                # append train sample
                                memories.append(
                                    Memory(
                                        sequence_i,
                                        action_mask_i,
                                        old_action_values_i,
                                        old_sequence_log_probs_shifted_i,
                                        ref_sequence_log_probs_shifted_i,
                                        whitened_advantages_i,
                                        returns_i
                                    )
                                )
                        pbar.set_description(f'rlhf learning in eps_{eps} | step_{step_i} | example: {len(memories)} | max_length: {complete_data["input_ids"].shape[1]}')
                pbar.update(1)
                if self.train_batch % self.args.learn_batch == 0 and self.args.lr > 0:
                    RLHF_lr_scheduler.step()
                    RLHF_lr = RLHF_lr_scheduler.get_lr()[0]
                    learn_PPO()
                    memories.clear()

            pbar.close()
            print('rlhf training complete')
        self.RLHF_val(writer)

    def RLHF_val(self, writer):
        val_steps = {int(_[:-13]): os.path.join(self.args.output, _[:-4]) for _ in os.listdir(self.args.output) if _.endswith('.pth')}
        if self.args.dry:
            val_steps[0] = None
        val_steps = {_: val_steps[_] for _ in sorted(val_steps, key=lambda k: k) if _ >= 0}
        ValTaskTemplate = {_: Val_task_group_mapping[_] for _ in self.args.RLHF_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.RLHF_val_tasks.split(',')}
        with self.accelerator.main_process_first():
            val_data = RLHFDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)
        if self.accelerator.is_main_process:
            print(val_steps.keys())
        val_loader = self.accelerator.prepare(val_loader)
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)]
        )
        self.eval()  # First batch users: ['A254N8K05ZWM4C', 'A2RM9L2MM9EQVE', 'A1D2LB4Z5RMO4M', 'A1D90MKRWB0ZKP']
        for train_step, model_file in val_steps.items():
            assert train_step == self.actor_critic.load_parameters(model_file)
            self.accelerator.wait_for_everyone()
            metrics_dict = Metrics(self.args.RLHF_val_tasks.split(','), self.args.topk, val_data.category2item, val_data.title2item)
            pbar = tqdm(total=len(val_loader), ncols=150, disable=not self.accelerator.is_main_process)
            for step_i, batch in enumerate(val_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                if self.accelerator.is_main_process and step_i % 1000 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                input_ids_length = input_data['input_ids'].shape[1]
                output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
                with torch.no_grad():
                    if train_step == 0:
                        output_ids = self.actor_critic.base_model.greedy_search(
                            **input_data,
                            stopping_criteria=stopping_criteria,
                        )
                    else:
                        output_ids = self.actor_critic.actor_model.greedy_search(
                            **input_data,
                            stopping_criteria=stopping_criteria,
                        )
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
                    output_title_list = [
                        [__.strip() for __ in _.strip().split('\n')]
                        for _ in output_title
                    ]
                    if self.args.idx:
                        output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
                    reward_data = [
                        val_data.get_list_reward_hard_encode_NR_7(
                            batch['task'][i],
                            batch['input_field_data'][i],
                            output_title_list[i])
                        for i in range(bs)
                    ]

                    list_rewards = [_[0][2] for _ in reward_data]
                    for i in range(bs):
                        task = batch['task'][i]
                        metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i],
                                                output_labels[i], list_rewards[i])
                pbar.update(1)

            metrics_dict.print()
            _reward_sum, _ndcg, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0, 0.0
            for task in metrics_dict.metrics_dict:
                task_count = metrics_dict[task]['Count']
                reward_sum = metrics_dict[task]['RewardSum']
                recall = metrics_dict[task][f'Recall@{metrics_dict.topk}']
                ndcg = metrics_dict[task][f'NDCG@{metrics_dict.topk}']
                non_exist_rate = metrics_dict[task][f'NonExistRate@{metrics_dict.topk}']
                repeat_rate = metrics_dict[task][f'RepeatRate@{metrics_dict.topk}']
                correct_count = metrics_dict[task][f'CorrectCount@{metrics_dict.topk}']
                if task == 'RLHFPersonalCategoryRate':
                    category_rate_correct = metrics_dict[task][f'CategoryRateCorrect@{metrics_dict.topk}']
                    log_d = torch.tensor(
                        [task_count, reward_sum, recall, ndcg, non_exist_rate, repeat_rate, correct_count, category_rate_correct],
                        device=self.accelerator.device)
                elif task == 'RLHFSeqRanking':
                    non_in_candidate_rate = metrics_dict[task][f'NotInCandidateRatio@{metrics_dict.topk}']
                    log_d = torch.tensor(
                        [task_count, reward_sum, recall, ndcg, non_exist_rate, repeat_rate, correct_count, non_in_candidate_rate],
                        device=self.accelerator.device)
                else:
                    log_d = torch.tensor(
                        [task_count, reward_sum, recall, ndcg, non_exist_rate, repeat_rate, correct_count],
                        device=self.accelerator.device)
                log_d = self.accelerator.reduce(log_d)
                with self.accelerator.main_process_first():
                    print(log_d)

                _reward_sum += log_d[1] / log_d[0]
                _ndcg += log_d[3] / log_d[0]
                _non_exist_rate += log_d[4] / log_d[0]
                _repeat_rate += log_d[5] / log_d[0]
                _correct_count += log_d[6] / log_d[0]

                if self.accelerator.is_main_process:
                    writer.add_scalar(f'valuating/{task}_Reward_mean', log_d[1] / log_d[0], train_step)
                    writer.add_scalar(f'valuating/{task}_Recall', log_d[2] / log_d[0], train_step)
                    writer.add_scalar(f'valuating/{task}_NDCG', log_d[3] / log_d[0], train_step)
                    writer.add_scalar(f'valuating/{task}_NonExist_rate', log_d[4] / log_d[0], train_step)
                    writer.add_scalar(f'valuating/{task}_Repeat_rate', log_d[5] / log_d[0], train_step)
                    writer.add_scalar(f'valuating/{task}_Correct_count', log_d[6] / log_d[0], train_step)
                    if task == 'RLHFPersonalCategoryRate':
                        writer.add_scalar(f'valuating/{task}_Category_rate_correct', log_d[7] / log_d[0], train_step)
                    if task == 'RLHFSeqRanking':
                        writer.add_scalar(f'valuating/{task}_NotInCandidate_rate', log_d[7] / log_d[0], train_step)
            if self.accelerator.is_main_process:
                writer.add_scalar(f'valuating/Total_Reward_mean', _reward_sum / len(ValTaskNum), train_step)
                writer.add_scalar(f'valuating/Total_NDCG', _ndcg / len(ValTaskNum), train_step)
                writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate / len(ValTaskNum), train_step)
                writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate / len(ValTaskNum), train_step)
                writer.add_scalar(f'valuating/Total_Correct_count', _correct_count / len(ValTaskNum), train_step)

        # "RLHFValSeqRec": {
        #     "NonExistRate@10": "0.4844",
        #     "RepeatRate@10": "0.4500",
        #     "CorrectCount@10": "1.0000",
        #     "Count": 640,
        #     "Recall@10": "0.1297",
        #     "MRR@10": "0.0934",
        #     "TargetCategoryRatio@10": "0.7766",
        #     "RewardSum": "0.3367"
        # }

    def RLHF_test(self):
        train_step = self.actor_critic.load_parameters(self.args.RLHF_load)
        torch.cuda.empty_cache()
        stopping_criteria = StoppingCriteriaList(
            [
                MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)
            ]
        )
        TestTaskTemplate = {self.args.RLHF_test_task: Test_task_group_mapping[self.args.RLHF_test_task]}
        TestTaskNum = {self.args.RLHF_test_task: 1}

        if self.args.RLHF_test_task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec', 'RLHFPersonalCategoryRate']:
            SeqRec_Result_file = f'{self.args.RLHF_load}_Result_RLHFSeqRec.pickle'
            self.data['RLHFSeqRec_Result'] = load_pickle(SeqRec_Result_file)
        test_data = RLHFDataset(self.args, TestTaskTemplate, TestTaskNum, self.data, self.tokenizer, 'test')
        test_loader = DataLoader(test_data, batch_size=self.args.test_batch_size, shuffle=False,
                                 collate_fn=test_data.collate_fn, drop_last=False)
        metrics_dict = Metrics([self.args.RLHF_test_task], self.args.topk, test_data.category2item, test_data.title2item)
        self.eval()

        result_file = f'{self.args.RLHF_load}_Result_{self.args.RLHF_test_task}_Top{self.args.topk}.pickle'
        result = load_pickle(result_file) or []
        result = result[:-(len(result) % self.args.batch_size) if (len(result) % self.args.batch_size) != 0 else None]
        pbar = tqdm(total=len(test_loader), ncols=150)

        with torch.no_grad():
            for step_i, batch in enumerate(test_loader):
                bs = len(batch['task'])
                input_data = batch['input_data']
                if step_i % 10000 == 0:
                    print(batch['input_text'][0])
                    print(input_data['input_ids'][0])
                output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
                input_ids_length = input_data['input_ids'].shape[1]
                if (step_i+1)*self.args.test_batch_size <= len(result):
                    output_title_list = [_[1] for _ in result[step_i*self.args.test_batch_size: (step_i+1)*self.args.test_batch_size]]
                else:
                    if train_step == 0:
                        output_ids = self.actor_critic.base_model.greedy_search(
                            **input_data,
                            stopping_criteria=stopping_criteria,
                        )
                    else:
                        output_ids = self.actor_critic.actor_model.greedy_search(
                            **input_data,
                            stopping_criteria=stopping_criteria,
                        )
                    output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:],
                                                               skip_special_tokens=True)
                    output_title_list = [[__.strip() for __ in _.strip().split('\n')] for _ in output_title]
                    if self.args.idx:
                        output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]

                    result += [[_, __] for _, __ in zip(output_labels, output_title_list)]
                    if step_i % 100 == 0 or (step_i + 1) == len(test_loader):
                        save_pickle(result, result_file)

                reward_data = [
                    test_data.get_list_reward_hard_encode_NR_7(
                        batch['task'][i],
                        batch['input_field_data'][i],
                        output_title_list[i])
                    for i in range(bs)
                ]

                list_rewards = torch.tensor([reward_data[i][0][2] for i in range(bs)], device=self.args.gpu)
                for i in range(bs):
                    task = batch['task'][i]
                    metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i], output_labels[i],
                                            float(list_rewards[i]))
                metrics_dict.print()
                pbar.update(1)
            pbar.close()

    def RLHF_adapter_merge(self):
        train_step = self.actor_critic.load_parameters(self.args.RLHF_load)
        self.actor_critic.lora_model.delete_adapter(self.actor_critic.critic_lora_scope)
        model = self.actor_critic.lora_model.merge_and_unload(progressbar=True)
        model.save_pretrained(f'{self.args.output}RLHF_Step{train_step}', safe_serialization=True)
        self.tokenizer.save_pretrained(f'{self.args.output}RLHF_Step{train_step}')


if __name__ == '__main__':
    pass

'''Best Parameters
--seed 0
--data_path data/dataset/sub_movie/
--output snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/
--backbone snap/ICR_SubMovie_Title64T_0_Q_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch38/
--item_index title64_t
--batch_size 8
--val_batch_size 12
--topk 10
--epoch 4
--gen_max_length 512
--train_stage RLHF
--RLHF_actor_lora_r 4
--RLHF_critic_lora_r 4
--RLHF_train_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate
--RLHF_val_tasks RLHFSeqRec,RLHFSeqRanking,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRate,RLHFItemCount
--backup_ip 0.0.0.0
--lr 0.000005               [√]
--lora_dropout 0.0          [√]
--weight_decay 0.0          [√]
--kl_coef 0.3               [√]
--entropy_weight 0.01       [√]
--vf_coef 0.1               [√]
--lm_head                   [√]
--policy_kl_threshold 0.05  [-]
--whiten_reward             [√]
--fine_grain_reward         [√]
--learn_batch 1             [√]
--sample_num 2              [√]
--clip_grad_norm 0.5                [√]
--gradient_accumulation_steps 4     [√]
--idx
--llama2_chat_template
--FA2
--gpu cuda:1
'''

'''
curl http://127.0.0.1:13580/generate \
-H "Content-Type: application/json" \
-d '{
"prompt": "[INST] <<SYS>>\nYou are an expert recommender engine. \n<</SYS>>\n\nYou need to tell me the category information of the given game.  [/INST] Ok, I will do it.  </s><s>[INST] What is the category of \"Gothic 1\".  [/INST]",
"max_tokens": 512,
"temperature": 0
}'

curl http://127.0.0.1:13579/generate \
-H "Content-Type: application/json" \
-d '{
"prompt": "[INST]  In the recommendation list, the ratio of \"Documentary\" item should be approximately 20%. Please generate a recommendation list with 20 different items.  [/INST]",
"max_tokens": 512,
"temperature": 0
}'

curl http://127.0.0.1:13579/generate \
-H "Content-Type: application/json" \
-d '{
"prompt": "[INST]  In the recommendation list, the proportion of \"documentary about music\" item should be approximately 40%. You need to generate a recommendation list with 5 different items.  [/INST]",
"max_tokens": 512,
"temperature": 0
}'

curl http://127.0.0.1:13579/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/",
"prompt": "[INST]  I prefer in 'RPG' items. Please generate a recommendation list with 10 different items.  [/INST]",
"max_tokens": 100
}'

curl http://127.0.0.1:13579/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/",
        "prompt": "San Francisco is a",
        "max_tokens": 1,
        "logprobs": 100
    }'
    
参考3篇论文：
ACL、EMNLP，微调LLM
RL的论文
论文框架合适的
'''

'''
CUDA_VISIBLE_DEVICES=3,4 nohup accelerate launch --num_processes 2 --gpu_ids all  train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/PALR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_IDX/ --backbone snap/Llama-2-7b-hf-chat/ --item_index title64_t --batch_size 2 --topk 10 --clip_grad_norm 1.0 --epoch 40 --gen_max_length 512 --lr 0.001 --gradient_accumulation_steps 12 --train_stage SFT --SFT_actor_lora_r 16 --SFT_actor_lora_a 8 --warmup_ratio 0.0125 --val_batch_size 16 --SFT_train_tasks SFTSeqRecPALR --SFT_val_tasks SFTTestSeqRec --backup_ip 0.0.0.0 --val_epoch 0 --share_chat_gpt_ratio 0.0 --FA2 --llama2_chat_template --idx > snap/PALR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_IDX/output.log &
CUDA_VISIBLE_DEVICES=8,9 nohup accelerate launch --num_processes 2 --gpu_ids all --main_process_port 29501  train.py --seed 0 --data_path data/dataset/steam/ --output snap/PALR_Steam_Title_0_Llama7bChat_LCT_E40_IDX/ --backbone snap/Llama-2-7b-hf-chat/ --item_index title --batch_size 2 --topk 10 --clip_grad_norm 1.0 --epoch 40 --gen_max_length 512 --lr 0.001 --gradient_accumulation_steps 12 --train_stage SFT --SFT_actor_lora_r 16 --SFT_actor_lora_a 8 --warmup_ratio 0.0125 --val_batch_size 16 --SFT_train_tasks SFTSeqRecPALR --SFT_val_tasks SFTTestSeqRec --backup_ip 0.0.0.0 --val_epoch 0 --share_chat_gpt_ratio 0.0 --FA2 --llama2_chat_template --idx > snap/PALR_Steam_Title_0_Llama7bChat_LCT_E40_IDX/output.log &
'''
# ToDo: python train.py --seed 0 --data_path data/dataset/sub_movie/ --output snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/ --backbone snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/ --item_index title64_t --batch_size 8 --val_batch_size 12 --gradient_accumulation_steps 4 --topk 10 --clip_grad_norm 0.5 --epoch 4 --gen_max_length 512 --candidate_num 15 --train_stage RLHF --RLHF_actor_lora_r 4 --RLHF_critic_lora_r 4 --RLHF_train_tasks RLHFSeqRec,RLHF+PersonalControlRec,RLHF-PersonalControlRec,RLHFPersonalCategoryRateLP,RLHFPersonalCategoryRateMP,RLHFPersonalCategoryRateEP --backup_ip 0.0.0.0 --lr 0.000005 --lora_drop 0.0 --weight_decay 0.0 --kl_coef 0.3 --entropy_weight 0.01 --vf_coef 0.1 --lm_head --policy_kl_threshold 0.05 --idx --llama2_chat_template --FA2 --gpu cuda:8 --lr_power 2.0 --learn_batch 1 --sample_num 2 --whiten_reward --num_episodes 2 --vague_mapping