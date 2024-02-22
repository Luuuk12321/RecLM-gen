from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from SFT.SFT_dataset import SFTDataset, Train_task_group_mapping, Val_task_group_mapping, Test_task_group_mapping
from torch.utils.tensorboard import SummaryWriter
from base.actor_critic import ActorCritic
from data.dataset import BaseDataset
from utils.metrics import Metrics
from base.trainer import Trainer
from utils.utils import *
from accelerate import Accelerator
from accelerate.utils import set_seed


class SFTTrainer(Trainer):
    def __init__(self, args):
        super(SFTTrainer, self).__init__(args)
        self.sft_loss_fct = CrossEntropyLoss(reduction='none')
        self.start_epoch = self.actor_critic.load_parameters(self.args.SFT_load)

        self.writer = None
        if self.accelerator.is_main_process:
            name = self.args.output.split('snap/')[-1]
            self.writer = SummaryWriter(log_dir=f'logs/SFT_train/{self.args.SFT_train_tasks}/{name}', flush_secs=30)

    def SFT_Loss(self, logit, label):
        shift_logit = logit[..., :-1, :].contiguous()
        shift_label = label[..., 1:].contiguous()
        loss = self.sft_loss_fct(shift_logit.view(-1, self.actor_critic.model_config.vocab_size), shift_label.view(-1))
        loss = loss.view(label.shape[0], -1)
        loss = loss.sum(dim=1) / (shift_label != -100).sum(dim=1)  # [bs]
        loss = masked_mean(loss, shift_label != -100, dim=1)  # [bs]
        return loss

    def SFT_train(self):
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                print(f'computing train and val datum info')
            if self.args.data_path:
                TaskTemplate = {_: Train_task_group_mapping[_] for _ in self.args.SFT_train_tasks.split(',')}
                TaskNum = {_: 1 for _ in self.args.SFT_train_tasks.split(',')}
                ValTaskTemplate = {_: Val_task_group_mapping[_.split('_')[0]] for _ in self.args.SFT_val_tasks.split(',')}
                ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}
                data = {
                    'category': load_pickle(self.args.data_path + 'category1.pickle'),
                    'metas': load_pickle(self.args.data_path + 'meta1.pickle'),
                    'sequential': load_pickle(self.args.data_path + 'sequential.pickle'),
                    'share_chat_gpt': load_pickle('data/dataset/share_chat_gpt2.pickle'),
                    'ranking_candidate': load_pickle(self.args.data_path + 'ranking_candidate.pickle'),
                }
                train_data = SFTDataset(self.args, TaskTemplate, TaskNum, data, self.tokenizer, 'train')
                val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, data, self.tokenizer, 'val')
            elif self.args.data_file:
                train_data = BaseDataset(self.args, self.tokenizer, 'train')
                val_data = BaseDataset(self.args, self.tokenizer, 'val')

        train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=train_data.collate_fn)
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)

        SFT_optim, SFT_lr_scheduler = self.get_optimizer_scheduler(self.actor_critic.actor_named_parameters, len(train_loader))

        warped_actor_critic, SFT_optim, train_loader, val_loader, SFT_lr_scheduler = self.accelerator.prepare(
            self.actor_critic, SFT_optim, train_loader, val_loader, SFT_lr_scheduler
        )

        if self.args.dry:
            self.SFT_evl_inference(self.start_epoch, val_loader)
        best_val_loss = float('inf')
        for epoch in range(self.start_epoch+1, self.args.epoch+1):
            task_loss = {_: 0.0 for _ in train_data.task_num}
            task_count = {_: 1e-10 for _ in train_data.task_num}
            pbar = tqdm(total=len(train_loader), ncols=210, disable=not self.accelerator.is_local_main_process)
            self.train()
            for step_i, batch in enumerate(train_loader):
                with self.accelerator.accumulate(warped_actor_critic):
                    # print(f'parameter {step_i}: ', self.actor_critic.actor_parameters[0].data.abs().max())
                    # self.accelerator.wait_for_everyone()
                    input_data, labels = batch['complete_text_data'], batch['complete_label_ids']
                    if self.accelerator.is_main_process and step_i % 10000 == 0:
                        print(batch['complete_text'][0])
                        print(input_data['input_ids'][0])
                    results = warped_actor_critic.forward(scope='actor', **input_data)
                    loss = self.SFT_Loss(results.logits, labels)

                    self.accelerator.backward(loss.mean())  # auto divide accumulate step, sync grad if arrive accumulate step
                    # print(f'grad {step_i}: ', self.actor_critic.actor_parameters[0].grad.abs().max())
                    # self.accelerator.wait_for_everyone()

                    if self.accelerator.sync_gradients:
                        # print(f'sync grad {step_i}: ', self.actor_critic.actor_parameters[0].grad.abs().max())
                        # self.accelerator.wait_for_everyone()
                        if self.args.clip_grad_norm > 0:
                            total_norm = self.accelerator.clip_grad_norm_(SFT_optim.param_groups[0]['params'], self.args.clip_grad_norm)
                            # writer.add_scalars('training/total_norm', {f'epoch{epoch}': float(total_norm)}, step_i)

                    SFT_optim.step()
                    SFT_lr_scheduler.step()
                    SFT_optim.zero_grad()

                for idx, task in enumerate(batch['task']):
                    task_loss[task] += (float(loss[idx]))
                    task_count[task] += 1

                if self.accelerator.sync_gradients:
                    losses = torch.tensor([_ for _ in task_loss.values()], device=self.accelerator.device)
                    counts = torch.tensor([_ for _ in task_count.values()], device=self.accelerator.device)
                    losses = self.accelerator.reduce(losses)  # [task_num]
                    counts = self.accelerator.reduce(counts)  # [task_num]
                    if self.accelerator.is_main_process:
                        for idx, task in enumerate(list(task_loss.keys())):
                            self.writer.add_scalars(f'training/{task}_Loss', {f'epoch{epoch}': losses[idx] / counts[idx]}, counts[idx])
                        ShareChatGPT_mask = torch.tensor(
                            [1.0 if _ != 'ShareChatGPT' else 0.0 for _ in task_loss.keys()],
                            device=self.accelerator.device
                        )
                        self.writer.add_scalars('training/All_Loss', {f'epoch{epoch}': float(masked_mean(losses/counts, ShareChatGPT_mask))}, step_i)
                        desc_str = f'E{epoch} | LR {SFT_lr_scheduler.get_lr()[0]:.4f}' \
                                   f' | {" | ".join([f"{task}: {losses[idx] / counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}'
                        pbar.set_description(desc_str, refresh=False)
                        pbar.update(self.args.gradient_accumulation_steps)

            pbar.close()
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self.actor_critic.save_parameters(f"Epoch{epoch:02d}")
            if epoch < self.args.val_epoch:
                continue
            val_loss = self.SFT_evl_inference(epoch, val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.actor_critic.save_parameters("BEST_EVAL_LOSS")

    def SFT_evl_loss(self, epoch):
        torch.cuda.empty_cache()
        self.eval()
        ValTaskTemplate = {_: Val_task_group_mapping[_] for _ in self.args.SFT_val_tasks.split(',')}
        ValTaskNum = {_: 1 for _ in self.args.SFT_val_tasks.split(',')}
        with self.accelerator.main_process_first():
            val_data = SFTDataset(self.args, ValTaskTemplate, ValTaskNum, self.data, self.tokenizer, 'val')
        val_loader = DataLoader(val_data, batch_size=self.args.val_batch_size, shuffle=False, collate_fn=val_data.collate_fn, drop_last=False)

        task_loss = {_: 0.0 for _ in val_data.task_num}
        task_count = {_: 1e-10 for _ in val_data.task_num}
        with torch.no_grad():
            for step_i, batch in tqdm(enumerate(val_loader), ncols=200, disable=not self.accelerator.is_local_main_process):
                input_data = batch['complete_text_data']
                if self.accelerator.is_main_process and step_i % 10000 == 0:
                    print(batch['complete_text'][0])
                    print(input_data['input_ids'][0])
                labels = batch['complete_label_ids']
                results = self.actor_critic.forward(scpoe='actor', **input_data)
                loss = self.SFT_Loss(results.logits, labels).detach()
                for idx, task in enumerate(batch['task']):
                    task_loss[task] += (float(loss[idx]))
                    task_count[task] += 1

        losses = torch.tensor([_ for _ in task_loss.values()], device=self.accelerator.device)
        counts = torch.tensor([_ for _ in task_count.values()], device=self.accelerator.device)
        losses = self.accelerator.reduce(losses)  # [task_num]
        counts = self.accelerator.reduce(counts)  # [task_num]
        val_loss = float((losses/counts).mean())
        if self.accelerator.is_main_process:
            print(f'Epoch {epoch} | {" | ".join([f"Val_{task}_Loss: {losses[idx]/counts[idx]:.4f}" for idx, task in enumerate(list(task_loss.keys()))])}')
            print(f'Epoch {epoch} | SFT_Val_Loss: {val_loss:.4f}\n')
            self.writer.add_scalars(f'valuating', {f'{task}_Loss': losses[idx]/counts[idx] for idx, task in enumerate(list(task_loss.keys()))}, epoch)
            self.writer.add_scalars(f'valuating', {'total_Loss': val_loss}, epoch)
        self.train()
        return val_loss

    def SFT_evl_inference(self, epoch, val_loader):
        torch.cuda.empty_cache()
        self.eval()
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=self.args.max_token_length + self.args.gen_max_length)])
        metrics_dict = Metrics(self.args.SFT_val_tasks.split(','), self.args.topk, val_loader.dataset.category2item, val_loader.dataset.title2item)
        for step_i, batch in tqdm(enumerate(val_loader), ncols=200, disable=not self.accelerator.is_local_main_process):
            bs = len(batch['task'])
            input_data = batch['input_data']
            if self.accelerator.is_main_process and step_i % 1000 == 0:
                print(batch['input_text'][0])
                print(input_data['input_ids'][0])
            input_ids_length = input_data['input_ids'].shape[1]

            output_labels = [[__ for __ in _.strip().split('\n')] for _ in batch['output_text']]
            with torch.no_grad():
                evl_model = self.actor_critic.base_model if epoch == 0 else self.actor_critic.actor_model
                output_ids = evl_model.greedy_search(**input_data, stopping_criteria=stopping_criteria)

            # process output text
            output_title = self.tokenizer.batch_decode(output_ids[:, input_ids_length:], skip_special_tokens=True)
            output_title_list = [
                [__.strip() for __ in _.strip().split('\n')] for _ in output_title
            ]
            if self.args.idx:
                output_labels = [[rm_idx(__) for __ in _] for _ in output_labels]
                output_title_list = [[rm_idx(__) for __ in _] for _ in output_title_list]
            for i in range(bs):
                task = batch['task'][i]
                metrics_dict.add_sample(task, batch['input_field_data'][i], output_title_list[i], output_labels[i])

        _ndcg, _non_exist_rate, _repeat_rate, _correct_count = 0.0, 0.0, 0.0, 0.0
        for task in metrics_dict:
            task_count = metrics_dict[task]['Count']
            recall = metrics_dict[task][f'Recall@{metrics_dict.topk}']
            ndcg = metrics_dict[task][f'NDCG@{metrics_dict.topk}']
            non_exist_rate = metrics_dict[task][f'NonExistRate@{metrics_dict.topk}']
            repeat_rate = metrics_dict[task][f'RepeatRate@{metrics_dict.topk}']
            correct_count = metrics_dict[task][f'CorrectCount@{metrics_dict.topk}']

            if task == 'SFTTestPersonalCategoryRate':
                category_rate_correct = metrics_dict[task][f'CategoryRateCorrect@{metrics_dict.topk}']
                log_d = torch.tensor(
                    [task_count, recall, ndcg, non_exist_rate, repeat_rate, correct_count, category_rate_correct],
                    device=self.accelerator.device)
            else:
                log_d = torch.tensor(
                    [task_count, recall, ndcg, non_exist_rate, repeat_rate, correct_count],
                    device=self.accelerator.device)
            log_d = self.accelerator.reduce(log_d)
            with self.accelerator.main_process_first():
                print(log_d)

            _ndcg += log_d[2] / log_d[0]
            _non_exist_rate += log_d[3] / log_d[0]
            _repeat_rate += log_d[4] / log_d[0]
            _correct_count += log_d[5] / log_d[0]

            if self.accelerator.is_main_process:
                self.writer.add_scalar(f'valuating/{task}_Recall', log_d[1] / log_d[0], epoch)
                self.writer.add_scalar(f'valuating/{task}_NDCG', log_d[2] / log_d[0], epoch)
                self.writer.add_scalar(f'valuating/{task}_NonExist_rate', log_d[3] / log_d[0], epoch)
                self.writer.add_scalar(f'valuating/{task}_Repeat_rate', log_d[4] / log_d[0], epoch)
                self.writer.add_scalar(f'valuating/{task}_Correct_count', log_d[5] / log_d[0], epoch)
                if task == 'RLHFPersonalCategoryRate':
                    self.writer.add_scalar(f'valuating/{task}_Category_rate_correct', log_d[6] / log_d[0], epoch)
        if self.accelerator.is_main_process:
            val_task_num = len(val_loader.dataset.task_num)
            self.writer.add_scalar(f'valuating/Total_NDCG', _ndcg / val_task_num, epoch)
            self.writer.add_scalar(f'valuating/Total_NonExist_rate', _non_exist_rate / val_task_num, epoch)
            self.writer.add_scalar(f'valuating/Total_Repeat_rate', _repeat_rate / val_task_num, epoch)
            self.writer.add_scalar(f'valuating/Total_Correct_count', _correct_count / val_task_num, epoch)
            print(f'Epoch {epoch} | SFT_Val_NDCG: {_ndcg:.4f}\n')
        self.train()
        return 0.0 - _ndcg

    def SFT_adapter_merge(self):
        model = self.actor_critic.lora_model.merge_and_unload(progressbar=True)
        model.save_pretrained(f'{self.args.output}SFT_Epoch{self.start_epoch:02d}', safe_serialization=True)
        self.tokenizer.save_pretrained(f'{self.args.output}SFT_Epoch{self.start_epoch:02d}')


if __name__ == "__main__":
    pass
