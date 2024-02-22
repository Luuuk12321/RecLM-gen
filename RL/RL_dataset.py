import copy

import math
import random
from typing import List

from tqdm import tqdm
import numpy as np
from RL.RL_template import *
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import get_item_list, load_pickle, side_tokenizer, get_item_ranking, get_history_text, save_pickle, \
    get_output_text
import Levenshtein


class ExperienceDataset(Dataset):
    def __init__(self, data: List[torch.Tensor], device=None):
        super().__init__()
        self.data = data
        self.device = device

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, ind):
        return tuple(map(lambda t: t[ind].to(self.device), self.data))


def create_dataloader(data, batch_size, shuffle=True, device=None, **kwargs):
    ds = ExperienceDataset(data, device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)


class RLHFDataset(Dataset):
    def __init__(self, args, task_template, task_num, data, tokenizer, mode='train'):
        self.args = args
        self.task_template = task_template
        self.task_num = task_num
        self.mode = mode
        self.tokenizer = tokenizer
        self.teacher_port = self.args.teacher_port

        self.metas = data['metas']
        self.sequential = data['sequential']
        # self.preference = data['preference']
        # self.intention = data['intention']
        self.category2item = data['category']
        self.ranking_candidate = data['ranking_candidate']
        if 'RLHFSeqRec_Result' in data:
            self.RLHFSeqRec_Result = {u: data['RLHFSeqRec_Result'][idx][1] for idx, u in enumerate(self.sequential)}

        self.item2category = {}
        for c in self.category2item:
            for i in self.category2item[c]:
                if self.item2category.get(i) is None:
                    self.item2category[i] = []
                self.item2category[i].append(c)
        self.title2item = {}
        for _ in self.metas:
            if self.title2item.get(self.metas[_][self.args.item_index]) is None:
                self.title2item[self.metas[_][self.args.item_index]] = []
            self.title2item[self.metas[_][self.args.item_index]].append(_)

        # self.title = [__[self.args.item_index] for _, __ in self.metas.items()]
        # self.decoder_start_token_id = self.tokenizer.bos_token_id
        # self.item_prefix_tree = self.create_item_prefix_tree()

        print('compute_datum_info')
        self.temp_candidate_items = None
        self.datum_info = []
        self.complete_datum_info_path = None
        if self.mode == 'val':
            self.complete_datum_info_path = args.data_path + f'RLHF_datum_info_{self.mode}_{self.args.RLHF_val_tasks}_Top{self.args.topk}.pickle'
        # elif self.mode == 'test':
        #     self.complete_datum_info_path = args.data_path + f'RLHF_datum_info_{self.mode}_{self.args.RLHF_test_task}_Top{self.args.topk}.pickle'
        elif self.mode == 'train':
            self.complete_datum_info_path = args.data_path + f'RLHF_datum_info_{self.mode}_{self.args.RLHF_train_tasks}{"_AS" if self.args.add_seq else ""}.pickle'
        self.complete_datum_info = load_pickle(self.complete_datum_info_path) or []
        self.compute_datum_info()
        self.best_ranking_score = [0]
        for i in range(0, self.args.topk*2):
            self.best_ranking_score.append(self.best_ranking_score[i]+self.ranking_score_func(i)/math.log2(i+2))

    def find_maximum_category(self, item_list, target_item, max_count=99999):
        category_count = {c: 0 for c in self.category2item if target_item not in self.category2item[c]}
        for o_i in item_list:
            for c in category_count:
                if o_i in self.category2item[c]:
                    category_count[c] += 1
        max_count = min(max_count, max(list(category_count.values())))
        category = [c for c in list(category_count.keys()) if category_count[c] >= max_count]
        return category

    def vague_selecting(self, title_list, candidates):
        res = copy.deepcopy(title_list)
        for idx, _ in enumerate(res):
            if _ not in candidates or _ in res[:idx]:
                closest_distance = float('inf')
                closest_match = None
                for __ in list(set(candidates)-set(res[:idx])):
                    distance = Levenshtein.distance(_, __)
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_match = __
                res[idx] = closest_match
        return res

    def vague_mapping(self, title_list):
        res = copy.deepcopy(title_list)
        for idx, _ in enumerate(res):
            if _ in self.title2item:
                continue
            for __ in list(set(self.title2item)-set(res[:idx])):
                if Levenshtein.distance(_, __) < 3:
                    res[idx] = __
                    break
        return res

    # def create_item_prefix_tree(self):
    #     title_index = [self.get_item_index(self.metas[_]['asin']) for _ in self.metas]
    #     item_list = [self.metas[_]['asin'] for _ in self.metas]
    #     title_ids = self.tokenizer.batch_encode_plus(title_index,
    #                                                  padding=True, truncation=True,
    #                                                  max_length=self.args.gen_max_length,
    #                                                  return_tensors='pt').data['input_ids']
    #     if torch.all(torch.eq(title_ids[:, 0], self.decoder_start_token_id)):
    #         title_ids = title_ids[:, 1:]
    #     item_prefix_tree = {str(self.decoder_start_token_id): []}
    #     for i, ids in enumerate(title_ids):
    #         temp = str(self.decoder_start_token_id)
    #         for token in ids:
    #             _next = int(token)
    #             if token == self.tokenizer.pad_token_id or token == self.tokenizer.eos_token:
    #                 break
    #             if item_prefix_tree.get(temp) is None:
    #                 item_prefix_tree[temp] = []
    #             if _next not in item_prefix_tree[temp]:
    #                 item_prefix_tree[temp].append(_next)
    #             temp = temp + ' ' + str(_next)
    #         if item_prefix_tree.get(temp) is None:
    #             item_prefix_tree[temp] = []
    #         item_prefix_tree[temp].append(item_list[i])
    #     return item_prefix_tree

    def compute_datum_info(self):
        val_num = 320
        for task, num in self.task_num.items():
            if task == "RLHFSeqRec":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*0:val_num*1]]
            elif task == "RLHF+PersonalControlRec":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*1:val_num*2]]
            elif task == "RLHF-PersonalControlRec":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*2:val_num*3]]
            elif task == 'RLHFPersonalCategoryRate':
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*3:val_num*4]]
            elif task.startswith('RLHFPersonalCategoryRate'):
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*3:val_num*4]]
            elif task == "RLHFSeqRanking":
                for _ in range(num):
                    if self.mode in ['train', 'test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*4:val_num*5]]
            elif task == "RLHFItemCount":
                for _ in range(num):
                    if self.mode in ['test']:
                        self.datum_info += [[task, u] for u in self.sequential]
                    elif self.mode == 'val':
                        self.datum_info += [[task, u] for u in list(self.sequential.keys())[val_num*5:val_num*6]]
            else:
                raise NotImplementedError
        if len(self.complete_datum_info) != len(self.datum_info):
            self.complete_datum_info = []
            for idx in tqdm(range(len(self.complete_datum_info), len(self.datum_info)), desc=f'computing {self.mode} datum info'):
                self.complete_datum_info.append(self.getitem(idx))
            if self.mode in ['val', 'test']:
                input_ids = self.tokenizer.batch_encode_plus([_['input_text'] for _ in self.complete_datum_info],
                                                             truncation=True, max_length=2)['input_ids']
                token_length = [len(_) for _ in input_ids]
                datum_info_index = np.argsort(token_length).tolist()
                self.complete_datum_info = [self.complete_datum_info[idx] for idx in datum_info_index]
            save_pickle(self.complete_datum_info, self.complete_datum_info_path)
        if self.mode == 'train':
            self.shuffle()

    def shuffle(self):
        random.shuffle(self.complete_datum_info)

    def __len__(self):
        return len(self.complete_datum_info)

    def get_item_index(self, item):
        if self.args.item_index == 'title':
            return self.metas[item]['title']
        elif self.args.item_index == 'title32':
            return self.metas[item]['title32']
        elif self.args.item_index == 'title64':
            return self.metas[item]['title64']
        elif self.args.item_index == 'item':
            return item
        else:
            return self.metas[item][self.args.item_index]

    def get_sub_sequential(self, user):
        if self.mode == 'train':
            sequential = self.sequential[user][:-2]
            target_item_index = random.choice(range(1, len(sequential)))
            min_start_item_index = max(0, target_item_index-self.args.max_item_length)
            start_item_index = random.choice(range(min_start_item_index, target_item_index))
            sub_sequential = sequential[start_item_index: target_item_index]
            target_item = sequential[target_item_index]
        elif self.mode == 'val':
            sub_sequential = self.sequential[user][-self.args.max_item_length-2:-2]
            target_item = self.sequential[user][-2]
        elif self.mode == 'test':
            sub_sequential = self.sequential[user][-self.args.max_item_length-1:-1]
            target_item = self.sequential[user][-1]
        else:
            raise NotImplementedError
        return sub_sequential, target_item

    def getitem(self, idx):
        task = self.datum_info[idx][0]
        template_id = random.choice(list(self.task_template[task].keys()))
        template_selected = self.task_template[task][template_id]
        item_count = random.choice(range(self.args.topk//2, self.args.topk))+1 if self.mode == 'train' else self.args.topk
        input_field_data, output_field_data = {'item_count': item_count}, {}
        if task in ["RLHFSeqRec", 'RLHF+PersonalControlRec', 'RLHF-PersonalControlRec', 'RLHFSeqRanking',
                    'RLHFItemCount'] or task.startswith('RLHFPersonalCategoryRate'):
            user = self.datum_info[idx][1]
            sub_sequential, target_item = self.get_sub_sequential(user)
            input_field_data.update({
                'history': get_history_text([f"'{self.get_item_index(_)}'" for _ in sub_sequential]),
                'user': user,
                'sub_sequential': sub_sequential,
                'target_item': target_item,
                'target_category': self.item2category[target_item][-1]
            })
            output_field_data.update({
                'item_list': get_output_text([self.get_item_index(target_item)], '', idx=False)
            })

            if task in ["RLHF+PersonalControlRec"]:
                if self.mode in ['train', 'val']:
                    # item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port)['inference'][0]
                    if self.mode == 'train':
                        target_category = random.choice(self.item2category[target_item])
                    else:
                        target_category = self.item2category[target_item][-1]
                    input_field_data.update({
                        'target_category': target_category,
                        # 'SeqRec_Result': [self.get_item_index(_) for _ in item_list]
                    })
                elif self.mode in ['test']:
                    target_category = self.item2category[target_item][-1]
                    input_field_data.update({
                        'target_category': target_category,
                        'SeqRec_Result': self.RLHFSeqRec_Result[user]
                    })
                intention_template_key = random.choice(list(Intention_plus_group.keys()))
                intention = Intention_plus_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                })
            elif task in ["RLHF-PersonalControlRec"]:
                if self.mode in ['train', 'val']:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port)['inference'][0]
                    if self.mode == 'train':
                        target_category = random.choice(self.find_maximum_category(item_list, target_item, max_count=2))
                    else:
                        target_category = self.find_maximum_category(item_list, target_item)[-1]
                    input_field_data.update({
                        'target_category': target_category,
                        'SeqRec_Result': [self.get_item_index(_) for _ in item_list]
                    })
                else:
                    item_list = [self.title2item[_][0] for _ in self.RLHFSeqRec_Result[user] if self.title2item.get(_)]
                    target_category = self.find_maximum_category(item_list, target_item)[-1]
                    input_field_data.update({
                        'target_category': target_category,
                        'SeqRec_Result': self.RLHFSeqRec_Result[user]
                    })
                intention_template_key = random.choice(list(Intention_minus_group.keys()))
                intention = Intention_minus_group[intention_template_key].get_input_text(input_field_data)
                input_field_data.update({
                    'synthetic_intention': intention,
                })
            elif task.startswith("RLHFPersonalCategoryRate"):
                if self.mode in ['train', 'val']:
                    item_list = get_item_list(self.args.backup_ip, [user], [sub_sequential], item_count, port=self.teacher_port)['inference'][0]
                    if self.mode == 'train':
                        if 'MP' in task or 'EP' in task:
                            target_category = random.choice(self.item2category[target_item])
                        elif 'LP' in task:
                            target_category = random.choice(self.find_maximum_category(item_list, target_item, max_count=2))
                        else:
                            raise NotImplementedError
                    else:
                        if 'MP' in task or 'EP' in task:
                            target_category = self.item2category[target_item][-1]
                        elif 'LP' in task:
                            target_category = self.find_maximum_category(item_list, target_item)[-1]
                        else:
                            raise NotImplementedError
                    input_field_data.update({
                        'target_category': target_category,
                        'SeqRec_Result': [self.get_item_index(_) for _ in item_list]
                    })
                else:
                    target_category = self.item2category[target_item][-1]
                    input_field_data.update({
                        'target_category': target_category,
                        'SeqRec_Result': self.RLHFSeqRec_Result[user]
                    })
                category_item_count = min(len(self.category2item[target_category]), item_count) if self.mode == 'train' else 5
                if self.mode != 'train':
                    target_category_item_count = 5
                    output_category_item_count = 5
                else:
                    if 'LP' in task:
                        target_category_item_count = random.choice(range(category_item_count))+1
                        output_category_item_count = target_category_item_count-1
                    elif 'MP' in task:
                        target_category_item_count = random.choice(range(category_item_count))
                        output_category_item_count = target_category_item_count
                    else:
                        target_category_item_count = random.choice(range(category_item_count+1))
                        output_category_item_count = target_category_item_count
                input_field_data.update({
                    'item_count': item_count,
                    'category_proportion': f"{int(target_category_item_count/item_count*10)}0%",
                    'category_count': output_category_item_count,
                })
            elif task in ["RLHFSeqRanking"]:
                if self.mode == 'train':
                    candidate_num = random.choice(range(item_count, self.args.candidate_num)) + 1
                    output_items = [target_item]
                    ranking_candidate = output_items + random.choices(list(set(self.metas.keys()) - set(output_items)), k=candidate_num-1)
                    random.shuffle(ranking_candidate)
                else:
                    ranking_candidate = self.ranking_candidate[user][:self.args.candidate_num - 1]
                    insert_idx = idx % self.args.candidate_num
                    ranking_candidate.insert(insert_idx, target_item)
                input_field_data.update({
                    'candidate_titles': ', '.join([f"'{self.get_item_index(_)}'" for _ in ranking_candidate]),
                    'candidate_items': ranking_candidate
                })
            elif task in ["RLHFItemCount"]:
                item_count = self.args.topk + 1 + idx % 5
                input_field_data.update({'item_count': item_count})

        input_text = template_selected.get_input_text(input_field_data, llama2_chat_template=self.args.llama2_chat_template).strip()
        output_text = template_selected.get_output_text(output_field_data).strip()
        out_dict = {
            'task': task,
            'input_text': input_text,
            'output_text': output_text,
            'input_field_data': input_field_data
        }
        return out_dict

    def __getitem__(self, idx):
        return self.complete_datum_info[idx]

    def collate_fn(self, batch):
        batch_entry = {}
        tasks = []
        output_text = []
        input_text = []
        input_field_data = []
        for i, entry in enumerate(batch):
            if isinstance(entry, dict):
                entry = [entry]
            for _ in entry:
                if 'task' in _:
                    tasks.append(_['task'])
                if 'input_text' in _:
                    input_text.append(_['input_text'])
                if 'input_field_data' in _:
                    input_field_data.append(_['input_field_data'])
                if 'output_text' in _:
                    output_text.append(_['output_text'])
        batch_entry['input_text'] = input_text
        batch_entry['output_text'] = output_text

        batch_entry['input_data'] = side_tokenizer(batch_entry['input_text'],
                                                   'left', self.tokenizer,
                                                   padding=True, truncation=True,
                                                   max_length=self.args.max_token_length,
                                                   return_tensors='pt').to(self.args.gpu).data

        if len(input_field_data) > 0:
            batch_entry['input_field_data'] = input_field_data
        if len(tasks) > 0:
            batch_entry['task'] = tasks
        return batch_entry

    '''
    1. 模型生成的推荐列表 符合用户的控制意图。
    2. 模型生成的推荐列表 具有更低的毒性：长度错误，重复item，不存在item（不存在候选集）。
    '''
    def get_list_reward_hard_encode_new(self, task, input_field_data, title_list, new_data=False):
        item_count = input_field_data['item_count']
        list_length = len(title_list)
        item_count_ratio = min(item_count/list_length, list_length/item_count)
        repeat_score, exceed_score, not_exist_score, target_score = -1.0, -1.0, -1.0, 1.0       # NR-5
        # repeat_score, exceed_score, not_exist_score, target_score = 0.0, 0.0, 0.0, 1.0            # NR-6
        candidates = input_field_data.get('candidate_titles') or self.title2item
        user = input_field_data['user']
        sub_sequential = input_field_data['sub_sequential']
        item_list = [self.title2item[_][0] if _ in self.title2item else list(self.metas.keys())[0] for _ in title_list]

        target_item = input_field_data['target_item']
        item_ranking = get_item_ranking(self.args.backup_ip, [user], [sub_sequential], [item_list], port=self.teacher_port)['ranking'][0]
        if task == 'RLHFSeqRanking':
            item_ranking = np.argsort(item_ranking)
            item_ranking = np.argsort(item_ranking).tolist()

        item_score = []
        list_score = []
        category_count = 0
        for idx, (_, __) in enumerate(zip(item_list, item_ranking)):
            temp_score = 1.0 if _ == target_item else 1.0 / math.log2(__ + 2)
            if idx >= item_count:
                item_score.append(exceed_score)
                list_score.append(exceed_score)
            elif title_list[idx] not in candidates:
                item_score.append(not_exist_score)
                list_score.append(not_exist_score)
            elif _ in item_list[:idx]:
                item_score.append(repeat_score)
                list_score.append(repeat_score)
            elif task in ['RLHFSeqRec', 'RLHFItemCount', 'RLHFSeqRanking']:
                item_score.append(temp_score)
                list_score.append(temp_score/math.log2(idx + 2))
            elif task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec']:
                target_category = input_field_data['target_category']
                if (task == 'RLHF+PersonalControlRec' and _ in self.category2item[target_category]) or \
                        (task == 'RLHF-PersonalControlRec' and _ not in self.category2item[target_category]):
                    item_score.append(1.0*0.8 + temp_score*0.2)
                    list_score.append(1.0*0.8 + temp_score/math.log2(idx + 2)*0.2)
                else:
                    item_score.append(temp_score*0.2)
                    list_score.append(temp_score/math.log2(idx + 2)*0.2)
            elif task == 'RLHFPersonalCategoryRate':
                target_category = input_field_data['target_category']
                if _ in self.category2item[target_category]:
                    if category_count < input_field_data['category_count']:
                        item_score.append(1.0*0.8 + temp_score*0.2)
                        list_score.append(1.0*0.8 + temp_score/math.log2(idx + 2)*0.2)
                    else:
                        item_score.append(temp_score*0.2)
                        list_score.append(temp_score/math.log2(idx + 2)*0.2)
                    category_count += 1
                else:
                    item_score.append(0.5*0.8 + temp_score*0.2)
                    list_score.append(0.5*0.8 + temp_score/math.log2(idx + 2)*0.2)
            else:
                raise NotImplementedError
        item_score = torch.tensor(item_score, device=self.args.gpu)
        list_score = torch.tensor(list_score, device=self.args.gpu)
        if task in ['RLHFSeqRec', 'RLHFItemCount', 'RLHFSeqRanking']:
            max_list_reward = self.best_ranking_score[item_count]
        elif task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec']:
            max_list_reward = item_count*0.8+self.best_ranking_score[item_count]*0.2
        elif task == 'RLHFPersonalCategoryRate':
            max_list_reward = (input_field_data['category_count']+(item_count-input_field_data['category_count'])*0.5)*0.8+self.best_ranking_score[item_count]*0.2
        else:
            raise NotImplementedError
        item_reward = item_score
        list_reward = list_score.sum()/max_list_reward

        res = [[task, title_list, list_reward, item_reward]]
        if new_data:
            new_title_list = self.vague_selecting(title_list, candidates)
            new_title_idx = torch.argsort(item_reward, descending=True)
            new_title_list_sorted = [new_title_list[idx] for idx in new_title_idx]
            if len(new_title_list_sorted) < item_count:
                extend_count = item_count-len(new_title_list_sorted)
                new_title_list_sorted.extend(random.choices(list(set(candidates)-set(new_title_list_sorted)), k=extend_count))
            new_title_list_sorted = new_title_list_sorted[:item_count]
            res += self.get_list_reward_hard_encode_new(task, input_field_data, new_title_list_sorted, False)
        return res

    def get_list_reward_hard_encode_NR_7(self, task, input_field_data, title_list, new_data=False):
        # ranking_score_frac, task_score_frac = 0.2, 0.8            # NR-11
        # ranking_score_frac, task_score_frac = 0.5, 0.5            # NR-13
        ranking_score_frac, task_score_frac = self.args.reward_alpha, 1.0-self.args.reward_alpha            # NR-13
        item_count = input_field_data['item_count']
        list_length = len(title_list)
        item_count_ratio = min(item_count/list_length, list_length/item_count)
        repeat_score, exceed_score, not_exist_score, in_history_score, target_score = -1.0, -1.0, -1.0, -1.0, 1.0
        candidates = input_field_data.get('candidate_titles') or self.title2item
        user = input_field_data['user']
        sub_sequential = input_field_data['sub_sequential']
        item_list = [self.title2item[_][0] if _ in self.title2item else list(self.metas.keys())[0] for _ in title_list]

        target_item = input_field_data['target_item']
        item_ranking = get_item_ranking(self.args.backup_ip, [user], [sub_sequential], [item_list], port=self.teacher_port)['ranking'][0]
        if task == 'RLHFSeqRanking':
            item_ranking = np.argsort(item_ranking)
            item_ranking = np.argsort(item_ranking).tolist()

        rank_score = []
        rank_corrected_score = []
        task_score = []
        in_category_count, out_category_count = 0, 0
        target_category = input_field_data['target_category']
        for idx, (_, __) in enumerate(zip(item_list, item_ranking)):
            if idx >= item_count:
                rank_score.append(exceed_score)
                rank_corrected_score.append(exceed_score)
                task_score.append(exceed_score)
            elif title_list[idx] not in candidates:
                rank_score.append(not_exist_score)
                rank_corrected_score.append(not_exist_score)
                task_score.append(not_exist_score)
            elif _ in item_list[:idx]:
                rank_score.append(repeat_score)
                rank_corrected_score.append(repeat_score)
                task_score.append(repeat_score)
            elif _ in input_field_data['sub_sequential']:
                rank_score.append(in_history_score)
                rank_corrected_score.append(in_history_score)
                task_score.append(in_history_score)
            else:
                # temp_score = 1.0 if _ == target_item else self.ranking_score_func(__)       # NR-12
                temp_score = 1.0 if _ == target_item else self.ranking_score_func(__+1)       # NR-13
                rank_score.append(temp_score)
                rank_corrected_score.append(temp_score / math.log2(idx + 2))
                if _ in self.category2item[target_category]:
                    in_category_count += 1
                else:
                    out_category_count += 1
                if task in ['RLHFSeqRec', 'RLHFItemCount', 'RLHFSeqRanking']:
                    pass
                elif task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec']:
                    if _ in self.category2item[target_category]:
                        if '+' in task:
                            task_score.append(1.0)
                        else:
                            task_score.append(0.0)
                    elif _ not in self.category2item[target_category]:
                        if '-' in task:
                            task_score.append(1.0)
                        else:
                            task_score.append(0.0)
                elif task.startswith('RLHFPersonalCategoryRate'):
                    if 'LP' in task:
                        # if in_category_count <= input_field_data['category_count']:
                        #     task_score.append(0.5)
                        # else:
                        #     if _ in self.category2item[target_category]:
                        #         task_score.append(0.0)
                        #     else:
                        #         task_score.append(1.0)
                        # NR-19
                        if out_category_count > (input_field_data['item_count']-input_field_data['category_count']):
                            task_score.append(0.5)
                        else:
                            if _ not in self.category2item[target_category]:
                                task_score.append(1.0)
                            elif in_category_count < input_field_data['category_count']:
                                task_score.append(0.5)
                            else:
                                task_score.append(0.0)

                    elif 'MP' in task:
                        # if out_category_count < (input_field_data['item_count'] - input_field_data['category_count']):
                        #     task_score.append(0.5)
                        # else:
                        #     if _ in self.category2item[target_category]:
                        #         task_score.append(1.0)
                        #     else:
                        #         task_score.append(0.0)
                        # NR-19
                        if in_category_count > input_field_data['category_count']:
                            task_score.append(0.5)
                        else:
                            if _ in self.category2item[target_category]:
                                task_score.append(1.0)
                            elif out_category_count < (input_field_data['item_count']-input_field_data['category_count']):
                                task_score.append(0.5)
                            else:
                                task_score.append(0.0)

                    elif 'EP' in task:
                        if _ in self.category2item[target_category]:
                            if in_category_count <= input_field_data['category_count']:
                                task_score.append(1.0)
                            else:
                                task_score.append(0.0)
                        else:
                            if in_category_count >= input_field_data['category_count']:
                                task_score.append(1.0)
                            else:
                                if out_category_count <= (input_field_data['item_count']-input_field_data['category_count']):
                                    task_score.append(0.5)
                                else:
                                    task_score.append(0.0)
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
        rank_score = torch.tensor(rank_score, device=self.args.gpu)
        rank_corrected_score = torch.tensor(rank_corrected_score, device=self.args.gpu)
        task_score = torch.tensor(task_score, device=self.args.gpu)
        if task in ['RLHFSeqRec', 'RLHFItemCount', 'RLHFSeqRanking']:
            item_reward = rank_score
            # list_task_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]       # NR-18
            list_task_reward = 1.0/math.log2(item_list.index(target_item)+2) if target_item in item_list else 0.0
        elif task in ['RLHF+PersonalControlRec', 'RLHF-PersonalControlRec']:
            item_reward = rank_score*ranking_score_frac + task_score*task_score_frac
            target_count = min(item_count, len(self.category2item[target_category])) if '+' in task else item_count
            # list_task_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]*ranking_score_frac + task_score.sum()/target_count*task_score_frac       # NR-18
            if '+' in task:
                # list_task_reward = 1.0 / math.log2(target_count-in_category_count+2)        # NR-15, 17
                list_task_reward = 1.0/(target_count-in_category_count+1)        # NR-18
                # list_task_reward = 2*in_category_count/target_count-1
            else:
                # list_task_reward = 1.0 / math.log2(target_count-out_category_count+2)        # NR-15, 17
                list_task_reward = 1.0/(target_count-out_category_count+1)        # NR-18
                # list_task_reward = 2*out_category_count/target_count-1
        elif task.startswith('RLHFPersonalCategoryRate'):
            item_reward = rank_score*ranking_score_frac + task_score*task_score_frac
            # list_task_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]*ranking_score_frac + task_score.sum()/target_count*task_score_frac       # NR-18
            if 'LP' in task and in_category_count <= input_field_data['category_count']:
                list_task_reward = 1.0
            elif 'MP' in task and in_category_count >= input_field_data['category_count']:
                list_task_reward = 1.0
            elif 'EP' in task and in_category_count == input_field_data['category_count']:
                list_task_reward = 1.0
            else:
                # list_task_reward = 1.0/(math.log2(abs(in_category_count-input_field_data['category_count'])+2))   # NR-15, 17
                list_task_reward = 1.0/(abs(in_category_count-input_field_data['category_count'])+1)   # NR-18
                # list_task_reward = -abs(in_category_count-input_field_data['category_count'])/input_field_data['item_count']
        else:
            raise NotImplementedError

        # list_reward = list_task_reward        # NR-14, 15
        list_reward = rank_corrected_score.sum()/self.best_ranking_score[item_count]*ranking_score_frac + list_task_reward*task_score_frac        # NR-17-21
        # res = [[task, title_list, list_reward, item_reward]]        # NR-10, 17, 18, 19
        # res = [[task, title_list, list_reward, item_reward*0.1]]    # NR-11
        # res = [[task, title_list, list_reward, item_reward*0.3]]    # NR-12
        # res = [[task, title_list, list_reward, item_reward*0.5]]    # NR-13
        # res = [[task, title_list, list_reward, item_reward*0.3]]    # NR-14, 15
        # res = [[task, title_list, list_reward*100, item_reward]]    # NR-16
        res = [[task, title_list, list_reward*10, item_reward]]    # NR-20
        # res = [[task, title_list, list_reward*10, item_reward*2]]    # NR-21
        if new_data:
            new_title_list = self.vague_selecting(title_list, candidates)
            new_title_idx = torch.argsort(item_reward, descending=True)
            new_title_list_sorted = [new_title_list[idx] for idx in new_title_idx]
            if len(new_title_list_sorted) < item_count:
                extend_count = item_count-len(new_title_list_sorted)
                new_title_list_sorted.extend(random.choices(list(set(candidates)-set(new_title_list_sorted)), k=extend_count))
            new_title_list_sorted = new_title_list_sorted[:item_count]
            res += self.get_list_reward_hard_encode_NR_7(task, input_field_data, new_title_list_sorted, False)
        return res

    def ranking_score_func(self, idx):
        if 'NR-9' in self.args.model_name:
            return 1.0-idx/len(self.metas)      # NR-9
        else:
            return 1.0/math.log2(idx+2)         # NR-8


Train_task_group_mapping = {
    "RLHFSeqRec": SeqRec_group,
    "RLHFSeqRanking": SeqRanking_group,
    "RLHF+PersonalControlRec": PersonalControlRec_group,
    "RLHF-PersonalControlRec": PersonalControlRec_group,
    "RLHFPersonalCategoryRate": PersonalCategoryRate_group,
    "RLHFPersonalCategoryRateLP": PersonalCategoryRateLP1_group,
    "RLHFPersonalCategoryRateMP": PersonalCategoryRateMP_group,
    "RLHFPersonalCategoryRateEP": PersonalCategoryRateEP_group,
}

Val_task_group_mapping = {
    "RLHFSeqRec": SeqRec_group,
    "RLHFSeqRanking": SeqRanking_group,
    "RLHFItemCount": SeqRec_group,
    "RLHF+PersonalControlRec": PersonalControlRec_group,
    "RLHF-PersonalControlRec": PersonalControlRec_group,
    "RLHFPersonalCategoryRate": PersonalCategoryRate_group,
    "RLHFPersonalCategoryRateLP": PersonalCategoryRateLP_group,
    "RLHFPersonalCategoryRateMP": PersonalCategoryRateMP_group,
    "RLHFPersonalCategoryRateEP": PersonalCategoryRateEP_group,
}

Test_task_group_mapping = {
    "RLHFSeqRec": SeqRec_group,
    "RLHFSeqRanking": SeqRanking_group,
    "RLHFItemCount": SeqRec_group,
    "RLHF+PersonalControlRec": PersonalControlRec_group,
    "RLHF-PersonalControlRec": PersonalControlRec_group,
    "RLHFPersonalCategoryRate": PersonalCategoryRate_group
}
