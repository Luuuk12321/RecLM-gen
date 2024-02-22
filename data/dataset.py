import copy
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from SFT.SFT_templates import *
from utils.utils import *


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        self.args = args
        self.mode = mode
        self.tokenizer = tokenizer
        self.teacher_port = self.args.teacher_port
        self.complete_datum_info = load_pickle(self.args.data_file)
        assert len(self.complete_datum_info) > 0

    def __len__(self):
        return len(self.complete_datum_info)

    def __getitem__(self, idx):
        return self.complete_datum_info[idx]

    def collate_fn(self, batch):
        batch_entry = {}
        complete_text = []
        for i, entry in enumerate(batch):
            assert 'input_text' in entry and 'output_text' in entry
            for k, v in entry.items():
                if k not in batch_entry:
                    batch_entry[k] = []
                batch_entry[k].append(v)
            complete_text.append(get_complete_text(entry['input_text'], entry['output_text']))
        batch_entry['complete_text'] = complete_text
        batch_entry['input_data'] = side_tokenizer(batch_entry['input_text'],
                                                   'left', self.tokenizer,
                                                   padding=True, truncation=True,
                                                   max_length=self.args.max_token_length,
                                                   return_tensors='pt').to(self.args.gpu).data
        batch_entry['output_data'] = side_tokenizer(batch_entry['output_text'],
                                                    'right', self.tokenizer,
                                                    padding=True, truncation=True,
                                                    max_length=self.args.gen_max_length,
                                                    return_tensors='pt').to(self.args.gpu).data
        batch_entry['complete_text_data'] = {
            'input_ids':
                torch.cat([batch_entry['input_data']['input_ids'], batch_entry['output_data']['input_ids'][:, 1:]], dim=-1),
            'attention_mask':
                torch.cat([batch_entry['input_data']['attention_mask'], batch_entry['output_data']['attention_mask'][:, 1:]], dim=-1)
        }
        prompt_length = batch_entry['input_data']['input_ids'].shape[-1]
        batch_entry['complete_label_ids'] = copy.deepcopy(batch_entry['complete_text_data']['input_ids'])
        batch_entry['complete_label_ids'][..., :prompt_length] = -100
        batch_entry['complete_label_ids'][batch_entry['complete_label_ids'] == self.tokenizer.pad_token_id] = -100

        return batch_entry

