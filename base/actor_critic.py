import os

from peft import TaskType, LoraConfig, inject_adapter_in_model, LoraModel

import torch
from torch import nn
import bitsandbytes as bnb
from einops.layers.torch import Rearrange
from transformers import T5Config, AutoConfig, AutoTokenizer, T5ForConditionalGeneration, \
    AutoModelForCausalLM, BitsAndBytesConfig

from utils.utils import eval_decorator, shift, huggingface_proxies


def layer_init(layer, std=2**0.5):
    nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layer.weight, gain=std)
    return layer


class ValueRewardHead(nn.Module):
    def __init__(self, hidden_size, inference):
        super().__init__()
        self.head = nn.Sequential(
            layer_init(nn.Linear(hidden_size, 64)),
            nn.GELU(),
            layer_init(nn.Linear(64, 1), std=1.0),
            Rearrange('... 1 -> ...'),
        )
        self.requires_grad_(not inference)

    def forward(self, emb: torch.Tensor):
        # emb_norm = emb / emb.square().mean(dim=1, keepdim=True)
        return self.head(emb)


def param_init(named_param):
    for n, p in named_param.items():
        if p.ndim >= 2 and 'lora_A' in n:
            try:
                nn.init.orthogonal_(p, gain=2**0.5)
            except RuntimeError as e:
                print(n, e)
        if 'lora_B' in n:
            nn.init.zeros_(p)


class ActorCritic(nn.Module):
    def __init__(self, args, device, actor_lora_scope='actor', critic_lora_scope='critic'):
        super().__init__()
        self.args = args
        self.model_config = self.create_model_config()
        self.tokenizer = self.create_tokenizer()
        self.model = self.create_model(device)

        self.actor_lora_scope = actor_lora_scope
        self.critic_lora_scope = critic_lora_scope
        if self.args.train_stage in ['SFT', 'SFT_Test', 'SFT_Merge']:
            if self.args.full_fine_tune:
                self.model.requires_grad_()
            elif self.args.SFT_actor_lora_r > 0:
                self.actor_lora_config = self.create_lora_config(
                    self.actor_lora_scope,
                    False,
                    self.args.SFT_actor_lora_r,
                    self.args.SFT_actor_lora_a
                )
                self.lora_model = LoraModel(self.model, self.actor_lora_config, adapter_name=self.actor_lora_scope)
                param_init(self.actor_named_parameters)
            else:
                raise NotImplementedError

        if self.args.train_stage in ['RLHF', 'RLHF_Test', 'RLHF_Merge']:
            assert (not self.args.full_fine_tune) and self.args.RLHF_actor_lora_r > 0 and self.args.RLHF_critic_lora_r > 0
            self.actor_lora_config = self.create_lora_config(
                self.actor_lora_scope,
                False,
                self.args.RLHF_actor_lora_r,
                self.args.RLHF_actor_lora_a
            )
            self.lora_model = LoraModel(self.model, self.actor_lora_config, adapter_name=self.actor_lora_scope)
            param_init(self.actor_named_parameters)

            self.critic_lora_config = self.create_lora_config(
                self.critic_lora_scope,
                False,
                self.args.RLHF_critic_lora_r,
                self.args.RLHF_critic_lora_a
            )
            inject_adapter_in_model(self.critic_lora_config, self.model, adapter_name=self.critic_lora_scope)
            self.critic_value_head = ValueRewardHead(self.model_config.hidden_size, inference=False)
            param_init(self.critic_named_parameters)

            self.critic_value_head = self.critic_value_head.to(device).bfloat16()

    def save_parameters(self, name='Epoch00'):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        param_dict = {}
        if self.args.train_stage in ['SFT', 'RLHF']:
            param_dict.update(self.actor_named_parameters)
        if self.args.train_stage in ['RLHF']:
            param_dict.update(self.critic_named_parameters)
        torch.save(param_dict, os.path.join(self.args.output, f"{name}_{self.args.train_stage}.pth"))

    def load_parameters(self, load_file):
        # self.args.load: xxx/{name}_{train_stage}
        if load_file is not None and os.path.exists(f"{load_file}.pth"):
            state_dict = torch.load(f"{load_file}.pth", map_location=self.device)
            results = self.load_state_dict(state_dict, strict=False)
            assert len(results.unexpected_keys) == 0, results.unexpected_keys
            print(f'{self.args.train_stage} model loaded of file {load_file}')
            return int(load_file.split('/')[-1][5:7]) if self.args.train_stage in ['SFT', 'SFT_Test', 'SFT_Merge'] else int(load_file.split('/')[-1][:-9])
        else:
            return 0

    def print_trainable_parameters(self):
        trainable_params = {
            self.actor_lora_scope: 0,
            self.critic_lora_scope: 0,
            "base": 0
        }
        all_param = {
            self.actor_lora_scope: 0,
            self.critic_lora_scope: 0,
            "base": 0
        }
        for _, param in self.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            if self.actor_lora_scope in _:
                all_param[self.actor_lora_scope] += num_params
            elif self.critic_lora_scope in _:
                all_param[self.critic_lora_scope] += num_params
            else:
                all_param["base"] += num_params

            if param.requires_grad:
                if self.actor_lora_scope in _:
                    trainable_params[self.actor_lora_scope] += num_params
                elif self.critic_lora_scope in _:
                    trainable_params[self.critic_lora_scope] += num_params
                else:
                    trainable_params["base"] += num_params

        print(f'trainable_params: {" - ".join([str(_) for _ in trainable_params.values()])} | '
              f'all_param: {" - ".join([str(_) for _ in all_param.values()])} | '
              f'percentage: {sum(trainable_params.values())/sum(all_param.values()):.4f}')

    def create_model_config(self):
        if 't5' in self.args.backbone:
            config_class = T5Config
        else:
            config_class = AutoConfig

        config = config_class.from_pretrained(self.args.backbone, proxies=huggingface_proxies if self.args.proxy else None)
        config.dropout_rate = self.args.dropout
        config.dropout = self.args.dropout
        config.attention_dropout = self.args.dropout
        config.activation_dropout = self.args.dropout
        return config

    def create_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.backbone,
            proxies=huggingface_proxies if self.args.proxy else None,
        )
        # tokenizer.add_tokens(['\n'] + [f'<{i+1}>' for i in range(20)])
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        self.model_config.pad_token_id = tokenizer.pad_token_id
        return tokenizer

    def create_model(self, device):
        if 't5' in self.args.backbone:
            model_class = T5ForConditionalGeneration
        else:
            model_class = AutoModelForCausalLM

        bnb_config = BitsAndBytesConfig(
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        if self.args.quantization:
            model = model_class.from_pretrained(self.args.backbone,
                                                config=self.model_config,
                                                quantization_config=bnb_config,
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                use_flash_attention_2=self.args.FA2,
                                                proxies=huggingface_proxies if self.args.proxy else None
                                                )
        else:
            model = model_class.from_pretrained(self.args.backbone,
                                                config=self.model_config,
                                                device_map=device,
                                                torch_dtype=torch.bfloat16,
                                                use_flash_attention_2=self.args.FA2,
                                                proxies=huggingface_proxies if self.args.proxy else None
                                                )
        model.requires_grad_(False)
        return model

    def find_all_linear_names(self, scope):
        # target_name = ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        target_name = ['']
        cls = bnb.nn.Linear4bit if self.args.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in self.model.named_modules():
            if 'lora' in name:
                continue
            if isinstance(module, cls) and any([tgt in name for tgt in target_name]):
                lora_module_names.add(name)

        if scope == self.actor_lora_scope:
            if self.args.lm_head:
                lora_module_names.remove('lm_head')
                self.model.lm_head.weight.requires_grad = True
        else:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_lora_config(self, scope, inference_mode, r, alpha):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=inference_mode,
            r=r,
            target_modules=self.find_all_linear_names(scope),
            lora_alpha=alpha,
            lora_dropout=self.args.lora_dropout,
            init_lora_weights=True,
            bias="none",
        )
        return lora_config

    @property
    def device(self):
        return self.model.device

    @property
    def actor_named_parameters(self):
        if self.args.full_fine_tune:
            return {n: p for n, p in self.named_parameters() if p.requires_grad}
        else:
            return {n: p for n, p in self.named_parameters() if self.actor_lora_scope in n or (n == 'model.lm_head.weight' and self.args.lm_head)}

    @property
    def critic_named_parameters(self):
        return {n: p for n, p in self.named_parameters() if self.critic_lora_scope in n}

    @property
    def actor_model(self):
        if not hasattr(self, 'lora_model'):
            return self.model
        self.lora_model.enable_adapter_layers()
        self.lora_model.set_adapter(self.actor_lora_scope)
        return self.lora_model

    @property
    def base_model(self):
        if not hasattr(self, 'lora_model'):
            return self.model
        self.lora_model.disable_adapter_layers()
        return self.lora_model

    @torch.no_grad()
    @eval_decorator
    def generate(self, scope, input_ids, **kwargs):
        if not hasattr(self, 'lora_model'):
            return self.model.generate(input_ids=input_ids, **kwargs)

        if scope == self.actor_lora_scope:
            self.lora_model.enable_adapter_layers()
            self.lora_model.set_adapter(scope)
            return self.lora_model.generate(input_ids=input_ids, **kwargs)
        elif scope == self.critic_lora_scope:
            raise NotImplementedError
        else:
            self.lora_model.disable_adapter_layers()
            return self.lora_model(input_ids=input_ids, **kwargs)

    def forward(self, scope, input_ids, **kwargs):
        if not hasattr(self, 'lora_model'):
            return self.model(input_ids=input_ids, **kwargs)

        if scope == self.actor_lora_scope:
            self.lora_model.enable_adapter_layers()
            self.lora_model.set_adapter(scope)
            return self.lora_model(input_ids=input_ids, **kwargs)
        elif scope == self.critic_lora_scope:
            self.lora_model.enable_adapter_layers()
            self.lora_model.set_adapter(scope)
            critic_token_embed = self.lora_model(input_ids=input_ids, output_hidden_states=True, **kwargs).hidden_states[-1]
            action_value = self.critic_value_head(critic_token_embed)
            return shift(action_value, shift=1, dim=-1)
        else:
            self.lora_model.disable_adapter_layers()
            return self.lora_model(input_ids=input_ids, **kwargs)
