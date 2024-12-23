from diffusers import DDPMScheduler
from colorama import Fore, Back, Style
import torch
from safetensors.torch import load_file, save_file
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torch.nn as nn
from torchvision import transforms as T, utils
import numpy as np
import argparse
import os
from argparse import Namespace
import time
import datetime
from diffusers import StableDiffusionPipeline


class LoraInjectedLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=False,
            r=4,
            dropout_p=0.1,
            scale=1.0,
    ):
        super().__init__()

        if r > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {r} must be less or equal than {min(in_features, out_features)}"
            )
        self.r = r
        self.linear = nn.Linear(in_features, out_features, bias)
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.dropout = nn.Dropout(dropout_p)
        self.lora_up = nn.Linear(r, out_features, bias=False)
        self.scale = scale
        self.selector = nn.Identity()

    def forward(self, input):
        return (
                self.linear(input)
                + self.dropout(self.lora_up(self.selector(self.lora_down(input))))
                * self.scale
        )


def merge_to_sd_model(text_encoder, unet, model, ratio=1.0, isdiffuser=False):
    # create module map
    if isdiffuser:
        strLinear = "LoRACompatibleLinear"
        strConv = "LoRACompatibleConv"
    else:
        strLinear = "Linear"
        strConv = "Conv2d"
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = "lora_te"
            target_replace_modules = ["CLIPAttention", "CLIPMLP"]
        else:
            prefix = "lora_unet"
            target_replace_modules = (
                    ["Transformer2DModel", "Attention"] + ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if i == 1:
                        if child_module.__class__.__name__ == strLinear or child_module.__class__.__name__ == strConv:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            name_to_module[lora_name] = child_module
                    else:
                        if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            name_to_module[lora_name] = child_module
    lora_sd = load_file(model)
    print(f"merging...")
    for key in lora_sd.keys():
        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            # find original module for this lora
            module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
            if module_name not in name_to_module:
                print(f"no module found for LoRA weight: {key}")
                continue
            module = name_to_module[module_name]
            # print(f"apply {key} to {module}")

            down_weight = lora_sd[key]
            up_weight = lora_sd[up_key]

            dim = down_weight.size()[0]
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim
            # W <- W + U * D
            weight = module.weight
            up_weight = up_weight.to(weight.dtype)
            down_weight = down_weight.to(weight.dtype)
            # print(module_name, down_weight.size(), up_weight.size())
            if len(weight.size()) == 2:
                # linear
                weight = weight + (ratio * (up_weight @ down_weight) * scale)
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                        weight
                        + (ratio
                           * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(
                            2).unsqueeze(3)
                           * scale).to(weight.dtype)
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # print(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + (ratio * conved * scale).to(weight.dtype)
            module.weight = torch.nn.Parameter(weight)


def merge_to_sd_model_train(text_encoder, unet, model, ratio=1.0, dropout_p=0.0):
    # create module map
    num_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = "lora_te"
            target_replace_modules = ["CLIPAttention", "CLIPMLP"]
        else:
            prefix = "lora_unet"
            target_replace_modules = (
                    ["Transformer2DModel", "Attention"] + ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if i == 1:
                        if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            name_list = child_name.split('.')
                            temp = module
                            for n in name_list[:-1]:
                                if n[0] not in num_list:
                                    temp = getattr(temp, n)
                                else:
                                    temp = temp[int(n)]
                            if name_list[-1][0] in num_list:
                                name_list[-1] = int(name_list[-1])
                            name_to_module[lora_name] = {'child': child_module, 'name': name_list[-1], 'parent': temp}
                    else:
                        if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            name_list = child_name.split('.')
                            temp = module
                            for n in name_list[:-1]:
                                if n[0] not in num_list:
                                    temp = getattr(temp, n)
                                else:
                                    temp = temp[int(n)]
                            if name_list[-1][0] in num_list:
                                name_list[-1] = int(name_list[-1])
                            name_to_module[lora_name] = {'child': child_module, 'name': name_list[-1], 'parent': temp}
    lora_sd = load_file(model)
    print(f"merging...")
    require_grad_params = []
    for key in lora_sd.keys():
        if "lora_down" in key:
            up_key = key.replace("lora_down", "lora_up")
            alpha_key = key[: key.index("lora_down")] + "alpha"

            # find original module for this lora
            module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
            if module_name not in name_to_module:
                print(f"no module found for LoRA weight: {key}")
                continue
            module = name_to_module[module_name]['child']
            parent = name_to_module[module_name]['parent']
            p_m_name = name_to_module[module_name]['name']
            # print(f"apply {key} to {module}")

            down_weight = lora_sd[key]
            up_weight = lora_sd[up_key]

            dim = down_weight.size()[0]
            alpha = lora_sd.get(alpha_key, dim)
            scale = alpha / dim

            # W <- W + U * D
            weight = module.weight
            bias = module.bias
            up_weight = up_weight.to(weight.dtype)
            down_weight = down_weight.to(weight.dtype)
            # print(module_name, down_weight.size(), up_weight.size())
            if len(weight.size()) == 2:
                # linear
                temp = LoraInjectedLinear(
                    module.in_features,
                    module.out_features,
                    bias is not None,
                    r=dim,
                    dropout_p=dropout_p,
                    scale=scale * ratio
                )
                temp.linear.weight = weight
                if bias is not None:
                    temp.linear.bias = bias
                temp.lora_up.weight = nn.Parameter(up_weight)
                temp.lora_down.weight = nn.Parameter(down_weight)
                if p_m_name.__class__.__name__ == 'str':
                    parent._modules[p_m_name] = temp
                    require_grad_params.append(parent._modules[p_m_name].lora_up.parameters())
                    require_grad_params.append(parent._modules[p_m_name].lora_down.parameters())
                    parent._modules[p_m_name].lora_up.weight.requires_grad = True
                    parent._modules[p_m_name].lora_down.weight.requires_grad = True
                else:
                    parent[p_m_name] = temp
                    require_grad_params.append(parent[p_m_name].lora_up.parameters())
                    require_grad_params.append(parent[p_m_name].lora_down.parameters())
                    parent[p_m_name].lora_up.weight.requires_grad = True
                    parent[p_m_name].lora_down.weight.requires_grad = True
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                        weight
                        + (ratio
                           * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(
                            2).unsqueeze(3)
                           * scale).to(weight.dtype)
                )
                module.weight = torch.nn.Parameter(weight)
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # print(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + (ratio * conved * scale).to(weight.dtype)
                module.weight = torch.nn.Parameter(weight)
    return require_grad_params


def save_lora(textencoder, unet, output_path):
    prefix = "lora_te"
    target_replace_modules = ["CLIPAttention", "CLIPMLP"]
    sd = {}
    for name, module in textencoder.named_modules():
        if module.__class__.__name__ in target_replace_modules:
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ == "LoraInjectedLinear":
                    lora_name = prefix + "_" + name + "_" + child_name
                    sd[lora_name.replace('.', '_') + '.lora_up.weight'] = child_module.lora_up.weight
                    sd[lora_name.replace('.', '_') + '.lora_down.weight'] = child_module.lora_down.weight
                    sd[lora_name.replace('.', '_') + '.alpha'] = torch.tensor(32., dtype=torch.float16)
    prefix = "lora_unet"
    target_replace_modules = (
            ["Transformer2DModel", "Attention"] + ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    )

    for name, module in unet.named_modules():
        if module.__class__.__name__ in target_replace_modules:
            for child_name, child_module in module.named_modules():
                if child_module.__class__.__name__ == "LoraInjectedLinear":
                    lora_name = prefix + "_" + name + "_" + child_name
                    sd[lora_name.replace('.', '_') + '.lora_up.weight'] = child_module.lora_up.weight
                    sd[lora_name.replace('.', '_') + '.lora_down.weight'] = child_module.lora_down.weight
                    sd[lora_name.replace('.', '_') + '.alpha'] = torch.tensor(32., dtype=torch.float16)
    save_file(sd, output_path)

class diffusion_step:
    def __init__(self, tokenizer, textencoder, unet, vae, device):
        self.tokenizer = tokenizer
        self.textencoder = textencoder
        self.unet = unet
        self.vae = vae
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                        num_train_timesteps=1000)
        self.device = device

    def get_embeding(self, prompts):
        input_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        embedding = self.textencoder(input_ids.to(self.device))[0]
        return embedding.detach()

    def step(self, images, prompts):
        batch_size = images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()
        with torch.no_grad():
            embedding = self.get_embeding(prompts)
        latents = self.vae.encode(images * 2 - 1).latent_dist.sample() * 0.18215
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=embedding).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss

def get_dm(args, device):
    pipe = StableDiffusionPipeline.from_pretrained(args.diffusion_path)
    tokenizer = pipe.tokenizer
    textencoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    del pipe
    if args.lora_path is not None:
        merge_to_sd_model(textencoder, unet, args.lora_path, 1.0, args.is_diffuser)
    textencoder, unet, vae = textencoder.to(device), unet.to(device), vae.to(device)
    textencoder.requires_grad_(False)
    textencoder.eval()
    unet.eval()
    vae.eval()
    dm = diffusion_step(tokenizer, textencoder, unet, vae, device)
    return dm