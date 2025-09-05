from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel, Image
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR, SplitDN4IL, DN4ILDataset
from transformers import ViTFeatureExtractor, ViTForImageClassification as ViT_Pretrained
from open_clip_vit import VisionTransformer
from modeling_vit import ViTForImageClassification
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from torchvision.transforms import InterpolationMode, CenterCrop, Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize, RandomResizedCrop, ToPILImage, ToTensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from localdatasets import make_VLCS, make_TI, make_DN4IL
from vil_datasets import build_continual_dataloader
from torch import Tensor
import peft

from transformers import AutoImageProcessor



from progbar import Progbar
from copy import deepcopy
import open_clip
import os
import numpy as np
from PIL import Image
import argparse

from typing import List, Optional, Tuple, Union
from inject import inject_linear_attention
from transformers.models.vit.modeling_vit import ViTEmbeddings, ViTLayer, ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput



import open_clip
# from 


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="OfficeHome")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()
args.label_offset = 0





def replace_layers(model, trainable, freeze_base=None, freeze_linear_base=None, freeze_lnorm=False):
    # print(model)
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, trainable, freeze_base, freeze_linear_base)

        if isinstance(module, nn.MultiheadAttention):
            # print(model)
            # print(module)
            # exit(0)
            # assert False, "MultiheadAttention should not be used in this script, use CustomAttention instead"
            setattr(model, n, CustomAttention(
                embed_dim=768,
                num_heads=12,
                bias=module.in_proj_bias is not None,
                batch_first=True,  
                in_proj_weight=module.in_proj_weight,
                in_proj_bias=module.in_proj_bias,
                out_proj=module.out_proj

            ))


class CustomAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=False,
        in_proj_weight=None,
        in_proj_bias=None,
        out_proj=None,
        **kwargs
    ):
        super().__init__()
        print("REPLACING###########################")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.batch_first = batch_first


        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # assert False
        assert in_proj_weight is not None
        if in_proj_weight is not None:
            assert in_proj_weight.shape == (3 * embed_dim, embed_dim)
            q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
            print("CALEED###################")
            self.q_proj.weight.data = q_w
            self.k_proj.weight.data = k_w
            self.v_proj.weight.data = v_w

        if in_proj_bias is not None:
            assert in_proj_bias.shape == (3 * embed_dim,)
            q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)
            self.q_proj.bias.data = q_b
            self.k_proj.bias.data = k_b
            self.v_proj.bias.data = v_b

        self.out_proj = out_proj if out_proj is not None else nn.Linear(embed_dim, embed_dim, bias=bias)
        
        
        #Add bias in forward also
        #FIrst use replace layers, then use peft model
        
        #USe method of replace layers to convert MHA to Custom attention
        
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query=None,
        key=None,
        value=None,
        # q_x=None,
        # k_x=None,
        # v_x=None,
        q_bias: Optional[Tensor] = None,
        k_bias: Optional[Tensor] = None,
        v_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        **kwargs
    ):
        query = query if query is not None else kwargs.get("q_x", None)
        key = key if key is not None else kwargs.get("k_x", None)
        value = value if value is not None else kwargs.get("v_x", None)

        if query is None or key is None or value is None:
            raise ValueError("CustomAttention expects either (query, key, value) or (q_x, k_x, v_x) to be passed.")
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        if q_bias is not None:
            q = q + q_bias
        if k_bias is not None:
            k = k + k_bias
        if v_bias is not None:
            v = v + v_bias

        scale = self.head_dim ** -0.5
        h = self.num_heads
        d = self.head_dim

        if self.batch_first:
            # [B, S, E]
            B, S, E = q.shape
            q = q.view(B, S, h, d).transpose(1, 2)  
            k = k.view(B, -1, h, d).transpose(1, 2) 
            v = v.view(B, -1, h, d).transpose(1, 2) 

            attn = (q @ k.transpose(-2, -1)) * scale     
            attn = attn.softmax(dim=-1)
            out  = (attn @ v)                              
            out  = out.transpose(1, 2).contiguous().view(B, S, E)
        else:
            S, B, E = q.shape
            q = q.view(S, B, h, d).permute(1, 2, 0, 3)     
            k = k.view(-1, B, h, d).permute(1, 2, 0, 3)   
            v = v.view(-1, B, h, d).permute(1, 2, 0, 3)   

            attn = (q @ k.transpose(-2, -1)) * scale       
            attn = attn.softmax(dim=-1)
            out  = (attn @ v)                              
            out  = out.permute(2, 0, 1, 3).contiguous().view(S, B, E)
        # print(out.shape)

        return self.out_proj(out), attn


@torch.no_grad()
def replace_attention_with_custom(model, lora_cfg=None):
    replace_layers(model, True)
    
@torch.no_grad()
def get_peft_lora_weights(model: nn.Module) -> dict:
    module_weight_dict = {}
    for i, resblock in enumerate(model.transformer.resblocks):
        lora_weights = []
        for proj_name in ["q_proj", "v_proj"]: 
            try:
                proj_layer = getattr(resblock.attn, proj_name)
                lora_A = proj_layer.lora_A['default'].weight.detach().cpu().reshape(-1)
                lora_B = proj_layer.lora_B['default'].weight.detach().cpu().reshape(-1)
                lora_weights.extend([lora_A, lora_B])
            except AttributeError:
                print(f"Warning: Could not find LoRA weights for resblock {i}, projection {proj_name}")
                continue
        if lora_weights:
            module_weight_dict[i] = torch.cat(lora_weights)
    return module_weight_dict

@torch.no_grad()
def get_values_lora(model: nn.Module) -> dict:
    weight_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            for attr in dir(module):
                if attr.endswith('hopfield_values'):
                    # print(f"Hopfield values for {name}: {getattr(module, attr)}")
                    #filtering name
                    new_name = name[name.find('resblocks.'):]
                    weight_dict[new_name] = getattr(module, attr).detach().clone()
                    
    return weight_dict

@torch.no_grad()
def get_keys_lora(model: nn.Module) -> dict:
    weight_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            for attr in dir(module):
                if attr.endswith('hopfield_keys'):
                    # print("Here", getattr(module, attr))
                    # print(f"Hopfield keys for {name}: {getattr(module, attr)}")
                    #filtering name
                    new_name = name[name.find('resblocks.'):]
                    weight_dict[new_name] = getattr(module, attr).detach().clone()
                    
    return weight_dict
                    # keys_param.copy_(weight_dict)
        # if i in weight_dict:
        #     current_pos = 0
        #     for proj_name in ["q_proj", "v_proj"]: 
        #         try:
        #             proj_layer = getattr(resblock.attn, proj_name)
                    
        #             if hasattr(proj_layer, "hopfield_keys"):
        #                 keys_param = proj_layer.hopfield_keys
        #                 len_K = keys_param.numel()
        #                 if current_pos + len_K <= weight_dict[i].numel():
        #                     with torch.no_grad():
        #                         keys_param.copy_(
        #                             weight_dict[i][current_pos:current_pos+len_K]
        #                             .reshape_as(keys_param).to(keys_param.device, dtype=keys_param.dtype)
        #                         )
        #                     current_pos += len_K
        #                 else:
        #                     print(f"Warning: no packed Hopfield keys for block {i} {proj_name}; skipping.")
        #         except AttributeError:
        #             print(f"Warning: Could not find LoRA layers in resblock {i}, projection {proj_name} to load weights.")
        #             continue

@torch.no_grad()
def hopfield_init(model):
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            # for attr in dir(module):
            #     print(f"{name}.{attr}")
            module.commit_to_hopfield()
            new_keys_shape = module.hopfield_keys.shape
            new_values_shape = module.hopfield_values.shape
            print(f"After committing, Hopfield keys for {name}: {new_keys_shape}")
            print(f"After committing, Hopfield values for {name}: {new_values_shape}")

@torch.no_grad()
def set_peft_lora_weights(model: nn.Module, weight_dict: dict):
    weight_dict = deepcopy(weight_dict)
    # for i in model.parameters():
        # print(i)
    # for i in model


            # print("Committing to hopfield for", name)
            # print(name)
    # for i in model.modules():
    #     # print(type(i))
    #     if type(i) == peft.tuners.lora.layer.Linear:
    #         i.commit_to_hopfield()
    #         print("Committing to hopfield")
    
    # for name, module in model.named_modules():
    #     if isinstance(module, peft.tuners.lora.layer.Linear):
    #         for attr in dir(module):
    #             if attr.endswith('hopfield_values'):
    #                 print(f"Hopfield keys for {name}: {getattr(module, attr)}")
    
    # exit(0)
    
    # exit(0)
                # print(f"{name}.{attr}" )
    
    # exit(0)
    
    for i, resblock in enumerate(model.transformer.resblocks):
        if i in weight_dict:
            current_pos = 0
            for proj_name in ["q_proj", "v_proj"]: 
                try:
                    proj_layer = getattr(resblock.attn, proj_name)
                    lora_A_layer = proj_layer.lora_A['default']
                    lora_B_layer = proj_layer.lora_B['default']
                    
                    # hopfield_keys = proj_layer.keys
                    
                    device = lora_A_layer.weight.device
                    
                    len_A = lora_A_layer.weight.numel()
                    lora_A_layer.weight.data = weight_dict[i][current_pos:current_pos+len_A].reshape(lora_A_layer.weight.shape).to(device)
                    current_pos += len_A
                    
                    len_B = lora_B_layer.weight.numel()
                    lora_B_layer.weight.data = weight_dict[i][current_pos:current_pos+len_B].reshape(lora_B_layer.weight.shape).to(device)
                    current_pos += len_B
                    # if hasattr(proj_layer, "keys") and isinstance(proj_layer.keys, torch.nn.Parameter):
                    #     keys_param = proj_layer.keys
                    #     len_K = keys_param.numel()
                    #     if current_pos + len_K <= weight_dict[i].numel():
                    #         with torch.no_grad():
                    #             keys_param.copy_(
                    #                 weight_dict[i][current_pos:current_pos+len_K]
                    #                 .reshape_as(keys_param).to(keys_param.device, dtype=keys_param.dtype)
                    #             )
                    #         current_pos += len_K
                    #     else:
                    #         print(f"Warning: no packed Hopfield keys for block {i} {proj_name}; skipping.")
                except AttributeError:
                    print(f"Warning: Could not find LoRA layers in resblock {i}, projection {proj_name} to load weights.")
                    continue


@torch.no_grad()
def get_peft_lora_weights_hf_vit(model: nn.Module) -> dict:
    out = {}

    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        attn = model.vit.encoder.layer[i].attention.attention
        for name in ["query", "value"]:
            lin = getattr(attn, name)

            if not hasattr(lin, "lora_A") or "default" not in lin.lora_A:
                continue
            A = lin.lora_A["default"].weight.detach().cpu().reshape(-1)
            B = lin.lora_B["default"].weight.detach().cpu().reshape(-1)
            out.setdefault(i, [])
            out[i].extend([A, B])
        if i in out:
            out[i] = torch.cat(out[i])
    return out

@torch.no_grad()
def set_peft_lora_weights_hf_vit(model: nn.Module, weight_dict: dict):
    weight_dict = deepcopy(weight_dict)
    num_layers = model.config.num_hidden_layers
    for i in range(num_layers):
        if i not in weight_dict:
            continue
        cur = weight_dict[i]
        pos = 0
        attn = model.vit.encoder.layer[i].attention.attention
        for name in ["query", "value"]:
            lin = getattr(attn, name)
            if not hasattr(lin, "lora_A") or "default" not in lin.lora_A:
                continue
            A_mod = lin.lora_A["default"]
            B_mod = lin.lora_B["default"]
            nA = A_mod.weight.numel()
            nB = B_mod.weight.numel()
            A_mod.weight.data.copy_(cur[pos:pos+nA].view_as(A_mod.weight)); pos += nA
            B_mod.weight.data.copy_(cur[pos:pos+nB].view_as(B_mod.weight)); pos += nB


if args.base_model == 'laion':




    laion_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")
    vit = laion_model.visual.cuda()
    # _ = vit(torch.randn(1, 3, 224, 224).cuda())
    # print(_.shape)
    # print(vit.transformer.resblocks[0].attn.in_proj_weight)
    # print(vit)
    # exit(0)
    

    replace_attention_with_custom(vit)
    print(vit)
    # vit = set_lorank_trainable(vit, True, freeze_base=True, freeze_linear_base=True, freeze_lnorm=True)
    # vit = convert_to_custom_attention(vit)
    # print(vit)
    # vit = inject_linear_attention(
    #     model = vit,
    #     encoders = {"transformer"},
    #     embed_dim=768,
    #     num_heads=12
    # )
    # print(vit)
    

    # print("Model structure before conversion:")
    # print(vit)
    # print("Converting model to use CustomAttention layers...")
    # vit = convert_to_custom_attention(vit)




elif args.base_model == 'vit-in21k':
    vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=args.num_classes).cuda()
    model_string = 'google/vit-base-patch16-224-in21k'
    preprocess_train = ViTFeatureExtractor.from_pretrained(model_string)
    preprocess_val = preprocess_train
    
    # print(vit)
    
    replace_attention_with_custom(vit)
    
    print("VIT", vit)

    @torch.no_grad()
    def get_store_dict(model):
        module_weight_dict = {}
        for i in range(12):
            module = model.vit.encoder.layer[i]
            lorank_values = []
            for n, p in module.named_parameters():
                if "lora" in n:
                    lorank_values.append(p.reshape(-1).detach())
            lorank_values = torch.cat(lorank_values, dim=0)
            module_weight_dict[i] = lorank_values
        return module_weight_dict

    @torch.no_grad()
    def set_store_dict(model, weight_dict):
        weight_dict = deepcopy(weight_dict)
        for i in range(12):
            module = model.vit.encoder.layer[i]
            for n, p in module.named_parameters():
                if "lora" in n:
                    p.data = weight_dict[i][:p.numel()].reshape(p.shape)
                    weight_dict[i] = weight_dict[i][p.numel():]

    def load_vit_weights(vit, pretrained_vit):
        vit_state_dict = vit.state_dict()
        pretrained_vit_state_dict = pretrained_vit.state_dict()
        for n, p in vit_state_dict.items():
            if n in pretrained_vit_state_dict and "classifier" not in n:
                vit_state_dict[n] = pretrained_vit_state_dict[n]

        vit.load_state_dict(vit_state_dict)


# Load PACS dataset from Hugging Face
is_hf_dataset = True
dataset_name = args.dataset
if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
elif dataset_name == "DomainNet":
    dataset = load_dataset("wltjr1007/DomainNet")
    train_domains = [0, 1, 2, 3, 4, 5]
elif dataset_name == "OfficeHome":
    dataset = load_dataset("flwrlabs/office-home")
    train_domains = ['Art', 'Clipart', 'Product', 'Real World']
elif dataset_name == "VLCS":
    train_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    try:
        dataset = load_dataset("ai22mtech12002/DG_VLCS")
    except:
        dataset = make_VLCS('data/VLCS')
elif dataset_name == "TI":
    try:
        dataset = load_dataset("ai22mtech12002/DG_TI")
    except:
        dataset = make_TI('data/terra_incognita')
    train_domains = ['location_38', 'location_43', 'location_46', 'location_100']

elif dataset_name == "DN4IL":
    train_domains = ["real", "clipart", "infograph", "painting", "quickdraw", "sketch"] 
    dataset = load_dataset("ai22mtech12002/DN4IL")
    # try:
    #     dataset = load_dataset("ai22mtech12002/DN4IL")
    # except:
    #     dataset = make_DN4IL('data/DN4IL')
elif dataset_name == "CDDB":
    train_domains = ['gaugan', 'biggan' , 'wild', 'whichfaceisreal', 'san']
    dataset = load_dataset("ai22mtech12002/CDDB-hard")

elif dataset_name == "iDigits-cil":
    args.num_tasks = 5
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/iDigits'
    args.task_type = 'cil'
    args.shuffle = True
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    preprocess_val = Compose([
            Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
elif dataset_name == "iDigits-dil":
    args.num_tasks = 4
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/iDigits'
    args.task_type = 'dil'
    args.shuffle = True
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    preprocess_val = Compose([
            Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
elif dataset_name == "CORe50-cil":
    args.num_tasks = 5
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/Core50'
    args.task_type = 'cil'
    args.shuffle = False
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    preprocess_val = Compose([
            Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
elif dataset_name == "CORe50-dil":
    args.num_tasks = 8
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/Core50'
    args.task_type = 'dil'
    args.shuffle = False
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    preprocess_val = Compose([
            Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
elif dataset_name == "DomainNet-cil":
    args.num_tasks = 5
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-raw'
    args.task_type = 'cil'
    args.shuffle = False
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    preprocess_val = Compose([
            Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
elif dataset_name == "DomainNet-dil":
    args.num_tasks = 6
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-dil'
    args.task_type = 'dil'
    args.shuffle = False
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    if args.base_model == 'vit-in21k':
        ip = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        img_mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        img_std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        eval_resize = 256
        eval_crop = 224
        preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224),
                            scale=(0.9, 1.0),
                            ratio=(0.75, 1.3333),
                            interpolation=InterpolationMode.BICUBIC,
                            antialias=True),
            ToTensor(),                          
            Normalize(mean=img_mean, std=img_std)
        ])
        preprocess_val = Compose([
            Resize(size=(eval_resize, eval_resize),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True),
            CenterCrop((eval_crop, eval_crop)),
            ToTensor(),
            Normalize(mean=img_mean, std=img_std)
        ])
    else:
        preprocess_train = Compose([
                RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])

    # preprocess_train = Compose([
    #         RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
    #         ToTensor(),
    #         Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    #     ])
    # preprocess_val = Compose([
    #         Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
    #         CenterCrop((224, 224)),
    #         ToTensor(),
    #         Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    #     ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False

# CIL Datasets
elif args.dataset == "cifar100":
    fixed_order = list(range(100))
    if args.seed is not None:
        np.random.seed(args.seed)
        fixed_order = np.random.permutation(fixed_order)
    num_classes=args.num_classes
    num_tasks = 100 // num_classes
    train_domains = list(range(num_tasks))
    assert 100 % num_classes == 0, "num_classes should be divisible by 100"
    benchmark = SplitCIFAR100(num_tasks, return_task_id=False, class_ids_from_zero_in_each_exp=True, fixed_class_order=fixed_order)
    is_hf_dataset = False

elif args.dataset == "tinyimagenet":
    fixed_order = list(range(200))
    if args.seed is not None:
        np.random.seed(args.seed)
        fixed_order = np.random.permutation(fixed_order)
    num_classes=args.num_classes
    num_tasks = 200 // num_classes
    train_domains = list(range(num_tasks))
    assert 200 % num_classes == 0, "num_classes should be divisible by 200"
    benchmark = SplitTinyImageNet(num_tasks, return_task_id=True, class_ids_from_zero_in_each_exp=True, fixed_class_order=fixed_order)
    is_hf_dataset = False

elif args.dataset == "cub":
    fixed_order = list(range(200))
    if args.seed is not None:
        np.random.seed(args.seed)
        fixed_order = np.random.permutation(fixed_order)
    num_classes=args.num_classes
    num_tasks = 200 // num_classes
    train_domains = list(range(num_tasks))
    assert 200 % num_classes == 0, "num_classes should be divisible by 200"
    benchmark = SplitCUB200(num_tasks, return_task_id=True, fixed_class_order=fixed_order)
    is_hf_dataset = False

elif args.dataset == "inetR":
    fixed_order = list(range(200))
    if args.seed is not None:
        np.random.seed(args.seed)
        fixed_order = np.random.permutation(fixed_order)
    num_classes=args.num_classes
    num_tasks = 200 // num_classes
    train_domains = list(range(num_tasks))
    assert 200 % num_classes == 0, "num_classes should be divisible by 200"
    benchmark = SplitImageNetR(dataset_root='/data/ai22mtech12002/projects/GlobalMemCL', n_experiences=num_tasks, return_task_id=True, fixed_class_order=fixed_order)
    is_hf_dataset = False
    
@torch.no_grad()
def load_hopfield_keys(model, hopfield_keys_dict):
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            for attr in dir(module):
                if attr.endswith('hopfield_keys'):
                    new_name = name[name.find('resblocks.'):]
                    if new_name in hopfield_keys_dict:
                        # print(f"Loading hopfield keys for {new_name}")
                        keys_param = getattr(module, attr)
                        print(keys_param.shape)
                        keys_param.copy_(hopfield_keys_dict[new_name].to(keys_param.device, dtype=keys_param.dtype))
@torch.no_grad()
def load_hopfield_values(model, hopfield_values_dict):
    # hopfield_values_dict = get_values_lora(model)
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.layer.Linear):
            for attr in dir(module):
                if attr.endswith('hopfield_values'):
                    new_name = name[name.find('resblocks.'):]
                    if new_name in hopfield_values_dict:
                        # print(f"Loading hopfield values for {new_name}")
                        values_param = getattr(module, attr)
                        values_param.copy_(hopfield_values_dict[new_name].to(values_param.device, dtype=values_param.dtype))



    
adapters_per_domain = args.adapters_per_domain
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes

train_loaders = {}
parent_dir = f'data/{dataset_name}'
os.makedirs(parent_dir, exist_ok=True)

class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, returns_domain=True):
        self.dataset = dataset
        self.preprocess = preprocess
        self.returns_domain = returns_domain
        # if not is_hf_dataset:
        #     if args.base_model == 'vit-in21k':
        #         self.vit_preprocess = Compose([
        #                 Resize(256),
        #                 RandomCrop(224),
        #                 RandomHorizontalFlip(0.5),
        #                 Normalize(preprocess.image_mean, preprocess.image_std)
        #             ])
        #     else:
        #         self.preprocess = Compose([
        #                 Resize(256),
        #                 RandomCrop(224),
        #                 RandomHorizontalFlip(0.5),
        #                 Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        #             ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(list(item['image'].keys()))
        # item['image'].verify()
        if is_hf_dataset:
            if args.base_model == 'vit-in21k':
                item['image'] = self.vit_preprocess(item['image'])
            else:
                item['image'] = self.preprocess(item['image'])
            return {'image': item['image'], 'label': item['label']}
        else:
            if self.returns_domain:
                image, label, _ = item
            else:
                image, label = item
            item = {}
            item['image'] = self.preprocess(image)
            item['label'] = label - args.label_offset
            return item


if args.dataset in ['iDigits-cil', 'iDigits-dil', 'CORe50-cil', 'DomainNet-cil', 'DomainNet-dil']:
    for i, dl in enumerate(dataloaders):
        dataset = dl['train']
        train_dataset = DomainDataset(dataset, preprocess_train, returns_domain=False)
        train_loaders[i] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        print(f"Training samples for domain {i}: {len(dataset)}")
elif is_hf_dataset:
    for domain in train_domains:
        train_dataset = dataset.filter(lambda x: x['domain'] == domain)
        try:
            train_dataset = train_dataset['train']
        except:
            pass
        print(f"Training samples for domain {domain}: {len(train_dataset)}")
        train_dataset = DomainDataset(train_dataset, preprocess_train)
        train_loaders[domain] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
else:
    for domain, (exp_train, exp_test) in enumerate(zip(benchmark.train_stream, benchmark.test_stream)):
        train_dataset = exp_train.dataset
        print(f"Training samples for domain {domain}: {len(train_dataset)}")
        train_dataset = DomainDataset(train_dataset, preprocess_train)
        train_loaders[domain] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)


train_domain_adaters_list = []
classifiers = {}
classifier = None



prev_keys = []
prev_values = []
for col in range(adapters_per_domain):
    

    
    #After every adapter, load hopfield keys and hopfield values in the model and then train
    #do it for if col > 0->load_hopfield_keys/values_function
    #hopfield values should match with adapters saved after flattening
    


    train_domain_adaters = {}
    domain_models = {}
    # train_domain_adaters_list.append(train_domain_adaters)

    # generate a dictionary mapping 0-6 -> 0-6 in a random permutation
    perm = torch.randperm(num_classes)
    perm_dict = {i: perm[i].item() for i in range(num_classes)}
    
    # if col > 0:
    #     keys_to_commit = get_keys_lora(model)
    #     values_to_commit = get_values_lora(model)

    if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
        args.label_offset = 0

    for did, domain_name in enumerate(train_domains):
        print(f"Domain {domain_name} ({did})")
        if args.base_model == 'laion':
            
            model = deepcopy(vit)
            lora_cfg = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], 
            lora_dropout=0.0,
            bias="none",
            # task_type="FEATURE_EXTRACTION"
            )
            model = get_peft_model(model, lora_cfg)
            classifier = nn.Linear(512, num_classes, bias=False).cuda()
            
            

            print("\n" + "="*50)
            print("🔎 MODEL STRUCTURE")
            print("="*50)
            # print(model) 

            print("\n" + "="*50)
            print("TRAINABLE PARAMETERS")
            print("="*50)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)

            print("="*50 + "\n")
            print(f"--- Training Domain: {domain_name} ---")
            # model.print_trainable_parameters()
            
            
            trainable_params = [classifier.weight]
            # for name, param in model.named_parameters():
            #     # num_total += param.numel()
            #     if "lora" in name or "classifier" in name:
            #         param.requires_grad = True
            #         # num_trainable += param.numel()
            #         trainable_params.append(param)
            #     if "classifier" in name:
            #         print(name + ":", "Training classifier" if param.requires_grad else "Freezing classifier")
            trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
            # trainable_params = []
            trainable_params.extend(list(classifier.parameters()))


            # print("Printing model structure :", model)
        elif args.base_model == 'vit-in21k':
            model = deepcopy(vit)
            lora_cfg = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
            lora_dropout=0.0,
            bias="none",
            # task_type="FEATURE_EXTRACTION"
        )
            model = get_peft_model(model, lora_cfg)
            # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
            # classifier = model.classifier
            classifier = nn.Linear(768, num_classes, bias=False).cuda()
            model.classifier = classifier
            # load_vit_weights(model, vit)
            
        
        train_loader = train_loaders[domain_name]

        num_trainable = 0
        num_total = 0
        # trainable_params = [classifier.weight]
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # trainable_params.append(classifier.weight)
        if args.base_model == 'laion':
            trainable_params.extend(list(classifier.parameters()))


        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
            
        # print("Trainable parameters: ", trainable_params)
        for name, param in model.named_parameters():
            num_total += param.numel()
            if "lora" in name or "classifier" in name:
                param.requires_grad = True
                num_trainable += param.numel()
                trainable_params.append(param)
            if "classifier" in name:
                print(name + ":", "Training classifier" if param.requires_grad else "Freezing classifier")        

        # print(f"Trainable parameters: {num_trainable} / {num_total}")

        if len(prev_keys) > 0 and len(prev_values) > 0:
            print(len(prev_keys), len(prev_values))
            i1 = 0
            i2 = 0
            with torch.no_grad():
                for name, module in model.named_modules():

                    if isinstance(module, peft.tuners.lora.layer.Linear):
                        for attr in dir(module):
                            if attr.endswith('hopfield_keys'):
                                print(f"Loading hopfield keys for {name} using {i1}th key")
                                # keys_param = getattr(module, attr)
                                setattr(module, attr, nn.Parameter(prev_keys[i1]))
                                
                                
                                # keys_param = nn.Parameter(prev_keys[i1])
                                # keys_param.copy_(prev_keys[i1].to(keys_param.device, dtype=keys_param.dtype))
                                print(f"After loading hopfield keys for {name}, keys_param shape: {getattr(module, attr).shape}")
                                i1 += 1
                                # print(f"Previous keys shape: {prev_keys[i].shape}, Current keys shape: {keys_param.shape}")
                            if attr.endswith('hopfield_values'):
                                # values_param = getattr(module, attr)
                                setattr(module, attr, prev_values[i2])
                                # print(f"Previous values shape: {prev_values[i].shape}, Current values shape: {values_param.shape}")
                                # if values_param.shape == prev_values[i].shape:
                                # values_param = prev_values[i2]
                                # values_param.copy_(prev_values[i2].to(values_param.device, dtype=values_param.dtype))
                                print(f"After loading hopfield values for {name}, values_param shape: {getattr(module, attr).shape}")

                                i2 += 1
        # if len(prev_keys) > 0 and len(prev_values) > 0:
        #     for name, module in model.named_modules():
        #         i1 = 0
        #         i2 = 0
        #         if isinstance(module, peft.tuners.lora.layer.Linear):
        #             for attr in dir(module):
        #                 if attr.endswith('hopfield_keys'):
        #                     keys_param = getattr(module, attr)
        #                     print(keys_param.shape)
        prev_keys = []
        prev_values = []
        

                        # keys_param.copy_(prev_keys[i1].to(keys_param.device, dtype=keys_param.dtype))
                        # i1 += 1
                        # print(f"Previous keys shape: {prev_keys[i].shape}, Current keys shape: {keys_param.shape}")
                    # if attr.endswith('hopfield_values'):
                    #     values_param = getattr(module, attr)
                    #     # print(f"Previous values shape: {prev_values[i].shape}, Current values shape: {values_param.shape}")
                    #     # if values_param.shape == prev_values[i].shape:
                    #     values_param.copy_(prev_values[i2].to(values_param.device, dtype=values_param.dtype))
                    #     i2 += 1
                    
        
        # Train the model
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-12)
        criterion = nn.CrossEntropyLoss()
        ###############################
        # with torch.no_grad():
        #     dummy_input = torch.randn(10, 1, 3, 224, 224).cuda()
        #     for i in range(10):
        #         start = time.time()
        #         _ = model(dummy_input[i])
        #         end = time.time()
        #     print(f"Time taken for forward pass {i+1}: {end - start:.4f} seconds")
        
        #Take average of middle 5        
        ##############################
        # if col > 0:
        #     print("Called here")
        #     load_hopfield_keys(model, keys_to_commit)
        #     load_hopfield_values(model, values_to_commit)
        model.train()
        model_v = torch.vmap(model)
        classifier_v = torch.vmap(classifier)

        for epoch in range(epochs):
            pbar = Progbar(len(train_loader))
            acc_list = []
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pixel_values = batch['image'].cuda()
                labels = batch['label'].cuda()
                if step == 0:
                    print("Labels: ", labels.min(), labels.max())
                if args.base_model == 'laion':
                    pixel_values = pixel_values[:,None,...]
                    outputs = classifier_v(model_v(pixel_values))
                    outputs = outputs.squeeze(1)
                elif args.base_model == 'vit-in21k':
                    out = model(pixel_values)
                    # print("Model Output", out)
                    outputs = out.logits
                    # print("Outputs: ", outputs)
                # print("OUTPUTS: ", outputs.shape)
                # print("OUTPUTS ################################################################")
                # print(model(pixel_values)[:, 128:256])
                loss = criterion(outputs, labels)
                # old_weights = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
                loss.backward()
                
                # #After every step print the difference between the newer and older weights
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         print(f"Weight change in {name}: {param.data - param.data.clone().detach()}")

                # if step == 0 or step == 10: 
                #     print("\n" + "="*50)
                #     print("🔎 GRADIENT CHECK (FIRST BATCH)")
                #     print("="*50)
                #     total_grad_norm = 0.0
                #     for name, param in model.named_parameters():
                #         if param.requires_grad and param.grad is not None:
                           
                #             grad_norm = param.grad.abs().sum()
                #             print(f"{name:<60} | Grad Norm: {grad_norm.item():.4f}")
                #             total_grad_norm += grad_norm
                #         elif param.requires_grad:
                            
                #             print(f"WARNING: {name:<60} | Grad is None!")
                    
                #     print("-" * 50)
                #     print(f"Total Gradient Norm on All Trainable Params: {total_grad_norm.item():.4f}")
                #     print("="*50 + "\n")
                optimizer.step()
                # break
                
                #Printing change
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         delta = param.detach() - old_weights[name]
                        # print(f"Change in {name}: mean={delta.abs().mean().item():.6f}, max={delta.abs().max().item():.6f}")  
                
                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                acc_list.append(acc)
                pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
            if sum(acc_list)/len(acc_list) > 0.99:
                break

        # train_domain_adaters[domain_name] = get_peft_lora_weights(model)
        classifiers[domain_name] = deepcopy(classifier)
        print("Verifying saved weights...")
        print(len(train_domain_adaters_list))
        if(len(train_domain_adaters_list) >= domain_name + 1):
            if args.base_model == 'laion':
                train_domain_adaters_list[domain_name] = get_peft_lora_weights(model)
            elif args.base_model == 'vit-in21k':
                train_domain_adaters_list[domain_name] = get_peft_lora_weights_hf_vit(model)
        else:
            if args.base_model == 'laion':
                train_domain_adaters_list.append(get_peft_lora_weights(model))
            elif args.base_model == 'vit-in21k':
                train_domain_adaters_list.append(get_peft_lora_weights_hf_vit(model))
        print(len(train_domain_adaters_list))
        
        
        # keys_before = get_keys_lora(model)

        test_model = model
        if args.base_model == 'laion':
            set_peft_lora_weights(test_model, train_domain_adaters_list[domain_name])
        elif args.base_model == 'vit-in21k':
            set_peft_lora_weights_hf_vit(test_model, train_domain_adaters_list[domain_name])
    
        if col> 0:
            for name, module in test_model.named_modules():
                if isinstance(module, peft.tuners.lora.layer.Linear):
                    for attr in dir(module):
                        if attr.endswith('hopfield_keys'):
                        
                            keys_param = getattr(module, attr)
                            print(f"Keys shape before init hopfield {col}: {keys_param.shape}")
                        if attr.endswith('hopfield_values'):
                            values_param = getattr(module, attr)
                            print(f"Values shape before init hopfield {col}: {values_param.shape}")

        hopfield_init(model)
        
        for name, module in model.named_modules():
            if isinstance(module, peft.tuners.lora.layer.Linear):
                for attr in dir(module):
                    if attr.endswith('hopfield_keys'):
                        keys_param = getattr(module, attr)
                        print(f"Keys shape {col}: {keys_param.shape}")
                        prev_keys.append(keys_param.clone().detach())
                    if attr.endswith('hopfield_values'):
                        values_param = getattr(module, attr)
                        prev_values.append(values_param.clone().detach())
        
        print(f"After loading in list, prev_keys length: {len(prev_keys)}, prev_values length: {len(prev_values)}")
            
        # keys_after = get_keys_lora(model)
        # test_model = get_peft_model(deepcopy(model), lora_cfg)
        # set_peft_lora_weights(test_model, train_domain_adaters[domain_name])
        test_model.eval()
        pbar = Progbar(len(train_loader))
        # dummy_keys = get_keys_lora(test_model)
        # for k, v in dummy_keys.items():
        #     print(f"Keys for block {k}: {v.shape}")
        #     exit(0)
        # torch.save(dummy_keys, f"weights/hopfield_keys_dummy_{args.base_model}_{dataset_name}{args.name_tag}.pt")
        # print(f"saved dummy keys to weights/hopfield_keys_dummy_{args.base_model}_{dataset_name}{args.name_tag}.pt")
        # exit(0)
        # print("Dummy keys", dummy_keys.keys())
        
        # lora_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     lora_dropout=0.1,
        #     target_modules=["q_proj", "v_proj"],
        #     bias="none",
        # )
        # new_model = get_peft_model(new_model, lora_config)
        # # train_domain_adaters[domain_name] = get_store_dict(model)
        # train_domain_adaters[domain_name] = get_peft_lora_weights(model)
        # # set_store_dict(new_model, train_domain_adaters[domain_name])
        # set_peft_lora_weights(new_model, train_domain_adaters[domain_name])
        # domain_models[domain_name] = deepcopy(model)
        # new_model.eval()

        # for n, m in model.state_dict().items():
        #     if not new_model.state_dict()[n].equal(m):
        #         print()
        #         if "classifier" in n:
        #             print("Classifier not equal")
        #         else:
        #             print(n)
        #         print()

        pbar = Progbar(len(train_loader))
        for step, batch in enumerate(train_loader):
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda()
            if args.base_model == 'laion':
                outputs = classifier(test_model(pixel_values))
            elif args.base_model == 'vit-in21k':
                out = test_model(pixel_values)
                outputs = out.logits
            # outputs = classifier(test_model(pixel_values))
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("verification acc", acc)])

        classifiers[domain_name] = deepcopy(classifier)

        if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
            args.label_offset += args.num_classes


# weight_dict_keys = {}
weight_dict_keys = get_keys_lora(test_model)

# weight_dict_values = {}
weight_dict_values = get_values_lora(test_model)
print("Saving weights to ", f"weights/train_domain_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")

torch.save(weight_dict_keys, f"weights/hopfield_keys_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(weight_dict_values, f"weights/hopfield_values_{args.base_model}_{dataset_name}{args.name_tag}.pt")

torch.save(train_domain_adaters_list, f"weights/train_domain_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(classifiers, f"weights/train_domain_classifiers_{args.base_model}_{dataset_name}{args.name_tag}.pt")