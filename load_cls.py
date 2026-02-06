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


# scratch_path = os.environ['SCRATCH']
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
                    if args.base_model == 'laion':
                        new_name = name[name.find('resblocks.'):]
                    elif args.base_model == 'vit-in21k':
                        new_name = name[name.find('encoder.layer.'):]
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
                    if args.base_model == 'laion':
                        new_name = name[name.find('resblocks.'):]
                    elif args.base_model == 'vit-in21k':
                        new_name = name[name.find('encoder.layer.'):]
                    weight_dict[new_name] = getattr(module, attr).detach().clone()
                    # print("New name keyword : ", new_name)
    # print(weight_dict)
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
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-dil/'
    
    # args.data_path = os.path.join(os.environ['WORK'], 'data/DomainNet-dil')
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



#Creating test loaders
test_domain_loaders = []
for domain_idx, domain in enumerate(train_domains):
    if args.dataset in ['iDigits-dil', 'CORe50-dil', 'DomainNet-dil']:
        dl = dataloaders[domain]
        train_dataset = dl['train']
        train_dataset = DomainDataset(train_dataset, preprocess_train, returns_domain=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_dataset = dl['test']
        test_dataset = DomainDataset(test_dataset, preprocess_val, returns_domain=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_domain_loaders.append(test_loader)
        print(f"Training samples for domain {domain}: {len(train_dataset)}")
        print(f"Testing samples for domain {domain}: {len(test_dataset)}")
    elif is_hf_dataset:
        train_dataset = dataset.filter(lambda x: x['domain'] == domain)
        try:
            train_dataset = train_dataset['train']
        except KeyError:
            pass  
        
        train_dataset = DomainDataset(train_dataset, preprocess_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        test_dataset = dataset.filter(lambda x: x['domain'] == domain)
        try:
            test_dataset = test_dataset['test']
        except KeyError:
            try:
                test_dataset = test_dataset['train']
            except:
                pass
        test_dataset = DomainDataset(test_dataset, preprocess_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_domain_loaders.append(test_loader)
    else:
        train_dataset = list(benchmark.train_stream)[domain_idx].dataset
        print(f"Training samples for domain {domain}: {len(train_dataset)}")
        train_dataset = DomainDataset(train_dataset, preprocess_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        test_dataset = list(benchmark.test_stream)[domain_idx].dataset
        print(f"Testing samples for domain {domain}: {len(test_dataset)}")
        test_dataset = DomainDataset(test_dataset, preprocess_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        test_domain_loaders.append(test_loader)



hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_vit-in21k_DomainNet-dilcls_tokens.pt')
hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_vit-in21k_DomainNet-dilcls_tokens.pt')
model = deepcopy(vit)

if args.base_model == 'laion':
    replace_attention_with_custom(model)
    lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.0,
    bias="none",
    # task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(model, lora_cfg)
    
    classifier = nn.Linear(512, args.num_classes, bias=False).cuda()
    # classifier.load_state_dict(last_head.state_dict())
    print(classifier)
elif args.base_model == 'vit-in21k':
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    classifier = nn.Linear(768, args.num_classes, bias=False).cuda()
    # classifier.load_state_dict(last_head.state_dict())
    model.classifier = model.model.classifier = nn.Identity()


domain_adapters = {}

# print(hopfield_keys.shape, hopfield_values.shape)
if args.base_model == 'laion':
    for domain_idx, domain in enumerate(train_domains):
        domain_adapters[domain_idx] = {
            "keys_q" : [
                hopfield_keys[f'resblocks.{i}.attn.q_proj'][:, domain_idx :: args.num_tasks]
                for i in range(12)
            ],
            "values_q" : [
                hopfield_values[f'resblocks.{i}.attn.q_proj'][domain_idx :: args.num_tasks, :]
                for i in range(12)
            ],
            "keys_v" : [
                hopfield_keys[f'resblocks.{i}.attn.v_proj'][:, domain_idx :: args.num_tasks]
                for i in range(12)
            ],
            "values_v" : [
                hopfield_values[f'resblocks.{i}.attn.v_proj'][domain_idx :: args.num_tasks, :]
                for i in range(12)
            ]
        }
elif args.base_model == 'vit-in21k':
    for domain_idx, domain in enumerate(train_domains):
        domain_adapters[domain_idx] = {
            "keys_q" : [
                hopfield_keys[f'encoder.layer.{i}.attention.attention.query'][:, domain_idx :: args.num_tasks]
                for i in range(12)
            ],
            "values_q" : [
                hopfield_values[f'encoder.layer.{i}.attention.attention.query'][domain_idx :: args.num_tasks, :]
                for i in range(12)
            ],
            "keys_v" : [
                hopfield_keys[f'encoder.layer.{i}.attention.attention.value'][:, domain_idx :: args.num_tasks]
                for i in range(12)
            ],
            "values_v" : [
                hopfield_values[f'encoder.layer.{i}.attention.attention.value'][domain_idx :: args.num_tasks, :]
                for i in range(12)
            ]
        }
for domain_idx, domain in enumerate(train_domains):
    keys_q = domain_adapters[domain_idx]["keys_q"]
    keys_v = domain_adapters[domain_idx]["keys_v"]
    
    values_q = domain_adapters[domain_idx]["values_q"]
    values_v = domain_adapters[domain_idx]["values_v"]
    prev_keys = None
    keys_new = []
    module_index = 0
    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, peft.tuners.lora.layer.Linear):
            # print("Found")
            module.use_hopfield = True
            
            if args.base_model == 'laion':
                prefix = f'resblocks.{module_index}.attn.' + ('q_proj')
                key_list = keys_q
                value_list = values_q
            elif args.base_model == 'vit-in21k':
                prefix = f'encoder.layer.{module_index}.attention.attention.' + 'query'
                key_list = keys_q
                value_list = values_q
            

            # exit()
            if name.endswith(prefix):
                # print(name, module_index, prefix)
                    
            # FInding module + prefix attr
            # for attr in dir(module):
            #     if 'prefix' in attr:
            #         print(attr)
            # print(module.prefix.hopfield_keys if hasattr(module.prefix, 'hopfield_keys') else "No prefix", prefix)
                # if module.hopfield_keys is None:
                #     module.hopfield_keys = nn.Parameter(torch.ones_like(key_list[module_index]).cuda().clone().detach())
                #     module.hopfield_values = value_list[module_index]
                # else:
                #     module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, torch.ones_like(key_list[module_index]).cuda()], dim=1).cuda().clone().detach())
                #     module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                # keys_new.append(module.hopfield_keys)
                # print(module)
                hopfield_keys = nn.Parameter(torch.ones_like(key_list[module_index]).cuda())
                if module.hopfield_keys is None:
                    module.hopfield_keys = torch.cat([hopfield_keys], dim=1)
                    module.hopfield_values = value_list[module_index]
                else:
                    module.hopfield_keys = torch.cat([module.hopfield_keys, hopfield_keys], dim=1)
                    module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                
                module_index += 1
                keys_new.append(hopfield_keys)
        # exit()
    module_index = 0
    
    for name, module in model.named_modules():
        # print(name)
        if isinstance(module, peft.tuners.lora.layer.Linear):
            # print("Found")
            module.use_hopfield = True
            
            if args.base_model == 'laion':
                prefix = f'resblocks.{module_index}.attn.' + ('v_proj')
                key_list = keys_v
                value_list = values_v
            elif args.base_model == 'vit-in21k':
                prefix = f'encoder.layer.{module_index}.attention.attention.' + 'value'
                key_list = keys_v
                value_list = values_v
            

            # exit()
            if name.endswith(prefix):
                if module.hopfield_keys is None:
                    module.hopfield_keys = nn.Parameter(key_list[module_index].cuda().clone().detach())
                    module.hopfield_values = value_list[module_index]
                else:
                    module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, key_list[module_index].cuda()], dim=1).cuda().clone().detach())
                    module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                # print(module)
                module_index += 1
                keys_new.append(module.hopfield_keys)
    
    print("Number of keys in model: ", len(keys_new))



train_domain_adaters_list = []
classifiers = {}
classifier = None





prev_keys = []
prev_values = []
# for col in range(adapters_per_domain):
    

    
#     #After every adapter, load hopfield keys and hopfield values in the model and then train
#     #do it for if col > 0->load_hopfield_keys/values_function
#     #hopfield values should match with adapters saved after flattening
    


#     train_domain_adaters = {}
#     domain_models = {}
    
#     # if col > 0:
#     #     keys_to_commit = get_keys_lora(model)
#     #     values_to_commit = get_values_lora(model)

#     if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
#         args.label_offset = 0

#     for did, domain_name in enumerate(train_domains):
#         print(f"Domain {domain_name} ({did})")
#         if args.base_model == 'laion':
            
#             model = deepcopy(vit)
#             lora_cfg = LoraConfig(
#             r=32,
#             lora_alpha=32,
#             target_modules=["q_proj", "v_proj"], 
#             lora_dropout=0.0,
#             bias="none",
#             # task_type="FEATURE_EXTRACTION"
#             )
#             model = get_peft_model(model, lora_cfg)
#             classifier = nn.Linear(512, num_classes, bias=False).cuda()
            
            

#             print("\n" + "="*50)
#             print("🔎 MODEL STRUCTURE")
#             print("="*50)
#             # print(model) 

#             print("\n" + "="*50)
#             print("TRAINABLE PARAMETERS")
#             print("="*50)
#             # for name, param in model.named_parameters():
#             #     if param.requires_grad:
#             #         print(name)

#             print("="*50 + "\n")
#             print(f"--- Training Domain: {domain_name} ---")
#             # model.print_trainable_parameters()
            
            
#             trainable_params = [classifier.weight]
#             # for name, param in model.named_parameters():
#             #     # num_total += param.numel()
#             #     if "lora" in name or "classifier" in name:
#             #         param.requires_grad = True
#             #         # num_trainable += param.numel()
#             #         trainable_params.append(param)
#             #     if "classifier" in name:
#             #         print(name + ":", "Training classifier" if param.requires_grad else "Freezing classifier")
#             trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
#             # trainable_params = []
#             trainable_params.extend(list(classifier.parameters()))


#             # print("Printing model structure :", model)
#         elif args.base_model == 'vit-in21k':
#             model = deepcopy(vit)
#             lora_cfg = LoraConfig(
#             r=32,
#             lora_alpha=32,
#             target_modules=["query", "value"],
#             modules_to_save=["classifier"],
#             lora_dropout=0.0,
#             bias="none",
#             # task_type="FEATURE_EXTRACTION"
#         )
#             model = get_peft_model(model, lora_cfg)
#             # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
#             # classifier = model.classifier
#             classifier = nn.Linear(768, num_classes, bias=False).cuda()
#             model.classifier = classifier
#             # load_vit_weights(model, vit)
            
        
#         train_loader = train_loaders[domain_name]



# weight_dict_keys = {}


model.eval()

@torch.no_grad()
def cls_per_block_hf_base(pixel_values):
    """
    Returns:
      cls_layers: [B, L, D]  (L layers, D hidden size)
    """
    base = model.vit if hasattr(model, "vit") else model  # encoder only
    out  = base(pixel_values, output_hidden_states=True, return_dict=True)
    # print(out.hidden_states.shape)
    hidden = out.hidden_states[1:]                  # drop embeddings -> per-layer outputs
    cls_layers = torch.stack([h[:, 0, :] for h in hidden], dim=1)
    # print("CLS Layers shape : ", cls_layers.shape) # [B, L, D]
    return cls_layers

@torch.no_grad()
def average_cls_per_layer(loader, device="cuda", l2norm=False):
    """Mean CLS per layer over the entire domain dataset."""
    model.eval().to(device)
    total = None
    count = 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        cls_layers = cls_per_block_hf_base(x)     # [B, L, D]
        if total is None:
            total = cls_layers.sum(dim=0)                # [L, D]
        else:
            total += cls_layers.sum(dim=0)
        count += cls_layers.size(0)
    print("Layers ##############", count)
    per_layer = total / max(count, 1)                    # [L, D]
    if l2norm:
        per_layer = F.normalize(per_layer, dim=-1)
    return per_layer

@torch.no_grad()
def seed_hopfield_keys_with_cls_means(per_layer_cls, domain_idx, adapter_number):

    L, D = per_layer_cls.shape
    assigned = 0
    module_index = 0
    for name, module in model.named_modules():
        # Only touch Hopfield-enabled LoRA Linear layers
        
        if isinstance(module, peft.tuners.lora.layer.Linear):
            
            if args.base_model == 'laion':
                prefix = f'resblocks.{module_index}.attn.' + ('q_proj')

            elif args.base_model == 'vit-in21k':
                prefix = f'encoder.layer.{module_index}.attention.attention.' + 'query'

        
            if name.endswith(prefix):
                hk = module.hopfield_keys
                print("shape before seeding", hk.shape)
                vec = per_layer_cls[module_index].to(hk.device, dtype=hk.dtype)  
                if domain_idx > 0 or(domain_idx == 0 and adapter_number > 0):
                    module.hopfield_keys = nn.Parameter(torch.cat([hk, vec.unsqueeze(1)], dim=1))
                elif domain_idx == 0 and adapter_number == 0:
                    module.hopfield_keys = nn.Parameter(vec.unsqueeze(1))
                    # hk.copy_(vec.unsqueeze(1))
                # module_index += 1
                print("after seeding" , module.hopfield_keys.shape, "CLS dim", vec.unsqueeze(0).shape)
        
        
        if isinstance(module, peft.tuners.lora.layer.Linear):
            if args.base_model == 'laion':
                prefix = f'resblocks.{module_index}.attn.' + ('v_proj')
            elif args.base_model == 'vit-in21k':
                prefix = f'encoder.layer.{module_index}.attention.attention.' + 'value'
        
            if name.endswith(prefix):
                hk = module.hopfield_keys
                print("shape before seeding", hk.shape)
                vec = per_layer_cls[module_index].to(hk.device, dtype=hk.dtype)
                if domain_idx > 0 or(domain_idx == 0 and adapter_number > 0):
                    module.hopfield_keys = nn.Parameter(torch.cat([hk, vec.unsqueeze(1)], dim=1))
                elif domain_idx == 0 and adapter_number == 0:
                    module.hopfield_keys = nn.Parameter(vec.unsqueeze(1))
                
                module_index += 1
                print("after seeding" , module.hopfield_keys.shape, "CLS dim", vec.unsqueeze(0).shape)
        # if not isinstance(module, peft.tuners.lora.layer.Linear):
        #     continue
        # if not hasattr(module, "hopfield_keys"):
        #     continue

        # Map module name -> encoder layer index
        # m = re.search(r"encoder\.layer\.(\d+)\.", name)
        # if not m:
        #     continue
        # i = int(m.group(1))
        # if i < 0 or i >= L:
        #     continue

        # hk = module.hopfield_keys
        # vec = per_layer_cls[i].to(hk.device, dtype=hk.dtype)  # [D]

        # # Accept common shapes: [D] or [M, D]
        # if hk.ndim == 1 and hk.numel() == D:
        #     hk.copy_(vec)
        #     assigned += hk.numel()
        # elif hk.ndim == 2 and hk.size(1) == D:
        #     hk.copy_(vec.unsqueeze(0).expand(hk.size(0), -1))
        #     assigned += hk.numel()
        # else:
        #     print(f"[skip] {name}: hopfield_keys shape {tuple(hk.shape)} incompatible with D={D}")
    print(f"Seeded hopfield_keys with {assigned} values from CLS means.") 

for col in range(adapters_per_domain):
    for did, domain_name in enumerate(train_domains):
        print(f"Computing and saving mean CLS for domain {domain_name}...")
        # train_loader = train_loaders[domain_name]
        # cls_embeddings = []
        # for step, batch in enumerate(train_loader):
        #     pixel_values = batch['image'].cuda()
        #     embeddings = model.embeddings(pixel_values)
            
        #     for i, layer in enumerate(model.vit.encoder.layer):
        #         embeddings = layer(embeddings)
        #         if step == 0:
        #             cls_embeddings.append(embeddings[:, 0, :].unsqueeze(1).cuda())
        #         else:
        #             cls_embeddings[i] = torch.cat([cls_embeddings[i], embeddings[:, 0, :].unsqueeze(1).cuda()], dim=1)

        mean_cls = average_cls_per_layer(train_loaders[domain_name], device="cuda", l2norm=False)
        seed_hopfield_keys_with_cls_means(mean_cls, did, col)
                

        
weight_dict_keys = get_keys_lora(model)
    

# weight_dict_values = {}
weight_dict_values = get_values_lora(model)
print("Saving weights to ", f"weights/train_domain_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")

torch.save(weight_dict_keys, f"weights/hopfield_keys_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(weight_dict_values, f"weights/hopfield_values_{args.base_model}_{dataset_name}{args.name_tag}.pt")

# print("here", weight_dict_keys.keys())

# hopfield_keys_loaded = torch.load(f"weights/hopfield_keys_{args.base_model}_{dataset_name}{args.name_tag}.pt")
# hopfield_values_loaded = torch.load(f"weights/hopfield_values_{args.base_model}_{

# print(weight_dict_keys.keys())
# print(hopfield_keys_loaded.keys())
# print(weight_dict_keys.keys())
torch.save(train_domain_adaters_list, f"weights/train_domain_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(classifiers, f"weights/train_domain_classifiers_{args.base_model}_{dataset_name}{args.name_tag}.pt")