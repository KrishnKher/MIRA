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

if args.base_model == 'laion':
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
            # print("REPLACING###########################")
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
                # print("CALEED########################")
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



    laion_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")
    vit = laion_model.visual.cuda()
    # _ = vit(torch.randn(1, 3, 224, 224).cuda())
    # print(_.shape)
    # print(vit.transformer.resblocks[0].attn.in_proj_weight)
    # print(vit)
    # exit(0)
    

    replace_attention_with_custom(vit)
    # vit = set_lorank_trainable(vit, True, freeze_base=True, freeze_linear_base=True, freeze_lnorm=True)
    # vit = convert_to_custom_attention(vit)
    # print(vit)
    # vit = inject_linear_attention(
    #     model = vit,
    #     encoders = {"transformer"},
    #     embed_dim=768,
    #     num_heads=12
    # )
    print(vit)
    

    # print("Model structure before conversion:")
    # print(vit)
    # print("Converting model to use CustomAttention layers...")
    # vit = convert_to_custom_attention(vit)


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
    def set_peft_lora_weights(model: nn.Module, weight_dict: dict):
        weight_dict = deepcopy(weight_dict)
        for i, resblock in enumerate(model.transformer.resblocks):
            if i in weight_dict:
                current_pos = 0
                for proj_name in ["q_proj", "v_proj"]: 
                    try:
                        proj_layer = getattr(resblock.attn, proj_name)
                        lora_A_layer = proj_layer.lora_A['default']
                        lora_B_layer = proj_layer.lora_B['default']
                        device = lora_A_layer.weight.device
                        
                        len_A = lora_A_layer.weight.numel()
                        lora_A_layer.weight.data = weight_dict[i][current_pos:current_pos+len_A].reshape(lora_A_layer.weight.shape).to(device)
                        current_pos += len_A
                        
                        len_B = lora_B_layer.weight.numel()
                        lora_B_layer.weight.data = weight_dict[i][current_pos:current_pos+len_B].reshape(lora_B_layer.weight.shape).to(device)
                        current_pos += len_B
                    except AttributeError:
                        print(f"Warning: Could not find LoRA layers in resblock {i}, projection {proj_name} to load weights.")
                        continue

elif args.base_model == 'vit-in21k':
    vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
    model_string = 'google/vit-base-patch16-224-in21k'
    preprocess_train = ViTFeatureExtractor.from_pretrained(model_string)
    preprocess_val = preprocess_train

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


shared_adapters_list = []
classifiers = {}


#only implemented for laion model right now

for col in range(adapters_per_domain):
    print(f"\n########## Training SHARED adapter set {col+1}/{adapters_per_domain} ##########")
    
    model = deepcopy(vit)
    lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.0,
    bias="none"
    # task_type="FEATURE_EXTRACTION")
    )
    model = get_peft_model(model, lora_cfg)
    
    use_shared_classifier = (args.dataset in ['iDigits-dil', 'DomainNet-dil', 'CORe50-dil'])
    
    if use_shared_classifier:
        classifier = nn.Linear(512, num_classes, bias=False).cuda()
    
    
    if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
            args.label_offset = 0
     


    # generate a dictionary mapping 0-6 -> 0-6 in a random permutation
    perm = torch.randperm(num_classes)
    perm_dict = {i: perm[i].item() for i in range(num_classes)}

    if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
        args.label_offset = 0

    for did, domain_name in enumerate(train_domains):
        
        train_loader = train_loaders[domain_name]
        
        
        if not use_shared_classifier:
            classifier = nn.Linear(512, num_classes, bias=False).cuda()
            
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        trainable_params += list(classifier.parameters())
        
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-12)
        criterion = nn.CrossEntropyLoss()
        
        print(f"\n--- Training domain/task: {domain_name} (shared adapters) ---")

        for epoch in range(epochs):
            pbar = Progbar(len(train_loader))
            acc_list = []
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pixel_values = batch['image'].cuda()
                labels = batch['label'].cuda()
                outputs = classifier(model(pixel_values))
                # print(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                acc_list.append(acc)
                pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
            if acc_list and (sum(acc_list)/len(acc_list) > 0.99):
                break
            
        
        classifiers[domain_name] = classifier
    
    shared_adapter = get_peft_lora_weights(model)
    
    shared_adapters_list.append(shared_adapter)

        


# os.makedirs("weights", exist_ok=True)
torch.save(shared_adapters_list, f"weights/shared_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(classifiers, f"weights/train_domain_classifiers_{args.base_model}_{dataset_name}{args.name_tag}.pt")
print("✅ Saved shared adapters (list) and classifiers.")