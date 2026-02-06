from datasets import load_dataset
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention

from transformers import ViTFeatureExtractor, ViTForImageClassification as ViT_Pretrained
from transformers import TrainingArguments, Trainer
from open_clip_vit import VisionTransformer, MultiheadAttention, to_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import *
from scipy.stats import ttest_ind

from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip
import argparse
import os
from localdatasets import make_VLCS, make_TI
from itertools import chain
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR
import math
from vil_datasets import build_continual_dataloader
import pickle as pkl
import time
import random
import re
from typing import Optional
from torch import Tensor
from peft import LoraConfig, get_peft_model
from transformers import AutoImageProcessor
from torchvision.transforms import InterpolationMode, CenterCrop, Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize, RandomResizedCrop, ToPILImage, ToTensor
import peft





parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DomainNet-dil")
parser.add_argument("--adapters_per_domain", type=int, default=3)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--later_epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=345)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--dgm_th', type=float, default=0.7)
parser.add_argument('--separation_function', type=str, default='softmax')
args = parser.parse_args()
args.label_offset = 0

if args.separation_function == 'affine' or args.separation_function is None:
    args.separation_function = to_dist
elif args.separation_function == 'softmax':
    args.separation_function = F.softmax
elif args.separation_function == 'relu':
    args.separation_function = lambda x, dim: F.relu(x)
elif args.separation_function == 'sigmoid':
    args.separation_function = lambda x, dim: torch.sigmoid(x)
elif args.separation_function == 'tanh':
    args.separation_function = lambda x, dim: torch.tanh(x)

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

dataset_name = args.dataset
is_hf_dataset = True

if args.base_model == 'laion':
    laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    vit = laion.visual.cuda()


elif args.base_model == 'vit-in21k':
    vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=args.num_classes).cuda()
    model_string = 'google/vit-base-patch16-224-in21k'
    preprocess_train = ViTFeatureExtractor.from_pretrained(model_string)
    preprocess_val = preprocess_train


if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
elif dataset_name == "DomainNet":
    dataset = load_dataset("wltjr1007/DomainNet")
    train_domains = [0, 1, 2, 3, 4, 5]
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet.pt')
elif dataset_name == "OfficeHome":
    dataset = load_dataset("flwrlabs/office-home")
    train_domains = ['Art', 'Clipart', 'Product', 'Real World']
elif dataset_name == "VLCS":
    train_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    try:
        dataset = load_dataset("ai22mtech12002/DG_VLCS")
    except:
        dataset = make_VLCS('data/VLCS')
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_VLCS.pt')
elif dataset_name == "TI":
    try:
        dataset = load_dataset("ai22mtech12002/DG_TI")
    except:
        dataset = make_TI('data/terra_incognita')
    train_domains = ['location_38', 'location_43', 'location_46', 'location_100']
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_TI.pt')
elif dataset_name == "DN4IL":
    train_domains = ["real", "clipart", "infograph", "painting", "quickdraw", "sketch"] 
    dataset = load_dataset("ai22mtech12002/DN4IL")
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DN4IL.pt')
elif dataset_name == "CDDB":
    train_domains = ['gaugan', 'biggan' , 'wild', 'whichfaceisreal', 'san']
    dataset = load_dataset("ai22mtech12002/CDDB-hard")
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_CDDB.pt')
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_iDigits-dil.pt')
elif dataset_name == "CORe50-dil":
    args.num_tasks = 8
    args.data_path = '/data1/ai22mtech12002/projects/WeightDG/data/Core50-dil'
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
    adapter_list = torch.load('/data1/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_CORe50-dil.pt')
elif dataset_name == "DomainNet-dil":
    args.num_tasks = 6
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-dil'
    args.task_type = 'dil'
    args.shuffle = True
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    if args.base_model == 'vit-in21k':
        print("Using vit-in21k preprocessing")
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
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
    # adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil.pt')
    # adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil_stage1_saksham_1.pt', weights_only=False)
    # adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil_peft_saksham_lora_custom.pt')
    # print("ADapter list", adapter_list)
    # adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil_peft_saksham_lora_custom_98.pt')
    # adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/shared_adapters_list_laion_DomainNet-dil_peft_saksham_shared_with_model.pt')
    
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_cifar100.pt')

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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_inetR.pt')

    is_hf_dataset = False


# laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
# vit = laion.visual.cuda()





# model = VisionTransformer(
#     224, 16, 768, 12, 12, 4
# ).cuda()
# # classifier = nn.Linear(512, args.num_classes, bias=False).cuda()
# load_laion_weights(model, vit)

epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
parent_dir = f'data/{dataset_name}'
os.makedirs(parent_dir, exist_ok=True)


# classifier = nn.Linear(512, num_classes).cuda()
# # schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()
# first_accs = {}


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



@torch.no_grad()
def eval():
    model.eval()
    domain_accs = {}
    domain_indices = list(range(len(test_domain_loaders)))
    random.shuffle(domain_indices)

# Function to get a unique random domain index
    def get_unique_random_domain():
        if not domain_indices:
            raise ValueError("No more domains left to sample.")
        return domain_indices.pop()

    
    for domain_idx, test_loader in enumerate(test_domain_loaders):
        # rand_domain = get_unique_random_domain()
        # weight_dict = list_model[domain_idx]
        # set_peft_lora_weights(weight_dict)
        # set_peft_lora_weights_hf_vit(weight_dict)
        # set_store_dict(vit, weight_dict)
        
        
        pbar = Progbar(len(test_loader))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # vit.to(device)
        for step, batch in enumerate(test_loader):
            # print(type(batch['image']))
            pixel_values = batch["image"].cuda()
            labels = batch['label'].cuda()
            if args.base_model == 'vit-in21k':
                out = model(pixel_values)
                outputs = out.logits
            else:
                outputs = classifier(model(pixel_values))
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        domain_accs[domain_idx] = pbar.get_values()['acc']
        if args.dataset == 'CORe50-dil':
            return domain_accs
    return domain_accs



test_domain_loaders = []
vit_hopfield_keys = []
dualGPM = None

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
    


import pickle as pkl


# model = pkl.load(open(f'{parent_dir}/{args.base_model}_{args.dataset}_{args.name_tag}.pkl', 'rb'))
# list_model = torch.load(f'/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil.pt', weights_only = False)
# list_model = torch.load(f'/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil_peft_saksham_lora_custom_98.pt', weights_only = False)
# list_model = torch.load(f'/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_vit-in21k_DomainNet-dil_peft_saksham_lora_vit21_98.pt', weights_only = False)
# # print(len(list_model), type(list_model[0]))
# print(model)
# weight_dict = list_model[0]
# print(model[0])
# print(model)
# set_peft_lora_weights(model, model_weights)

# print("a", a)
# list_classifier = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_vit-in21k_DomainNet-dil_peft_saksham_lora_vit21_98.pt', weights_only = False)
# list_classifier = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_DomainNet-dil_peft_saksham_lora_custom_98.pt', weights_only = False)
# state_dict = torch.load(
#     '/data/ai22mtech12002/projects/WeightDG/weights/classifier_without_adapters_laion_DomainNet-dilstage_1_laion_withoutadapters_witheval.pt',
#     map_location="cuda"  
# )

state_dict = torch.load(
    'weights/train_domain_classifiers_vit-in21k_DomainNet-dilstage_1_vitin21k_withadapters_witheval_0409.pt',
    map_location="cuda",
    weights_only=False
)
last_head = state_dict[5]


model = deepcopy(vit)

if args.base_model == 'laion':
    classifier = nn.Linear(512, args.num_classes, bias=False).cuda()
    # classifier.load_state_dict(state_dict)
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
    
    print(classifier)
elif args.base_model == 'vit-in21k':
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
    print(model)
    
    classifier = nn.Linear(768, args.num_classes, bias=False).cuda()
    # classifier.load_state_dict(state_dict)
    model.classifier = classifier
    
hopfield_keys = torch.load('weights/hopfield_keys_vit-in21k_DomainNet-dilstage_1_vitin21k_withadapters_witheval_0409.pt')
hopfield_values = torch.load('weights/hopfield_values_vit-in21k_DomainNet-dilstage_1_vitin21k_withadapters_witheval_0409.pt')

    

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
    
    keys_new = []
    module_index = 0
    hopfield_query_params = []

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
                # print(name, module_index, prefix)
                    
            # FInding module + prefix attr
            # for attr in dir(module):
            #     if 'prefix' in attr:
            #         print(attr)
            # print(module.prefix.hopfield_keys if hasattr(module.prefix, 'hopfield_keys') else "No prefix", prefix)
                if module.hopfield_keys is None:
                    module.hopfield_keys = nn.Parameter(key_list[module_index].cuda().clone().detach())
                    module.hopfield_values = value_list[module_index]
                else:
                    module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, key_list[module_index].cuda()], dim=1).cuda().clone().detach())
                    module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                # print(module)
                module_index += 1
                keys_new.append(module.hopfield_keys)

    # epochs = args.epochs if (args.later_epochs is None or domain_idx == 0) else args.later_epochs




print(model)


eval_accs = eval()
print("Final eval accs: ", eval_accs)