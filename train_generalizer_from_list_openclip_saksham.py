import os
os.environ['CURL_CA_BUNDLE'] = ''

from datasets import load_dataset
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention
from transformers import TrainingArguments, Trainer
from open_clip_vit import VisionTransformer, MultiheadAttention, to_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR
from vil_datasets import build_continual_dataloader
from typing import Optional, Tuple
from torch import Tensor 
from transformers import ViTFeatureExtractor, ViTForImageClassification as ViT_Pretrained
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel, CLIPImageProcessor

from torchvision.transforms import Compose, Resize, CenterCrop, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from PIL import Image
from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip
import argparse
import os
from localdatasets import make_VLCS, make_TI
from margin_loss import LargeMarginLoss
import cv2
import numpy as np
from torchvision.transforms import *
from transformers import AutoImageProcessor
import peft
from peft import LoraConfig, get_peft_model

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
                    elif args.base_model == 'vit-b32':
                        new_name = name[name.find('encoder.layers.'):]
                    weight_dict[new_name] = getattr(module, attr).detach().clone()

    return weight_dict

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
                    elif args.base_model == 'vit-b32':
                        new_name = name[name.find('encoder.layers.'):]
                    weight_dict[new_name] = getattr(module, attr).detach().clone()
                    
    return weight_dict

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

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DomainNet")
parser.add_argument("--adapters_per_domain", type=int, default=5)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=345)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k', 'vit-b32'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument("--test_domain", type=str, default='5')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--separation_function', type=str, default='affine')
parser.add_argument('--include_seen_domain', action='store_true')
# parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])


#take argument for mode = eval and 
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
    
dataset_name = args.dataset
is_hf_dataset = True

if args.base_model == 'laion':
    laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    vit = laion.visual.cuda()

elif args.base_model == 'vit-in21k':
    vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=args.num_classes).cuda()
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]

    model_string = 'google/vit-base-patch16-224-in21k'
    preprocess_train = ViTFeatureExtractor.from_pretrained(model_string)
    preprocess_val = preprocess_train
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
elif args.base_model == 'vit-b32':
    vit = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
    ip = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    preprocess_train = Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0),
                          ratio=(0.75, 1.3333),
                          interpolation=InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
        Normalize(mean=ip.image_mean, std=ip.image_std),
    ])
    preprocess_val = Compose([
        Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=ip.image_mean, std=ip.image_std),
    ])
    
    print(vit)
    
if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    test_domain = args.test_domain
    args.num_tasks = 4
    
    if args.base_model == 'vit-in21k':
        ip = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
    elif args.base_model == 'vit-b32':
        ip = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
    
    train_domains.remove(args.test_domain)
    if args.base_model == 'laion':
        hopfield_keys = torch.load('weights/hopfield_keys_laion_PACSstage1_PACS_1110.pt')
        hopfield_values = torch.load('weights/hopfield_values_laion_PACSstage1_PACS_1110.pt')
        
    
elif dataset_name == "DomainNet":
    dataset = load_dataset("wltjr1007/DomainNet")
    train_domains = [0, 1, 2, 3, 4, 5]
    if not args.include_seen_domain:
        train_domains.remove(int(args.test_domain))
    test_domain = int(args.test_domain)
    
    args.num_tasks = 6
elif dataset_name == "OfficeHome":
    dataset = load_dataset("flwrlabs/office-home")
    train_domains = ['Art', 'Clipart', 'Product', 'Real World']
    train_domains.remove(args.test_domain)
    args.num_tasks = 4
    test_domain = args.test_domain
    if args.base_model == 'vit-in21k':
        ip = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
    elif args.base_model == 'vit-b32':
        ip = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
    if args.base_model == 'laion':
        hopfield_keys = torch.load('weights/hopfield_keys_laion_OfficeHomestage1_officehome_1110.pt')
        hopfield_values = torch.load('weights/hopfield_values_laion_OfficeHomestage1_officehome_1110.pt')
        
        
    # train_domains.remove(args.test_domain)
elif dataset_name == "VLCS":
    train_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    args.num_tasks =4
    try:
        dataset = load_dataset("ai22mtech12002/DG_VLCS")
    except:
        dataset = make_VLCS('data/VLCS')
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    if args.base_model == 'vit-in21k':
        ip = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
    elif args.base_model == 'vit-b32':
        ip = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
    if args.base_model == 'laion':
        hopfield_keys = torch.load('weights/hopfield_keys_laion_VLCSstage1_VLCS_1110.pt')
        hopfield_values = torch.load('weights/hopfield_values_laion_VLCSstage1_VLCS_1110.pt')
elif dataset_name == "TI":
    try:
        dataset = load_dataset("ai22mtech12002/DG_TI")
    except:
        dataset = make_TI('data/terra_incognita')
    train_domains = ['location_38', 'location_43', 'location_46', 'location_100']
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain

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
    elif args.base_model == 'vit-b32':
        ip = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        mean = ip.image_mean if hasattr(ip, "image_mean") else [0.5, 0.5, 0.5]
        std  = ip.image_std  if hasattr(ip, "image_std")  else [0.5, 0.5, 0.5]
        
        preprocess_train = Compose([
                RandomResizedCrop(
                    size=(224, 224),
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.3333),
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        preprocess_val = Compose([
                Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=mean, std=std),
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
    test_domain = int(args.test_domain)
    train_domains.remove(int(test_domain))
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
    is_hf_dataset = False


print("Train domains: ", train_domains)
adapters_per_domain = args.adapters_per_domain
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
parent_dir = f'data/{dataset_name}'
os.makedirs(parent_dir, exist_ok=True)

preprocess_train = Compose([
    RandomHorizontalFlip(),
    preprocess_train
])

model = deepcopy(vit)
# if args.base_model == 'vit-in21k':
#     state_dict = torch.load(
#         '/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers__vit-in21k_DomainNet-dilcls_tokens.pt',
#         map_location="cuda",
#         weights_only=False
#     )
# elif args.base_model == 'laion':
#     state_dict = torch.load(
#         '/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_DomainNet-dilstage_1_laion_withadapters_witheval.pt',
#         map_location="cuda",
#         weights_only=False
#     )
# last_head = state_dict[5]

if args.base_model == 'laion':
    replace_attention_with_custom(model)
    classifier = nn.Linear(512, num_classes, bias=False).cuda()
    # classifier.load_state_dict(last_head.state_dict())
    lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.0,
    bias="none",
    # task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(model, lora_cfg)
elif args.base_model == 'vit-in21k':
    classifier = nn.Linear(768, num_classes, bias=False).cuda()
    # classifier.load_state_dict(last_head.state_dict())
    # model.classifier = classifier
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
    model.classifier = model.model.classifier = nn.Identity()

elif args.base_model == 'vit-b32':
    # model = deepcopy(vit)
    lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    # modules_to_save=["classifier"],
    lora_dropout=0.0,
    bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    classifier = nn.Linear(768, num_classes, bias=False).cuda()



# hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_vit-in21k_DomainNet-dilstage_1_vitin21k_withadapters_witheval_0409.pt')
# hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_vit-in21k_DomainNet-dilstage_1_vitin21k_withadapters_witheval_0409.pt')
if args.base_model == 'vit-in21k':
    hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_vit-in21k_DomainNet-dilcls_tokens.pt')
    hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_vit-in21k_DomainNet-dilcls_tokens.pt')
    
    hopfield_keys = torch.load('weights/hopfield_keys_vit-in21k_PACSpacs_stage1_0310.pt')
    hopfield_values = torch.load('weights/hopfield_values_vit-in21k_PACSpacs_stage1_0310.pt')
    
    hopfield_keys = torch.load('weights/hopfield_keys_vit-in21k_OfficeHomeOfficeHome_stage1_0510.pt')
    hopfield_values = torch.load('weights/hopfield_values_vit-in21k_OfficeHomeOfficeHome_stage1_0510.pt')

    # hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_vit-in21k_VLCSvlcs_stage1_0410.pt')
    # hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_vit-in21k_VLCSvlcs_stage1_0410.pt')

    # hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/data/PACS/hopfield_keys_PACS_art_painting_vit-in21k_adapters10_epochs2_bs128_lr0.001vit_in21k_pacs_stage2.pt')
    # # hopfield_values = torch.load('')
    # hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/data/PACS/hopfield_values_PACS_art_painting_vit-in21k_adapters10_epochs2_bs128_lr0.001vit_in21k_pacs_stage2.pt')
    # print(hopfield_keys['encoder.layer.9.attention.attention.value'])
    # print(list(hopfield_values.keys()))
    # exit(0)

elif args.base_model == 'laion':
    hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_laion_DomainNet-dilstage_1_laion_withadapters_witheval.pt')
    hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_laion_DomainNet-dilstage_1_laion_withadapters_witheval.pt')

elif args.base_model == 'vit-b32':
    #OH
    hopfield_keys = torch.load('weights/hopfield_keys_vit-b32_OfficeHomestage1_OH_withoutcls_1910.pt')
    hopfield_values = torch.load('weights/hopfield_values_vit-b32_OfficeHomestage1_OH_withoutcls_1910.pt')
    
    #PACS
    hopfield_keys = torch.load('weights/hopfield_keys_vit-b32_PACSstage1_pacs_cls_1910.pt')
    hopfield_values = torch.load('weights/hopfield_values_vit-b32_PACSstage1_pacs_cls_1910.pt')
    
    #VLCS
    hopfield_keys = torch.load('weights/hopfield_keys_vit-b32_VLCSstage1_vlcs_cls_1910.pt')
    hopfield_values = torch.load('weights/hopfield_values_vit-b32_VLCSstage1_vlcs_cls_1910.pt')
    
    if args.dataset == "DomainNet-dil" or args.dataset == "DomainNet":
        hopfield_keys = torch.load('weights/hopfield_keys_vit-b32_DomainNet-dilstage1_DomainNet_2110.pt')
        hopfield_values = torch.load('weights/hopfield_values_vit-b32_DomainNet-dilstage1_DomainNet_2110.pt')

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
elif args.base_model == 'vit-b32':
    for domain_idx, domain in enumerate(train_domains):
        domain_adapters[domain_idx] = {
            "keys_q" : [
                hopfield_keys[f'encoder.layers.{i}.self_attn.q_proj'][:, domain_idx :: args.num_tasks]
                for i in range(12)
            ],
            "values_q" : [
                hopfield_values[f'encoder.layers.{i}.self_attn.q_proj'][domain_idx :: args.num_tasks, :]
                for i in range(12)
            ],
            "keys_v" : [
                hopfield_keys[f'encoder.layers.{i}.self_attn.v_proj'][:, domain_idx :: args.num_tasks]
                for i in range(12)
            ],
            "values_v" : [
                hopfield_values[f'encoder.layers.{i}.self_attn.v_proj'][domain_idx :: args.num_tasks, :]
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
            elif args.base_model == 'vit-b32':
                prefix = f'encoder.layers.{module_index}.self_attn.' + 'q_proj'
                key_list = keys_q
                value_list = values_q
            

            # exit()
            if name.endswith(prefix):
                # print("Found", name)
                # exit(0)
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
                # hopfield_keys = nn.Parameter(torch.ones_like(key_list[module_index]).cuda())
                if module.hopfield_keys is None:
                    # module.hopfield_keys = torch.cat([hopfield_keys], dim=1)
                    module.hopfield_keys = nn.Parameter(key_list[module_index].cuda().clone().detach(), requires_grad=True)
                    # module.hopfield_keys.requires_grad = True
                    module.hopfield_values = value_list[module_index]
                else:
                    # module.hopfield_keys = torch.cat([module.hopfield_keys, hopfield_keys], dim=1)
                    module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, key_list[module_index].cuda()], dim=1).cuda().clone().detach(), requires_grad=True)
                    # module.hopfield_keys.requires_grad = True
                    module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                
                module_index += 1
                keys_new.append(module.hopfield_keys)
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
            elif args.base_model == 'vit-b32':
                prefix = f'encoder.layers.{module_index}.self_attn.' + 'v_proj'
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
                    module.hopfield_keys = nn.Parameter(key_list[module_index].cuda().clone().detach(), requires_grad=True)
                    # module.hopfield_keys.requires_grad = True
                    module.hopfield_values = value_list[module_index]
                else:
                    module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, key_list[module_index].cuda()], dim=1).cuda().clone().detach(), requires_grad=True)
                    # module.hopfield_keys.requires_grad = True
                    module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                # print(module)
                module_index += 1
                keys_new.append(module.hopfield_keys)
    
trainable_params = list(classifier.parameters())

opt = optim.AdamW(keys_new, lr=args.lr * 1e-1, weight_decay=1e-2)

cls_opt = optim.AdamW(list(classifier.parameters()), lr=args.lr, weight_decay=1e-2)


# if args.base_model == 'laion':
#     cls_opt = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=1e-2)
# elif args.base_model == 'vit-in21k':
#     # trainable_params += list(model.classifier.parameters())
#     cls_opt = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=1e-2)



schd = optim.lr_scheduler.StepLR(opt, 10, 0.5)
criterion = nn.CrossEntropyLoss()



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
                item['image'] = self.preprocess(item['image'])
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


if args.dataset in ['DomainNet-dil']:
    dls = []
    train_datasets = []
    for d in train_domains:
        dls.append(dataloaders[d])
        train_datasets.append(dls[-1]['train'])
    train_dataset = [DomainDataset(ds, preprocess_train, returns_domain=False) for ds in train_datasets]
    train_dataset = torch.utils.data.ConcatDataset(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_dataset = dataloaders[test_domain]['test']
    test_dataset = DomainDataset(test_dataset, preprocess_val, returns_domain=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
else:
    train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)
    try:
        train_dataset = train_dataset['train']
    except KeyError:
        pass    
    train_dataset = DomainDataset(train_dataset, preprocess_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = dataset.filter(lambda x: x['domain'] == test_domain)
    try:
        test_dataset = test_dataset['train']
    except KeyError:
        pass
    test_dataset = DomainDataset(test_dataset, preprocess_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

@torch.no_grad()
def mixup(image_batch, mixup_times=1):
    alpha = 0.4
    for i in range(mixup_times):
        lam = np.random.beta(alpha, alpha)
        rand_perm = torch.randperm(image_batch.size(0))
        image_batch = lam * image_batch + (1 - lam) * image_batch[rand_perm]
    return image_batch
@torch.no_grad()
def cutmix(images: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    B, C, H, W = images.size()
    device = images.device

    lam = np.random.beta(beta, beta)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    perm = torch.randperm(B, device=device)
    mixed = images.clone()

    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    return mixed

def eval():
    model.eval()
    pbar = Progbar(len(test_loader))
    for step, batch in enumerate(test_loader):
        pixel_values = batch['image'].cuda()
        labels = batch['label'].cuda()
        if args.base_model == 'laion':
            outputs = classifier(model(pixel_values))
        elif args.base_model == 'vit-b32':
            out = model(pixel_values)
            feats = out.pooler_output
            outputs = classifier(feats)
        else:
            out = model(pixel_values)
            outputs = classifier(out.logits)
            # outputs = model(pixel_values).logits
        loss = criterion(outputs, labels)

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        

marginloss = LargeMarginLoss()


# eval()
# print()


for epoch in range(args.epochs):
    model.train()
    running_acc = 0
    pbar = Progbar(len(train_loader))
    print(f"Epoch: {epoch}")
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
        cls_opt.zero_grad()
        pixel_values = batch['image'].cuda()
        labels = batch['label'].cuda()
        
        mixed_batch = []
        new_labels = []
        hopfield_masks = []
        for l in np.unique(labels.cpu().numpy()):
            idx = labels == l
            mx = mixup(pixel_values[idx], 1)    
            mixed_batch.append(mx)
            new_labels.append(labels[idx])
            # hopfield_mask = torch.ones(labels[idx].shape[0], len(train_domains)*adapters_per_domain).cuda()
            # hopfield_mask[:, l::len(train_domains)] = -torch.inf
            # hopfield_masks.append(hopfield_mask.cuda())
        pixel_values = torch.cat(mixed_batch, dim=0)
        new_labels = torch.cat(new_labels, dim=0)
        

        # hopfield_masks = torch.cat(hopfield_masks, dim=0)
        hopfield_masks = 1
        labels = new_labels
        feats = model(pixel_values)
        if args.base_model == 'laion':
            outputs = classifier(feats)
        elif args.base_model == 'vit-b32':
            outputs = classifier(feats.pooler_output)
        else:
            outputs = classifier(feats.logits)
        loss = criterion(outputs, labels) # + marginloss(outputs, labels, [feats])
        # if torch.isnan(loss).any():
        #     print("NaN loss")
        #     continue
        loss.backward()
        # for name, module in model.named_modules():
        #     if isinstance(module, peft.tuners.lora.layer.Linear):
        #         if module.hopfield_keys.grad is not None:
                    
        #             print(f"{name} grad norm : {module.hopfield_keys.grad}")
        #         else:
        #             print(f"{name} grad norm : None")
        # mod_keys = get_model_keys(model)
        # for i, key in enumerate(mod_keys):
        #     if key.grad is None or torch.isnan(key.grad).any():
        #         print(f"Key {i} has NaN grad")
        #         opt.zero_grad()
        #         cls_opt.zero_grad()
        #         continue
        opt.step()
        cls_opt.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])


    eval()
#     print()
    
torch.save(classifier.state_dict(), os.path.join(parent_dir, f'classifier_{dataset_name}_{test_domain}_{args.base_model}_adapters{adapters_per_domain}_epochs{epochs}_bs{batch_size}_lr{args.lr}{args.name_tag}.pt'))
hopfield_keys_save = get_keys_lora(model)
torch.save(hopfield_keys_save, os.path.join(parent_dir, f'hopfield_keys_{dataset_name}_{test_domain}_{args.base_model}_adapters{adapters_per_domain}_epochs{epochs}_bs{batch_size}_lr{args.lr}{args.name_tag}.pt'))
hopfield_values_save = get_values_lora(model)
torch.save(hopfield_values_save, os.path.join(parent_dir, f'hopfield_values_{dataset_name}_{test_domain}_{args.base_model}_adapters{adapters_per_domain}_epochs{epochs}_bs{batch_size}_lr{args.lr}{args.name_tag}.pt'))

#For random keys
# for domain_idx, domain in enumerate(train_domains):
#     prev_keys = None
#     keys_new = []
#     module_index = 0
#     for name, module in model.named_modules():
#         # print(name)
#         if isinstance(module, peft.tuners.lora.layer.Linear):
#             # print("Found")
#             # module.use_hopfield = True
            
#             if args.base_model == 'laion':
#                 prefix = f'resblocks.{module_index}.attn.' + ('q_proj')
#             elif args.base_model == 'vit-in21k':
#                 prefix = f'encoder.layer.{module_index}.attention.attention.' + 'query'
            

#             # exit()
#             if name.endswith(prefix):
#                 module.hopfield_keys = nn.Parameter(torch.rand_like(module.hopfield_keys))
#                 # if module.hopfield_keys is None:
#                 #     # module.hopfield_keys = torch.cat([hopfield_keys], dim=1)
#                 #     module.hopfield_keys = nn.Parameter(key_list[module_index].cuda().clone().detach(), requires_grad=True)
#                 #     # module.hopfield_keys.requires_grad = True
#                 #     module.hopfield_values = value_list[module_index]
#                 # else:
#                 #     # module.hopfield_keys = torch.cat([module.hopfield_keys, hopfield_keys], dim=1)
#                 #     module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, key_list[module_index].cuda()], dim=1).cuda().clone().detach(), requires_grad=True)
#                 #     # module.hopfield_keys.requires_grad = True
#                 #     module.hopfield_values = torch.cat([module.hopfield_values, value_list[module_index].cuda()], dim=0)
                
#                 module_index += 1
#         # exit()
#     module_index = 0
    
#     for name, module in model.named_modules():
#         # print(name)
#         if isinstance(module, peft.tuners.lora.layer.Linear):
#             # print("Found")
            
#             if args.base_model == 'laion':
#                 prefix = f'resblocks.{module_index}.attn.' + ('v_proj')

#             elif args.base_model == 'vit-in21k':
#                 prefix = f'encoder.layer.{module_index}.attention.attention.' + 'value'

#             # exit()
#             if name.endswith(prefix):
#                 module.hopfield_keys = nn.Parameter(torch.rand_like(module.hopfield_keys))
#                 module_index += 1

# eval()
