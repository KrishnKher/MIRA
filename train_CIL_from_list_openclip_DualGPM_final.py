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
from torchvision.transforms import *
from transformers import ViTFeatureExtractor, ViTForImageClassification as ViT_Pretrained
import peft
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoImageProcessor
from typing import Optional


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
from vil_datasets import build_continual_dataloader

import math

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="OfficeHome")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--later_epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--dgpm_th', type=float, default=0.7)
parser.add_argument('--disable_memory', action='store_true', default=False)
parser.add_argument('--classifier_only', action='store_true', default=False)
parser.add_argument('--separation_function', type=str, default='affine')
args = parser.parse_args()

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

class DualGPM:
    """
    Dual Gradient Projection Memory (DualGPM) for continual learning,
    now supporting:
      - A classifier whose #rows (classes) grows over time.
      - A MultiheadAttention with hopfield_keys whose #rows (keys) grows.
    We project EACH ROW of those parameters through a fixed basis
    of dimension d (the embedding dim), so row‐count can change freely.
    """
    def __init__(self,
                 backbone: nn.Module,
                 classifier : nn.Linear,
                 eps_th: float = 0.9):
        self.backbone   = backbone
        if args.base_model == 'laion':
            self.classifier = classifier
        elif args.base_model == 'vit-in21k':
            self.classifier = classifier
        self.eps_th     = eps_th

        # memories[module] = {'basis': M (d×k), 'is_Ml': bool}
        self.memories = {}

        # Register every MultiheadAttention by its key‐dim D:
        #need to change to peft.lora.Linear
    
        for m in backbone.modules():
            if isinstance(m, peft.tuners.lora.layer.Linear):
                # print("Registering ViTSelfAttention for DualGPM")
                D = m.hopfield_keys.size(-1)   # dim of each key‐vector
                self.memories[m] = {
                    'basis': torch.empty(D, 0).cuda(),
                    'is_Ml': True
                }
                if hasattr(m, 'key_generator') and m.key_generator is not None:
                    for mm in m.key_generator:
                        if isinstance(mm, nn.Linear):
                            D = mm.weight.size(-1)
                            self.memories[mm] = {
                                'basis': torch.empty(D, 0).cuda(),
                                'is_Ml': True
                            }
                if hasattr(m, 'query_generator') and m.query_generator is not None:
                    for mm in m.query_generator:
                        if isinstance(mm, nn.Linear):
                            D = mm.weight.size(-1)
                            self.memories[mm] = {
                                'basis': torch.empty(D, 0).cuda(),
                                'is_Ml': True
                            }

        # Register the classifier (row‐dim = in_features)
        D = self.classifier.in_features
        print("D", D)
        self.memories[self.classifier] = {
            'basis': torch.empty(D, 0).cuda(),
            'is_Ml': True
        }
        print(f"DualGPM: registered {len(self.memories)} modules.")

    def update(self, dataloader):
        """
        Collect one batch of inputs per registered module,
        then expand/reduce each basis via SVD (Eqs. 5–8).
        """
        # 1) Hook each module to grab its input features R ∈ ℝ^{d×N}
        
        inputs = {m: [] for m in self.memories}
        # print("INPUTS", inputs)
        hooks = []
        for m in inputs:
            def hook_fn(mod, inp, out, m=m):
                # print(inp.shape)
                x = inp[0].detach()
                # For attention: x.shape = [B, S, D] → [D, B·S]
                if x.ndim == 3:
                    B,S,D = x.shape
                    x = x.reshape(B*S, D)
                # For classifier: x.shape = [B, D]
                inputs[m].append(x.T)
            hooks.append(m.register_forward_hook(hook_fn))

        # 2) Run one forward pass
        self.backbone.eval()
        with torch.no_grad():
            for data in dataloader:
                feats = self.backbone(data['image'].cuda())
                if args.base_model == 'laion':
                    _     = self.classifier(feats)
                elif args.base_model == 'vit-in21k':
                    _     = self.classifier(feats.logits)
                break

        # remove hooks
        for h in hooks:
            h.remove()

        # 3) Expand or reduce each basis
        for m, mem in self.memories.items():
            try:
                R       = torch.cat(inputs[m], dim=1)  # [d×N]
            except Exception as e:
                print("inputs[m]", m)
                raise e
            M, flag = mem['basis'], mem['is_Ml']

            if flag:
                M_new, new_flag = self._expand_Ml(M, R)
            else:
                M_new, new_flag = self._reduce_Mperp(M, R)

            mem['basis'] = M_new
            mem['is_Ml'] = new_flag

    def project(self):
        """
        For each registered module, project its gradient **row‐wise**:
         - Classifier.weight: shape [C, D]
         - hopfield_keys:      shape [K, D]
        Other parameters (e.g. linear/conv weights) are still
        flattened as before if you register them.
        """
        for m, mem in self.memories.items():
            M, flag = mem['basis'], mem['is_Ml']

            # --- Classifier: row‐wise on [C×D] --- #
            if m is self.classifier:
                G = m.weight.grad  # [C, D]
                # print(G.shape, M.shape)
                # exit()
                if G is None or M.numel()==0:
                    continue
                # (G @ M) is [C,k], then @Mᵀ → [C,D]
                if flag:
                    Gp = G - (G @ M) @ M.T
                else:
                    Gp = (G @ M) @ M.T
                m.weight.grad.copy_(Gp)

            # --- MultiheadAttention hopfield_keys: row‐wise on [K×D] --- #
            elif isinstance(m, CustomAttention) or isinstance(m, peft.tuners.lora.layer.Linear):
                Gk = m.hopfield_keys.grad  # [K, D]
                if Gk is None or M.numel()==0:
                    continue
                if flag:
                    Gp = Gk - ((Gk.T @ M) @ M.T).T
                else:
                    Gp = (Gk @ M) @ M.T
                m.hopfield_keys.grad.copy_(Gp)
            
            elif isinstance(m, nn.Linear):
                # --- Linear: row‐wise on [C×D] --- #
                G = m.weight.grad
                if G is None or M.numel()==0:
                    continue
                if flag:
                    Gp = G - (G @ M) @ M.T
                else:
                    Gp = (G @ M) @ M.T
                m.weight.grad.copy_(Gp)                

            # --- Fallback: flatten-vector (if you ever register others) --- #
            else:
                w = m.weight if not isinstance(m, nn.ParameterList) else m
                g = w.grad.reshape(-1,1)
                if g.numel()==0 or M.numel()==0:
                    continue
                if flag:
                    gp = g - M @ (M.T @ g)
                else:
                    gp = M @ (M.T @ g)
                w.grad.copy_(gp.view_as(w))

    def _expand_Ml(self, M, R):
        # Eq. (5–6) expansion
        R_proj = M @ (M.T @ R) if M.numel() else torch.zeros_like(R)
        R_hat  = R - R_proj
        U, S, _ = torch.linalg.svd(R_hat, full_matrices=False)

        E_tot  = (R**2).sum()
        E_proj = (R_proj**2).sum()
        E_cum  = torch.cumsum(S**2, dim=0)

        req = self.eps_th * E_tot - E_proj
        u   = int(torch.searchsorted(E_cum, req, right=False).item()) + 1

        # print("@@@@@@@@@")
        # print(M.shape)
        M_new = torch.cat([M, U[:, :u]], dim=1) if M.numel() else U[:, :u]
        # print(M_new.shape)
        # print("@@@@@@@@@")

        d, k = R.shape[0], M_new.shape[1]
        if k > d - k:
            # take nullspace columns k…d
            U_all, _, _ = torch.linalg.svd(M_new, full_matrices=True)
            return U_all[:, k:], False
        return M_new, True

    def _reduce_Mperp(self, M, R):
        # Eq. (7–8) reduction
        M.cuda()
        R.cuda()
        R_hat_p = M @ (M.T @ R)
        U_p, S_p, _ = torch.linalg.svd(R_hat_p, full_matrices=False)

        E_cum_p = torch.cumsum(S_p**2, dim=0)
        thr     = (1 - self.eps_th) * (R**2).sum()
        k       = int((E_cum_p <= thr).sum().item())

        Z       = U_p[:, :k]
        M_hat   = M - Z @ (Z.T @ M)
        U_e, S_e, _ = torch.linalg.svd(M_hat, full_matrices=False)

        nz    = (S_e.abs() > 1e-12)
        M_new = U_e[:, nz]

        d, p = R.shape[0], M_new.shape[1]
        if (d - p) < p:
            # take nullspace of M_perp: columns p…d  
            U_all, _, _ = torch.linalg.svd(M_new, full_matrices=True)
            return U_all[:, p:], True
        return M_new, False


class GrowingLinearClassifier(nn.Module):
    """
    A linear classifier whose output dimension (number of classes) can grow.
    When new classes are added, the old weights/biases are zero‐padded
    so that any attached DualGPM memory can also be padded correspondingly.
    """
    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim:   dimensionality of feature vectors coming in
            num_classes: initial number of output classes
        """
        super().__init__()
        self.in_features   = input_dim
        self.num_classes = num_classes

        # Initialize weight [C × D] and bias [C]
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Same scheme as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        returns logits: [batch_size, num_classes]
        """
        return F.linear(x, self.weight)

    def add_classes(self, k: int, weight_init=None):
        """
        Expand the classifier to handle k additional classes.

        After this call:
          • self.num_classes increases by k
          • self.weight and self.bias gain k new rows of zeros at the bottom
        """
        if k <= 0:
            return

        C_old = self.num_classes
        C_new = C_old + k

        # Extract old values
        w_old = self.weight.data
        device, dtype = w_old.device, w_old.dtype

        # Make new, zero‐initialized weight & bias
        w_new = torch.zeros(C_new, self.in_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(w_new, a=math.sqrt(5))

        # Copy old parameters into the top rows
        w_new[:C_old] = w_old

        if weight_init is not None:
            w_new[C_old:].data = weight_init.data

        # Reassign
        self.num_classes = C_new
        self.weight      = nn.Parameter(w_new)

laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
vit = laion.visual.cuda()

if args.base_model == 'laion':
    laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    vit = laion.visual.cuda()

    
    #In set_peft no need to load lora_a and lora_b, load just hopfield_keys and hopfield_values
    #use module.hopfield_keys =  and model.hopfield_values = 
    #then make hopfield_keys as nn.Parameter and pass them to optimizer
    
    #FOr every peft.linear module, make module.use_hopfield = True
    
    model = deepcopy(vit)
    replace_attention_with_custom(model)
    lora_cfg = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.0,
    bias="none",
    # task_type="FEATURE_EXTRACTION"
    )
    
    print(model)
    model = get_peft_model(model, lora_cfg)

elif args.base_model == 'vit-in21k':
    vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=args.num_classes).cuda()
    model_string = 'google/vit-base-patch16-224-in21k'
    preprocess_train = ViTFeatureExtractor.from_pretrained(model_string)
    preprocess_val = preprocess_train
    
    # print(vit)
    
    # replace_attention_with_custom(vit)
    
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
    
    
dataset_name = args.dataset
is_hf_dataset = True
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

elif dataset_name == "iDigits-cil":
    # Epochs 1 and 2
    args.num_tasks  = num_tasks = 5
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
        
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
    # adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_iDigits-cil.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_vit-in21k_iDigits-cilstage1_icil_1110.pt', weights_only=False)
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_iDigits-cilstage1_idigitscil_cls_2210.pt', weights_only=False)
elif dataset_name == "CORe50-cil":
    args.num_tasks  = num_tasks = 5
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/Core50'
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_CORe50-cil_set1.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_CORe50-cil.pt', weights_only=False)
elif dataset_name == "DomainNet-cil":
    #  python train_CIL_from_list_openclip_DualGPM.py --infer_after 2000 --dataset DomainNet-cil --num_classes 69 --epochs 2 --later_epochs 2 --lr 1e-3 --batch_size 128
    args.num_tasks  = num_tasks = 5
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-dil'
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-cil_gelu_set1.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_DomainNet-cil_gelu_set1.pt', weights_only=False)


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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_cifar100_5tasks.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_cifar100_5tasks.pt', weights_only=False)
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_tinyimagenet_5tasks.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_tinyimagenet_5tasks.pt', weights_only=False)
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_inetR_10tasks.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_inetR_10tasks.pt', weights_only=False)
    is_hf_dataset = False





if args.base_model == 'vit-in21k':
    hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_vit-in21k_iDigits-cilstage1_icil_1110.pt')
    hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_vit-in21k_iDigits-cilstage1_icil_1110.pt')
elif args.base_model == 'laion':
    hopfield_keys = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_keys_laion_DomainNet-dilstage_1_laion_withadapters_witheval.pt')
    hopfield_values = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/hopfield_values_laion_DomainNet-dilstage_1_laion_withadapters_witheval.pt')

    #idigits
    hopfield_keys = torch.load('weights/hopfield_keys_laion_iDigits-cilstage1_idigitscil_cls_2210.pt')
    hopfield_values = torch.load('weights/hopfield_values_laion_iDigits-cilstage1_idigitscil_cls_2210.pt')



adapters_per_domain = args.adapters_per_domain
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
parent_dir = f'data/{dataset_name}'
os.makedirs(parent_dir, exist_ok=True)


# classifier = nn.Linear(512, num_classes).cuda()
if args.base_model == 'laion':
    classifier = GrowingLinearClassifier(512, num_classes).cuda()
    classifier.weight.data = classifier_list[0].weight.data
elif args.base_model == 'vit-in21k':
    classifier = GrowingLinearClassifier(768, num_classes).cuda()
    print(classifier_list)
    # exit(0)
    classifier.weight.data = classifier_list[0].weight.data
    model.classifier = model.model.classifier = nn.Identity()
    

# schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, returns_domain=True, label_offset = None):
        self.dataset = dataset
        self.preprocess = preprocess
        self.label_offset = label_offset
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
            if args.base_model == 'vit-in21k':
                item['image'] = self.preprocess(image)
            else:
                item['image'] = self.preprocess(image)
            item['label'] = (label - self.label_offset) if self.label_offset is not None else label
            return item


# Function to implement mixup, mixing different parts from all images in the batch
def mixup(image_batch, mixup_times=1):
    alpha = 0.4
    for i in range(mixup_times):
        lam = np.random.beta(alpha, alpha)
        rand_perm = torch.randperm(image_batch.size(0))
        image_batch = lam * image_batch + (1 - lam) * image_batch[rand_perm]
    return image_batch


@torch.no_grad()
def eval():
    model.eval()
    domain_accs = {}
    for domain_idx, test_loader in enumerate(test_domain_loaders):
        pbar = Progbar(len(test_loader))
        for step, batch in enumerate(test_loader):
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda() + args.num_classes * domain_idx
            if args.base_model == 'laion':
                outputs = classifier(model(pixel_values))
            else:
                outputs = classifier(model(pixel_values).logits)
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        domain_accs[domain_idx] = pbar.get_values()['acc']
    return domain_accs


test_domain_loaders = []
model_hopfield_keys = []
dualGPM = None
preprocess_train = Compose([
    preprocess_train,
    RandomHorizontalFlip(0.5)
])
first_accs = {}
domain_adapters = {}

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
    if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
        dl = dataloaders[domain]
        train_dataset = dl['train']
        train_dataset = DomainDataset(train_dataset, preprocess_train, returns_domain=False, label_offset=args.num_classes * domain_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_dataset = dl['test']
        test_dataset = DomainDataset(test_dataset, preprocess_val, returns_domain=False, label_offset=args.num_classes * domain_idx)
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
                # hopfield_keys = nn.Parameter(torch.ones_like(key_list[module_index]).cuda())
                if module.hopfield_keys is None:
                    # module.hopfield_keys = torch.cat([hopfield_keys], dim=1)
                    module.hopfield_keys = nn.Parameter(key_list[module_index].cuda().clone().detach())
                    module.hopfield_keys.requires_grad = True
                    module.hopfield_values = value_list[module_index]
                else:
                    # module.hopfield_keys = torch.cat([module.hopfield_keys, hopfield_keys], dim=1)
                    module.hopfield_keys = nn.Parameter(torch.cat([module.hopfield_keys, key_list[module_index].cuda()], dim=1).cuda().clone().detach())
                    module.hopfield_keys.requires_grad = True
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

    
    # for p in model_hopfield_keys + train_module_params:
    #     total_params += p.numel()
    # print(f"Total params: {total_params}")
    if args.disable_memory or args.classifier_only:
        print("Not training hopfield keys")
        opt = optim.AdamW(list(classifier.parameters()), lr=args.lr, weight_decay=1e-2)
    else:
        # trainable_params = list(model.parameters())
        trainable_params = list(classifier.parameters())
        opt = optim.AdamW(keys_new  + trainable_params, lr=args.lr, weight_decay=1e-3)

    if dualGPM is None:
        # 0.7 for CIL inetR 5 tasks
        dualGPM = DualGPM(model, classifier, args.dgpm_th)
        
    # print("Trainable Parameters for domain ", domain, ":")
    # print(trainable_params)
    
    # for param in trainable_params:
    #     print(param.requires_grad)


    epochs = args.epochs if (args.later_epochs is None or domain_idx == 0) else args.later_epochs
    for epoch in range(args.epochs if domain_idx == 0 else args.later_epochs):
        model.train()
        running_acc = 0
        pbar = Progbar(len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} for domain {domain}")
        for step, batch in enumerate(train_loader):
            opt.zero_grad()
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda() + args.num_classes * domain_idx
            
            mixed_batch = []
            new_labels = []
            for l in np.unique(labels.cpu().numpy()):
                idx = labels == l
                mixed_batch.append(mixup(pixel_values[idx], 0))
                new_labels.append(labels[idx])
            pixel_values = torch.cat(mixed_batch, dim=0)
            new_labels = torch.cat(new_labels, dim=0)
            labels = new_labels
            if args.base_model == 'vit-in21k':
                out = model(pixel_values)
                outputs = classifier(out.logits)
            elif args.base_model == 'laion':
                outputs = classifier(model(pixel_values))
            # outputs = classifier(model(pixel_values, hopfield_masks=hopfield_masks))
            outputs[:, :-num_classes] = -1e6
            loss = criterion(outputs, labels)

            loss.backward()
            mod_keys = keys_new
            for i, key in enumerate(mod_keys):
                if key is None:
                    print(f"Key {i} is None")
                    # print(prev_keys[i])
                    continue
                assert not torch.isnan(key).any(), f"Key {i} has NaN values"
                # print(key.grad)
            if domain_idx > 0:
                dualGPM.project()
            # for name, param in model.named_parameters():
            #     if param.requires_grad and torch.isnan(param).any():
            #         print(f"Parameter {name} has NaN values after backward")
            #     elif param.requires_grad and param.grad is None:
            #         print(name, " grad is None")
            #     elif param.requires_grad and torch.mean(param.grad) != 0:
            #         print(name)
            opt.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        all_accs = eval()
        forgetting = 0
        for i in range(domain_idx):
            forgetting += first_accs[i] - all_accs[i]
        first_accs[domain_idx] = all_accs[domain_idx]
        print("Avg acc: ", np.mean(list(all_accs.values())))
        print("Avg forgetting: ", forgetting / domain_idx if domain_idx > 0 else 0)
        print()
    if domain_idx < num_tasks - 1:
        dualGPM.update(train_loader)
        # classifier.add_classes(num_classes, classifier_list[0][domain_idx + 1].weight)
        classifier.add_classes(num_classes, classifier_list[domain_idx + 1].weight)
        

# torch.save('key_vectors.pth', model_hopfield_keys)
