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
                 classifier: nn.Linear,
                 eps_th: float = 0.9):
        self.backbone   = backbone
        self.classifier = classifier
        self.eps_th     = eps_th

        # memories[module] = {'basis': M (d×k), 'is_Ml': bool}
        self.memories = {}

        # Register every MultiheadAttention by its key‐dim D:
        for m in backbone.modules():
            if isinstance(m, MultiheadAttention):
                D = m.hopfield_keys.size(-1)   # dim of each key‐vector
                self.memories[m] = {
                    'basis': torch.empty(D, 0),
                    'is_Ml': True
                }
                if not isinstance(m.hopfield_query_generator, nn.Identity):
                    for l in m.hopfield_query_generator.modules():
                        if isinstance(l, nn.Linear):
                            D = l.weight.size(1)
                            self.memories[l] = {
                                'basis': torch.empty(D, 0),
                                'is_Ml': True
                            }

        # Register the classifier (row‐dim = in_features)
        D = classifier.in_features
        self.memories[classifier] = {
            'basis': torch.empty(D, 0),
            'is_Ml': True
        }

    def update(self, dataloader):
        """
        Collect one batch of inputs per registered module,
        then expand/reduce each basis via SVD (Eqs. 5–8).
        """
        # 1) Hook each module to grab its input features R ∈ ℝ^{d×N}
        inputs = {m: [] for m in self.memories}
        hooks = []
        for m in inputs:
            def hook_fn(mod, inp, out, m=m):
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
                _     = self.classifier(feats)
                break

        # remove hooks
        for h in hooks:
            h.remove()

        # 3) Expand or reduce each basis
        for m, mem in self.memories.items():
            R       = torch.cat(inputs[m], dim=1)  # [d×N]
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
                continue

            # --- MultiheadAttention hopfield_keys: row‐wise on [K×D] --- #
            if isinstance(m, MultiheadAttention):
                Gk = m.hopfield_keys.grad  # [K, D]
                if Gk is None or M.numel()==0:
                    continue
                if flag:
                    Gp = Gk - ((Gk.T @ M) @ M.T).T
                else:
                    Gp = (Gk @ M) @ M.T
                m.hopfield_keys.grad.copy_(Gp)
                continue
            
            if isinstance(m, nn.Linear):
                # --- Linear: row‐wise on [C×D] --- #
                G = m.weight.grad
                if G is None or M.numel()==0:
                    continue
                if flag:
                    Gp = G - (G @ M) @ M.T
                else:
                    Gp = (G @ M) @ M.T
                m.weight.grad.copy_(Gp)
                continue

            # --- Fallback: flatten-vector (if you ever register others) --- #
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
    '/data/ai22mtech12002/projects/WeightDG/weights/classifier_without_adapters_vit-in21k_DomainNet-dilstage_1_vitin21k_withoutadapters_witheval.pt',
    map_location="cuda"
)
model = deepcopy(vit)

if args.base_model == 'laion':
    classifier = nn.Linear(512, args.num_classes, bias=False).cuda()
    classifier.load_state_dict(state_dict)
    print(classifier)
elif args.base_model == 'vit-in21k':
    classifier = nn.Linear(768, args.num_classes, bias=False).cuda()
    classifier.load_state_dict(state_dict)
    model.classifier = classifier
# classifier = pkl.load(open(f'{parent_dir}/{args.base_model}_{args.dataset}_{args.name_tag}_classifier.pkl', 'rb'))
# print(get_model_keys(model))
# set_random_keys(model)
# print(get_model_keys(model))

        

# with torch.no_grad():
#     dummy_input = torch.randn(10, 1, 3, 224, 224).cuda()
#     for i in range(10):
#         start = time.time()
#         _ = model(dummy_input[i])
#         end = time.time()
#         print(f"Time taken for forward pass {i+1}: {end - start:.4f} seconds")



# laion_vit, preprocess_train, preprocess_val = open_clip.create_model_and_transforms("hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K")
# vit = laion_vit.visual.cuda()








#loading saved state_dict in vit

model.load_state_dict(torch.load('/data/ai22mtech12002/projects/WeightDG/weights/model_without_adapters_vit-in21k_DomainNet-dilstage_1_vitin21k_withoutadapters_witheval.pt'))
model = model.cuda()







# vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=args.num_classes).cuda()
# model_string = 'google/vit-base-patch16-224-in21k'
# preprocess_train = ViTFeatureExtractor.from_pretrained(model_string)
# preprocess_val = preprocess_train
# print(vit)
# print(model)
# _ = vit(torch.randn(1, 3, 224, 224).cuda())
# print(_.shape)
# print(vit.transformer.resblocks[0].attn.in_proj_weight)
# print(vit)
# exit(0)


# replace_attention_with_custom(vit)

print(model)

# lora_cfg = LoraConfig(
# r=32,
# lora_alpha=32,
# target_modules=["q_proj", "v_proj"], 
# lora_dropout=0.0,
# bias="none",
# # task_type="FEATURE_EXTRACTION"
# )

# lora_cfg_vitin = LoraConfig(
# r=32,
# lora_alpha=32,
# target_modules=["query", "value"], 
# lora_dropout=0.0,
# bias="none",
# # task_type="FEATURE_EXTRACTION"
# )
# vit = get_peft_model(vit, lora_cfg)

# set_peft_lora_weights(weight_dict)

# t_stat, p_val = eval_with_ttest()

eval_accs = eval()
print("Final eval accs: ", eval_accs)