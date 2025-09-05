from datasets import load_dataset
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention
from transformers import ViTImageProcessor, ViTForImageClassification as ViT_Pretrained
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode, RandomResizedCrop, ToTensor, CenterCrop, Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize

from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip
from vil_datasets import build_continual_dataloader
import argparse
import os
from localdatasets import make_VLCS, make_TI
from itertools import chain
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR
import math
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="OfficeHome")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--later_epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()


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
            if isinstance(m, ViTSelfAttention):
                D = m.hopfield_keys.size(-1)   # dim of each key‐vector
                self.memories[m] = {
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
            if isinstance(m, ViTSelfAttention):
                Gk = m.hopfield_keys.grad  # [K, D]
                if Gk is None or M.numel()==0:
                    continue
                if flag:
                    Gp = Gk - (Gk @ M) @ M.T
                else:
                    Gp = (Gk @ M) @ M.T
                m.hopfield_keys.grad.copy_(Gp)
                continue

            # --- Fallback: flatten-vector #
            w = m.weight
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

        M_new = torch.cat([M, U[:, :u]], dim=1) if M.numel() else U[:, :u]

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

elif dataset_name == "DomainNet-cil":
    #  python train_CIL_from_list_openclip_DualGPM.py --infer_after 2000 --dataset DomainNet-cil --num_classes 69 --epochs 2 --later_epochs 2 --lr 1e-3 --batch_size 128
    args.num_tasks  = num_tasks = 5
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-raw'
    args.task_type = 'cil'
    args.shuffle = True
    args.versatile_inc = False
    args.num_workers = 8
    args.pin_mem = True
    preprocess_train = Compose([
            RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    preprocess_val = Compose([
            Resize(size=(256, 256), interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    dataloaders, _, _ = build_continual_dataloader(args=args)
    train_domains = list(range(args.num_tasks))
    is_hf_dataset = False
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_vit_DomainNet-cil.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_vit_DomainNet-cil.pt', weights_only=False)

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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_vit_cifar100.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_vit_cifar100.pt', weights_only=False)
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


model_string = 'google/vit-base-patch16-224-in21k'
preprocess_train = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(0.5),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
preprocess_val = Compose([
        Resize(256),
        CenterCrop(224),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

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


def load_vit_weights(vit):
    base_vit = ViT_Pretrained.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
    vit_state_dict = vit.state_dict()
    pretrained_vit_state_dict = base_vit.state_dict()
    for n, p in vit_state_dict.items():
        if n in pretrained_vit_state_dict and "classifier" not in n:
            vit_state_dict[n] = pretrained_vit_state_dict[n]

    vit.load_state_dict(vit_state_dict)
    del base_vit, pretrained_vit_state_dict
    

def make_hopfield(model, domain):
    # hopfield_query_module = nn.Sequential(
    #     nn.Linear(768, 256),
    #     nn.GELU(),
    #     nn.Linear(256, 256),
    #     nn.GELU(),
    #     nn.Linear(256, 768),
    # ).cuda()
    hopfield_query_module = nn.Identity()
    train_module_params = list(hopfield_query_module.parameters())
    all_keys = []
    for adapters in adapter_list:
        adapters = adapters[domain]
        for name, module in model.named_modules():
            if isinstance(module, ViTSelfAttention):
                if not module.use_hopfield:
                    # hopfield_query_module = nn.Sequential(
                    #     nn.Linear(768, 1024),
                    #     nn.GELU(),
                    #     nn.Linear(1024, 1024),
                    #     nn.GELU(),
                    #     nn.Linear(1024, 768),
                    # ).cuda()
                    hopfield_query_module = nn.Identity()
                    module.init_hopfield(0.5, hopfield_query_module) # , separation_function=F.softmax)
                    train_module_params += list(hopfield_query_module.parameters())
                    module.hopfield_query_generator.cuda()
                keys = torch.ones(1, 768).cuda() + torch.randn(1, 768).cuda() * 1e-6
                # keys =  torch.randn(1, 768).cuda()
                keys = keys / torch.norm(keys, dim=-1, keepdim=True)
                keys = [key for key in keys]
                set_store_dict(model, adapters)
                for k in keys:
                    k.requires_grad = True
                    module.add_hopfield_element(k)
                all_keys += keys
    return all_keys, train_module_params


model = ViTForImageClassification.from_pretrained(model_string).cuda()
load_vit_weights(model)

adapters_per_domain = args.adapters_per_domain
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
parent_dir = f'data/{dataset_name}'
os.makedirs(parent_dir, exist_ok=True)


# classifier = nn.Linear(512, num_classes).cuda()
classifier = GrowingLinearClassifier(768, num_classes).cuda()
classifier.weight.data = classifier_list[0][0].weight.data
# schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, returns_domain=True, label_offset=0):
        self.dataset = dataset
        self.preprocess = preprocess
        self.returns_domain = returns_domain
        self.label_offset = label_offset
        self.to_tensor = ToTensor()
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(list(item['image'].keys()))
        # item['image'].verify()
        if is_hf_dataset:
            item['image'] = self.preprocess(item['image'])
            return {'image': item['image'], 'label': item['label']}
        else:
            if self.returns_domain:
                image, label, _ = item
            else:
                image, label = item
            item = {}
            # print(type(image))
            # exit()
            if isinstance(image, JpegImageFile) or isinstance(image, PngImageFile) or isinstance(image, Image.Image):
                image = self.to_tensor(image)
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
    for domain_idx, test_loader in enumerate(test_domain_loaders):
        pbar = Progbar(len(test_loader))
        for step, batch in enumerate(test_loader):
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda() + args.num_classes * domain_idx
            outputs = classifier(model(pixel_values))
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])


test_domain_loaders = []
model_hopfield_keys = []
dualGPM = None

def get_model_keys(model):
    model_keys = []
    for name, module in model.named_modules():
        if isinstance(module, ViTSelfAttention):
            if module.use_hopfield:
                module_keys = module.hopfield_keys
                model_keys.append(module_keys)
    return model_keys


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
    

    keys, train_module_params = make_hopfield(model, domain)
    if dualGPM is None:
        # 0.7 for CIL inetR 5 tasks
        dualGPM = DualGPM(model, classifier, 0.7)
    
    model_hopfield_keys = get_model_keys(model)
    total_params = 0
    for p in model_hopfield_keys + train_module_params:
        total_params += p.numel()
    print(f"Total params: {total_params}")
    # opt = optim.AdamW(model_hopfield_keys + train_module_params, lr=args.lr, weight_decay=1e-2)
    opt = optim.AdamW(model_hopfield_keys + train_module_params + list(classifier.parameters()), lr=args.lr, weight_decay=1e-3)
    # opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()), lr=args.lr, weight_decay=1e-4)

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
            hopfield_masks = []
            for l in np.unique(labels.cpu().numpy()):
                idx = labels == l
                mixed_batch.append(mixup(pixel_values[idx], 0))
                new_labels.append(labels[idx])
            pixel_values = torch.cat(mixed_batch, dim=0)
            new_labels = torch.cat(new_labels, dim=0)
            hopfield_masks = 1
            labels = new_labels
            outputs = classifier(model(pixel_values))
            outputs[:, :-num_classes] = -1e6
            loss = criterion(outputs, labels)

            loss.backward()
            if domain_idx > 0:
                dualGPM.project()
            k = get_model_keys(model)
            for _k in k:
                assert k.grad is not None, "Gradient for hopfield keys is None"
            opt.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        eval()
        print()
    if domain_idx < num_tasks - 1:
        dualGPM.update(train_loader)
        classifier.add_classes(num_classes, classifier_list[0][domain_idx + 1].weight)
