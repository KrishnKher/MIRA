from datasets import load_dataset
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention
from transformers import ViTImageProcessor, ViTForImageClassification as ViT_Pretrained
from open_clip_vit import VisionTransformer, MultiheadAttention, to_dist
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR
from vil_datasets import build_continual_dataloader
from torchvision.transforms import ToTensor, CenterCrop, Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile

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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="OfficeHome")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=2000)
parser.add_argument("--test_domain", type=str)
parser.add_argument('--lr', type=float, default=1e-3)
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

dataset_name = args.dataset
is_hf_dataset = True
if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_PACS.pt')
elif dataset_name == "DomainNet":
    dataset = load_dataset("wltjr1007/DomainNet")
    train_domains = [0, 1, 2, 3, 4, 5]
    train_domains.remove(int(args.test_domain))
    test_domain = int(args.test_domain)
    args.num_tasks = 6
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_vit_DomainNet.pt')
elif dataset_name == "OfficeHome":
    dataset = load_dataset("flwrlabs/office-home")
    train_domains = ['Art', 'Clipart', 'Product', 'Real World']
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_OfficeHome.pt')
elif dataset_name == "VLCS":
    train_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    try:
        dataset = load_dataset("ai22mtech12002/DG_VLCS")
    except:
        dataset = make_VLCS('data/VLCS')
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_VLCS_r16.pt')
elif dataset_name == "TI":
    try:
        dataset = load_dataset("ai22mtech12002/DG_TI")
    except:
        dataset = make_TI('data/terra_incognita')
    train_domains = ['location_38', 'location_43', 'location_46', 'location_100']
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_TI.pt')
elif dataset_name == "DomainNet-dil":
    args.num_tasks = 6
    args.data_path = '/data/ai22mtech12002/projects/WeightDG/data/DomainNet-dil'
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
    test_domain = int(args.test_domain)
    train_domains.remove(int(test_domain))
    is_hf_dataset = False
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet-dil.pt')

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
    

def make_hopfield(model):
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
        for name, module in model.named_modules():
            if isinstance(module, ViTSelfAttention):
                if not module.use_hopfield:
                    module.init_hopfield(0.5, hopfield_query_module) # , separation_function=F.softmax)
                    module.hopfield_query_generator.cuda()
                keys = torch.ones(len(train_domains), 768).cuda() + torch.randn(len(train_domains), 768).cuda() * 3e-4
                keys = keys / torch.norm(keys, dim=-1, keepdim=True)
                keys = [key for key in keys]
                for i, domain_name in enumerate(train_domains):
                    set_store_dict(model, adapters[domain_name])
                    keys[i].requires_grad = True
                    module.add_hopfield_element(keys[i])
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


classifier = nn.Linear(768, num_classes).cuda()
# schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess
        self.to_tensor = ToTensor()
       
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # print(list(item['image'].keys()))
        # item['image'].verify()
        if is_hf_dataset:
            if isinstance(item['image'], JpegImageFile) or isinstance(item['image'], PngImageFile):
                item['image'] = self.to_tensor(item['image'].convert("RGB"))
            item['image'] = self.preprocess(item['image'])
            return {'image': item['image'], 'label': item['label']}
        else:
            image, label, _ = item
            item = {}
            item['image'] = self.preprocess(image)
            item['label'] = label
            return item

criterion = nn.CrossEntropyLoss()

print(len(train_domains))

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

train_domain_adaters_list = []

classifier = nn.Linear(768, num_classes, bias=False).cuda()
keys, train_module_params = make_hopfield(model)
total_params = 0
for p in keys + train_module_params:
    total_params += p.numel()
print(f"Total params: {total_params}")
opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()), lr=args.lr, weight_decay=1e-2)
# opt = optim.AdamW(keys + train_module_params + [output_matrix], lr=args.lr, weight_decay=1e-2)

# Function to implement mixup, mixing different parts from all images in the batch
@torch.no_grad()
def mixup(image_batch, mixup_times=1):
    alpha = 0.4
    for i in range(mixup_times):
        lam = np.random.beta(alpha, alpha)
        rand_perm = torch.randperm(image_batch.size(0))
        image_batch = lam * image_batch + (1 - lam) * image_batch[rand_perm]
    return image_batch

# Function to implement cutmix, mixing different parts from all images in the batch
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


# def classifier(preds):
#     # preds = preds / torch.norm(preds, dim=-1, keepdim=True)
#     # ops = output_matrix / torch.norm(output_matrix, dim=-1, keepdim=True)
#     return (preds @ output_matrix)

# @torch.no_grad()
def eval():
    model.eval()
    pbar = Progbar(len(test_loader))
    for step, batch in enumerate(test_loader):
        pixel_values = batch['image'].cuda()
        labels = batch['label'].cuda()
        outputs = classifier(model(pixel_values))
        loss = criterion(outputs, labels)

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])



for epoch in range(args.epochs):
    model.train()
    running_acc = 0
    pbar = Progbar(len(train_loader))
    print(f"Epoch: {epoch}")
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
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
        outputs = classifier(feats)
        loss = criterion(outputs, labels) # + marginloss(outputs, labels, [feats])
        if torch.isnan(loss).any():
            print("NaN loss")
            continue
        loss.backward()
        opt.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
    eval()
    print()
    # schd.step()
    
