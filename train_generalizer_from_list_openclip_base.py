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


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DomainNet")
parser.add_argument("--adapters_per_domain", type=int, default=5)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=345)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument("--test_domain", type=str, default='5')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--separation_function', type=str, default='affine')
parser.add_argument('--include_seen_domain', action='store_true')
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
    
if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    test_domain = args.test_domain
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
    
    train_domains.remove(args.test_domain)
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
        
elif dataset_name == "VLCS":
    train_domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    try:
        dataset = load_dataset("ai22mtech12002/DG_VLCS")
    except:
        dataset = make_VLCS('data/VLCS')
    train_domains.remove(args.test_domain)
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
state_dict = torch.load(
    '/data/ai22mtech12002/projects/WeightDG/weights/classifier_without_adapters_vit-in21k_PACSbase_vit_pacs_stage1_0310.pt',
    map_location="cuda"
    # weights_only=False
)
# last_head = state_dict[5]
if args.base_model == 'laion':
   model.load_state_dict(torch.load('/data/ai22mtech12002/projects/WeightDG/weights/model_without_adapters_laion_DomainNet-dilstage_1_laion_withoutadapters_witheval.pt'))
elif args.base_model == 'vit-in21k':
    # model.load_state_dict(torch.load('/data/ai22mtech12002/projects/WeightDG/weights/model_without_adapters_vit-in21k_DomainNet-dilstage_1_vitin21k_withoutadapters_witheval.pt'), strict = False)
    # model.load_state_dict(torch.load('/data/ai22mtech12002/projects/WeightDG/weights/model_without_adapters_vit-in21k_PACSbase_vit_pacs_stage1_0310.pt'), strict = False)
    # model.load_state_dict(torch.load('weights/model_without_adapters_vit-in21k_OfficeHome.pt'), strict = False)
    model.load_state_dict(torch.load('weights/model_without_adapters_vit-in21k_VLCSvlcs_base_stage1_0410.pt'), strict = False) 
model = model.cuda()
if args.base_model == 'laion':
    # replace_attention_with_custom(model)
    classifier = nn.Linear(512, num_classes, bias=False).cuda()
    # classifier.load_state_dict(state_dict)

elif args.base_model == 'vit-in21k':
    classifier = nn.Linear(768, num_classes, bias=False).cuda()
    # classifier.load_state_dict(state_dict)
    # model.classifier = classifier
    
    model.classifier = classifier

    # model.classifier = model.model.classifier = nn.Identity()





criterion = nn.CrossEntropyLoss()
if args.base_model == 'laion':
    opt = optim.AdamW(list(classifier.parameters()), lr=args.lr, weight_decay=1e-2)
else:
    trainable_params = []
    for name,param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True
            trainable_params.append(param)
    opt = optim.AdamW(trainable_params, lr = args.lr, weight_decay=1e-2)
schd = optim.lr_scheduler.StepLR(opt, 10, 0.5)

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
        else:
            # out = model(pixel_values)
            # outputs = classifier(out.logits)
            
            outputs = model(pixel_values).logits
        loss = criterion(outputs, labels)

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        

marginloss = LargeMarginLoss()
for epoch in range(args.epochs):
    model.train()
    running_acc = 0
    pbar = Progbar(len(train_loader))
    print(f"Epoch: {epoch}")
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
        # cls_opt.zero_grad()
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
        else:
            # outputs = classifier(feats.logits)
            outputs = feats.logits
        loss = criterion(outputs, labels) # + marginloss(outputs, labels, [feats])
        # if torch.isnan(loss).any():
        #     print("NaN loss")
        #     continue
        loss.backward()
        # mod_keys = get_model_keys(model)
        # for i, key in enumerate(mod_keys):
        #     if key.grad is None or torch.isnan(key.grad).any():
        #         print(f"Key {i} has NaN grad")
        #         opt.zero_grad()
        #         cls_opt.zero_grad()
        #         continue
        opt.step()
        # cls_opt.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
    eval()
    print()