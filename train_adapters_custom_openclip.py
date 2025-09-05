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

from localdatasets import make_VLCS, make_TI, make_DN4IL
from vil_datasets import build_continual_dataloader

from progbar import Progbar
from copy import deepcopy
import open_clip
import os
import numpy as np
from PIL import Image
import argparse

import open_clip


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
    laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
    vit = laion.visual.cuda()


    @torch.no_grad()
    def get_store_dict(model):
        module_weight_dict = {}
        for i in range(12):
            module = model.transformer.resblocks[i]
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
            module = model.transformer.resblocks[i]
            for n, p in module.named_parameters():
                if "lora" in n:
                    p.data = weight_dict[i][:p.numel()].reshape(p.shape)
                    weight_dict[i] = weight_dict[i][p.numel():]

    def load_laion_weights(vit, laion_vit):
        vit_state_dict = vit.state_dict()
        laion_vit_state_dict = laion_vit.state_dict()
        for n, p in vit_state_dict.items():
            if n in laion_vit_state_dict:
                vit_state_dict[n] = laion_vit_state_dict[n]

        vit.load_state_dict(vit_state_dict)
        

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


train_domain_adaters_list = []
classifiers = {}
classifier = None

for col in range(adapters_per_domain):
    train_domain_adaters = {}
    domain_models = {}
    train_domain_adaters_list.append(train_domain_adaters)

    # generate a dictionary mapping 0-6 -> 0-6 in a random permutation
    perm = torch.randperm(num_classes)
    perm_dict = {i: perm[i].item() for i in range(num_classes)}

    if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
        args.label_offset = 0

    for did, domain_name in enumerate(train_domains):
        if args.base_model == 'laion':
            model = VisionTransformer(
                224, 16, 768, 12, 12, 4
            ).cuda()
            

            classifier = nn.Linear(512, num_classes, bias=False).cuda()
            load_laion_weights(model, vit)
            
            # print("Printing model structure :", model)
        elif args.base_model == 'vit-in21k':
            model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
            classifier = nn.Linear(768, num_classes, bias=False).cuda()
            model.classifier = classifier
            load_vit_weights(model, vit)
            
        
        train_loader = train_loaders[domain_name]

        num_trainable = 0
        num_total = 0
        trainable_params = [classifier.weight]
        # trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        # trainable_params.append(classifier.weight)
        # if args.base_model == 'laion':
        #     trainable_params.extend(list(classifier.parameters()))
            
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

        print(f"Trainable parameters: {num_trainable} / {num_total}")
        print(model)
        # Train the model
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            pbar = Progbar(len(train_loader))
            acc_list = []
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pixel_values = batch['image'].cuda()
                labels = batch['label'].cuda()
                if step == 0:
                    print("Labels: ", labels.min(), labels.max())
                outputs = classifier(model(pixel_values))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                acc_list.append(acc)
                pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
            if sum(acc_list)/len(acc_list) > 0.99:
                break

        print()
        if args.base_model == 'laion':
            new_model = VisionTransformer(
                224, 16, 768, 12, 12, 4
            ).cuda()
            load_laion_weights(new_model, vit)
            
        elif args.base_model == 'vit-in21k':
            new_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k').cuda()
            new_model.classifier = deepcopy(classifier)
            load_vit_weights(new_model, vit)
        

        train_domain_adaters[domain_name] = get_store_dict(model)
        # train_domain_adaters[domain_name] = get_peft_lora_weights(model)
        set_store_dict(new_model, train_domain_adaters[domain_name])
        # set_peft_lora_weights(new_model, train_domain_adaters[domain_name])
        domain_models[domain_name] = deepcopy(model)
        new_model.eval()

        for n, m in model.state_dict().items():
            if not new_model.state_dict()[n].equal(m):
                print()
                if "classifier" in n:
                    print("Classifier not equal")
                else:
                    print(n)
                print()

        pbar = Progbar(len(train_loader))
        for step, batch in enumerate(train_loader):
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda()
            outputs = classifier(new_model(pixel_values))
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])

        classifiers[domain_name] = deepcopy(classifier)

        if args.dataset in ['iDigits-cil', 'CORe50-cil', 'DomainNet-cil']:
            args.label_offset += args.num_classes

print("Saving weights to ", f"weights/train_domain_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(train_domain_adaters_list, f"weights/train_domain_adapters_list_{args.base_model}_{dataset_name}{args.name_tag}.pt")
torch.save(classifiers, f"weights/train_domain_classifiers_{args.base_model}_{dataset_name}{args.name_tag}.pt")