from datasets import load_dataset
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention
from transformers import TrainingArguments, Trainer
from open_clip_vit import VisionTransformer, MultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip
import argparse
import os
from localdatasets import make_VLCS, make_TI


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="OfficeHome")
parser.add_argument("--train_domain", type=str, default="art_painting")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

dataset_name = args.dataset
if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domain = args.train_domain
elif dataset_name == "DomainNet":
    dataset = load_dataset("wltjr1007/DomainNet")
    train_domain = int(args.train_domain)
elif dataset_name == "OfficeHome":
    dataset = load_dataset("flwrlabs/office-home")
    train_domain = args.train_domain
elif dataset_name == "VLCS":
    train_domain = args.train_domain
    try:
        dataset = load_dataset("ai22mtech12002/DG_VLCS")
    except:
        dataset = make_VLCS('data/VLCS')
elif dataset_name == "TI":
    try:
        dataset = load_dataset("ai22mtech12002/DG_TI")
    except:
        dataset = make_TI('data/terra_incognita')
    train_domain = args.train_domain



laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
vit = laion.visual.cuda()

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
                if "hopfield_keys" in n or "hopfield_values" in n:
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
                if "hopfield_keys" in n or "hopfield_values" in n:
                    p.data = weight_dict[i][:p.numel()].reshape(p.shape)
                    weight_dict[i] = weight_dict[i][p.numel():]

    def load_laion_weights(vit, laion_vit):
        vit_state_dict = vit.state_dict()
        laion_vit_state_dict = laion_vit.state_dict()
        for n, p in vit_state_dict.items():
            if n in laion_vit_state_dict:
                vit_state_dict[n] = laion_vit_state_dict[n]

        vit.load_state_dict(vit_state_dict)

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
        for name, module in model.named_modules():    
            if isinstance(module, MultiheadAttention):    
                for adapter_num in range(args.adapters_per_domain):
                    if not module.use_hopfield:
                        module.init_hopfield(0.5, hopfield_query_module)
                    key = torch.randn(768).cuda()
                    key.requires_grad = True
                    module.reset_lora_parameters()
                    module.add_hopfield_element(key)
                train_module_params += module.prepare_hopfield_params(as_list=True)
        return train_module_params

elif args.base_model == 'vit-in21k':
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


model = VisionTransformer(
    224, 16, 768, 12, 12, 4
).cuda()
# classifier = nn.Linear(512, args.num_classes, bias=False).cuda()
load_laion_weights(model, vit)

adapters_per_domain = args.adapters_per_domain
epochs = args.epochs
batch_size = args.batch_size
num_classes = args.num_classes
parent_dir = f'data/{dataset_name}'
os.makedirs(parent_dir, exist_ok=True)

output_matrix = (torch.randn(512, num_classes) / num_classes).cuda()
output_matrix.requires_grad = True
train_module_params = make_hopfield(model)
total_params = 0
for p in train_module_params:
    if p.requires_grad:
        total_params += p.numel()
        print(p.shape)
print(f"Total params: {total_params}")

# store_dict = get_store_dict(model)
# opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()) + [output_matrix], lr=args.lr, weight_decay=1e-2)
opt = optim.AdamW(train_module_params + [output_matrix], lr=args.lr, weight_decay=1e-2)

schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {'image': self.preprocess(item['image']), 'label': item['label']}
    
full_dataset = dataset.filter(lambda x: x['domain'] == train_domain)
train_dataset = full_dataset['train']
train_dataset = DomainDataset(train_dataset, preprocess_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

test_dataset = full_dataset['test']
test_dataset = DomainDataset(test_dataset, preprocess_val)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

train_domain_adaters_list = []


# Function to implement mixup, mixing different parts from all images in the batch
def mixup(image_batch, mixup_times=1):
    alpha = 0.4
    for i in range(mixup_times):
        lam = np.random.beta(alpha, alpha)
        rand_perm = torch.randperm(image_batch.size(0))
        image_batch = lam * image_batch + (1 - lam) * image_batch[rand_perm]
    return image_batch

def classifier(preds):
    # preds = preds / torch.norm(preds, dim=-1, keepdim=True)
    # ops = output_matrix / torch.norm(output_matrix, dim=-1, keepdim=True)
    return (preds @ output_matrix)

@torch.no_grad()
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
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
        pixel_values = batch['image'].cuda()
        labels = batch['label'].cuda()
        
        mixed_batch = []
        new_labels = []
        hopfield_masks = []
        for l in np.unique(labels.cpu().numpy()):
            idx = labels == l
            mixed_batch.append(mixup(pixel_values[idx], 0))
            new_labels.append(labels[idx])
            # hopfield_mask = torch.ones(labels[idx].shape[0], len(train_domains)*adapters_per_domain).cuda()
            # hopfield_mask[:, l::len(train_domains)] = -torch.inf
            # hopfield_masks.append(hopfield_mask.cuda())
        pixel_values = torch.cat(mixed_batch, dim=0)
        new_labels = torch.cat(new_labels, dim=0)
        # hopfield_masks = torch.cat(hopfield_masks, dim=0)
        hopfield_masks = 1
        labels = new_labels
        outputs = classifier(model(pixel_values, hopfield_masks=hopfield_masks))
        loss = criterion(outputs, labels)
        key_similarities = 0
        for name, module in model.named_modules():    
            if isinstance(module, MultiheadAttention):    
                key_similarities += module.get_key_similarities().abs()
        # loss += 0.001 * key_similarities
        loss.backward()
        opt.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc
        if step % 20 == 0:
            print(f"Step: {step}, Running acc: {running_acc / (step+1)}, Key Similarities: {key_similarities.item() / 12}")
        if step % args.infer_after == 0 and step > 0:
            eval()
            model.train()
    eval()
    schd.step()
    
weights = get_store_dict(model)
os.makedirs(f'weights/{dataset_name}_{train_domain}', exist_ok=True)
