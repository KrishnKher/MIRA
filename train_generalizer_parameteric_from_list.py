from localdatasets import load_dataset
from transformers import ViTFeatureExtractor
from modeling_vit_extmem import ViTForImageClassification, ViTSelfAttention
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from models import SimSelModel

from progbar import Progbar
from copy import deepcopy

# Load PACS dataset from Hugging Face
dataset = load_dataset("flwrlabs/pacs")

classifier = nn.Linear(768, 7).cuda()

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
    # module_weight_dict['classifier_weight'] = model.classifier.weight.reshape(-1).detach()
    # module_weight_dict['classifier_bias'] = model.classifier.bias.reshape(-1).detach()
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
    # model.classifier.weight.data = weight_dict['classifier_weight'].reshape(model.classifier.weight.shape)
    # model.classifier.bias.data = weight_dict['classifier_bias'].reshape(model.classifier.bias.shape)
    model.classifier.weight.data = classifier.weight.data
    model.classifier.bias.data = classifier.bias.data


simselmodel_adapters = SimSelModel(768, 49152).cuda()
simselmodel_cls = SimSelModel(768, 768*7 + 7).cuda()

def make_hopfield(model):
    model.init_hopfield(simselmodel_cls)
    # model.hopfield_query_generator.cuda()
    for adapters in adapter_list[:3]:
        for i, domain_name in enumerate(train_domains):
            set_store_dict(model, adapters[domain_name])
            model.add_hopfield_element()

        for name, module in model.named_modules():
            if isinstance(module, ViTSelfAttention):
                if not module.use_hopfield:
                    module.init_hopfield(simselmodel_adapters)
                    # module.hopfield_query_generator.cuda()
                for i, domain_name in enumerate(train_domains):
                    set_store_dict(model, adapters[domain_name])
                    module.add_hopfield_element()
        
adapter_list = torch.load('train_domain_adapters_list.pt')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=7).cuda()
model.eval()


# Example access: domains are 'art_painting', 'cartoon', 'photo', 'sketch'
# train_domains = ['art_painting', 'cartoon', 'photo']
# test_domain = 'sketch'
train_domains = ['cartoon', 'photo', 'sketch']
test_domain = 'art_painting'

make_hopfield(model)
from itertools import chain
opt = optim.AdamW(chain(simselmodel_adapters.parameters(), simselmodel_cls.parameters(), classifier.parameters()), lr=1e-3)

criterion = nn.CrossEntropyLoss()

# for did, domain_name in enumerate(train_domains):
train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)['train']
train_dataset = train_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = dataset.filter(lambda x: x['domain'] == test_domain)['train']
test_dataset = test_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
test_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Function to implement mixup
def mixup(image_batch):
    alpha = 0.4
    lam = np.random.beta(alpha, alpha)
    rand_perm = torch.randperm(image_batch.size(0))
    mixed_image_batch = lam * image_batch + (1 - lam) * image_batch[rand_perm]
    return mixed_image_batch, lam, rand_perm
    
@ torch.no_grad()
def eval():
    pbar = Progbar(len(test_loader))
    for step, batch in enumerate(test_loader):
        pixel_values = batch['pixel_values'].cuda()
        labels = batch['label'].cuda()
        outputs = model(pixel_values)
        loss = criterion(outputs.logits, labels)

        acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])

for epoch in range(10):
    model.train()
    running_acc = 0
    # pbar = Progbar(len(train_loader))
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
        pixel_values = batch['pixel_values'].cuda()
        labels = batch['label'].cuda()
        
        mixed_batch = []
        new_labels = []
        for l in np.unique(labels.cpu().numpy()):
            idx = labels == l
            mixup_images = pixel_values[idx]
            for i in range(2):
                mixup_images = mixup(mixup_images)[0]
            mixed_batch.append(mixup(mixup_images)[0])
            new_labels.append(labels[idx])
        pixel_values = torch.cat(mixed_batch, dim=0)
        new_labels = torch.cat(new_labels, dim=0)
        labels = new_labels
        outputs = model(pixel_values)
        loss = criterion(outputs.logits, labels)

        loss.backward()
        opt.step()
        acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc

        if step % 5 == 0:
            print(f"Step: {step}, Running acc: {running_acc / (step+1)}")
            
        if step % 20 == 0 and step > 0:
            eval()
    eval()
    # eval()
