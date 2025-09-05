from localdatasets import load_dataset
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip


# Load PACS dataset from Hugging Face
dataset = load_dataset("flwrlabs/pacs")
classifier = nn.Linear(768, 7, bias=False).cuda()


@torch.no_grad()
def set_store_dict(model, weight_dict):
    weight_dict = deepcopy(weight_dict)
    for i in range(12):
        module = model.vit.encoder.layer[i]
        for n, p in module.named_parameters():
            if "lora" in n:
                p.data = weight_dict[i][:p.numel()].reshape(p.shape)
                weight_dict[i] = weight_dict[i][p.numel():]
    model.classifier.weight.data = weight_dict['classifier_weight'].reshape(model.classifier.weight.shape)
    model.classifier.bias.data = weight_dict['classifier_bias'].reshape(model.classifier.bias.shape)
    # model.classifier.weight.data = classifier.weight.data
    # model.classifier.bias.data = classifier.bias.data


def make_hopfield(model):
    train_module_params = []
    all_keys = []
    model.init_hopfield()
    model.hopfield_query_generator.cuda()
    train_module_params += list(model.hopfield_query_generator.parameters())
    for adapters in adapter_list:
        keys = torch.randn(3, 768).cuda()
        keys = [key for key in keys]
        for i, domain_name in enumerate(train_domains):
            set_store_dict(model, adapters[domain_name])
            keys[i].requires_grad = True
            model.add_hopfield_element(keys[i])
        all_keys += keys

        for name, module in model.named_modules():
            if isinstance(module, ViTSelfAttention):
                if not module.use_hopfield:
                    module.init_hopfield()
                    module.hopfield_query_generator.cuda()
                    train_module_params += list(module.hopfield_query_generator.parameters())
                keys = torch.randn(3, 768).cuda()
                # keys = nn.init.kaiming_normal_(keys)
                keys = [key for key in keys]
                for i, domain_name in enumerate(train_domains):
                    set_store_dict(model, adapters[domain_name])
                    keys[i].requires_grad = True
                    module.add_hopfield_element(keys[i])
                all_keys += keys
    return all_keys, train_module_params

adapter_list = torch.load('train_domain_adapters_list.pt')[:10]
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=7).cuda()
model.eval()


# Example access: domains are 'art_painting', 'cartoon', 'photo', 'sketch'
# train_domains = ['cartoon', 'photo', 'sketch']
# test_domain = 'art_painting'
# train_domains = ['art_painting', 'photo', 'sketch']
# test_domain = 'cartoon'
# train_domains = ['art_painting', 'cartoon', 'sketch']
# test_domain = 'photo'
train_domains = ['art_painting', 'cartoon', 'photo']
test_domain = 'sketch'
train_domain_adapters = torch.load('train_domain_adapters.pt')

keys, train_module_params = make_hopfield(model)
total_params = 0
for p in keys + train_module_params:
    total_params += p.numel()
print(f"Total params: {total_params}")
opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()), lr=1e-3)

criterion = nn.CrossEntropyLoss()

# for did, domain_name in enumerate(train_domains):
train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)['train']
train_dataset = train_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = dataset.filter(lambda x: x['domain'] == test_domain)['train']
test_dataset = test_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
test_dataset.set_format(type='torch', columns=['pixel_values', 'label'])
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

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
    pbar = Progbar(len(test_loader))
    for step, batch in enumerate(test_loader):
        pixel_values = batch['pixel_values'].cuda()
        labels = batch['label'].cuda()
        outputs = classifier(model(pixel_values).sequence_output)
        loss = criterion(outputs, labels)

        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])

for epoch in range(10):
    model.train()
    # pbar = Progbar(len(train_loader))
    running_acc = 0
    for step, batch in enumerate(train_loader):
        opt.zero_grad()
        pixel_values = batch['pixel_values'].cuda()
        labels = batch['label'].cuda()
        # outputs = classifier(model(pixel_values).sequence_output)
        # loss = criterion(outputs, labels)
        
        mixed_batch = []
        new_labels = []
        for l in np.unique(labels.cpu().numpy()):
            idx = labels == l
            mixed_batch.append(mixup(pixel_values[idx]))
            new_labels.append(labels[idx])
        pixel_values = torch.cat(mixed_batch, dim=0)
        new_labels = torch.cat(new_labels, dim=0)
        labels = new_labels
        outputs = classifier(model(pixel_values).sequence_output)
        loss = criterion(outputs, labels)

        loss.backward()
        opt.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc
        if step % 5 == 0:
            print(f"Step: {step}, Running acc: {running_acc / (step+1)}")
        if step % 20 == 0 and step > 0:
            eval()
            model.train()
    eval()
    # eval()

    
