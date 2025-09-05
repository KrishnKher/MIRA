from localdatasets import load_dataset
from transformers import ViTFeatureExtractor
from modeling_vit import ViTForImageClassification
from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model

from progbar import Progbar
from copy import deepcopy

# Load PACS dataset from Hugging Face
dataset = load_dataset("flwrlabs/pacs")

# Example access: domains are 'art_painting', 'cartoon', 'photo', 'sketch'
train_domains = ['sketch', 'art_painting', 'cartoon', 'photo']

train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)['train']

# Load the ViT model
model_string = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_string)

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
    module_weight_dict['classifier_weight'] = model.classifier.weight.reshape(-1).detach()
    module_weight_dict['classifier_bias'] = model.classifier.bias.reshape(-1).detach()
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
    model.classifier.weight.data = weight_dict['classifier_weight'].reshape(model.classifier.weight.shape)
    model.classifier.bias.data = weight_dict['classifier_bias'].reshape(model.classifier.bias.shape)

train_domain_adaters_list = []
classifier = None

domain_models_list = []

for col in range(20):
    train_domain_adaters = {}
    domain_models = {}
    train_domain_adaters_list.append(train_domain_adaters)
    domain_models_list.append(domain_models)

    # generate a dictionary mapping 0-6 -> 0-6 in a random permutation
    perm = torch.randperm(7)
    perm_dict = {i: perm[i].item() for i in range(7)}

    for did, domain_name in enumerate(train_domains):
        model = ViTForImageClassification.from_pretrained(model_string, num_labels=7).cuda()

        train_dataset = dataset.filter(lambda x: x['domain'] == domain_name)['train']
        train_dataset = train_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
        train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        num_trainable = 0
        num_total = 0
        trainable_params = []
        for name, param in model.named_parameters():
            num_total += param.numel()
            if "lora" in name or "classifier" in name:
                param.requires_grad = True
                num_trainable += param.numel()
                trainable_params.append(param)
            if "classifier" in name:
                print(name + ":", "Training classifier" if param.requires_grad else "Freezing classifier")        

        print(f"Trainable parameters: {num_trainable} / {num_total}")

        # Train the model
        optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(2):
            pbar = Progbar(len(train_loader))
            acc_list = []
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                pixel_values = batch['pixel_values'].cuda()
                labels = batch['label'].cuda()
                perm_labels = torch.tensor([perm_dict[l.item()] for l in labels]).cuda()
                labels = perm_labels
                outputs = model(pixel_values)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
                acc_list.append(acc)
                pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
            if sum(acc_list)/len(acc_list) > 0.99:
                break
        new_model = ViTForImageClassification.from_pretrained(model_string, num_labels=7).cuda()
        train_domain_adaters[domain_name] = get_store_dict(model)
        set_store_dict(new_model, train_domain_adaters[domain_name])
        domain_models[domain_name] = deepcopy(model)

        for n, m in model.state_dict().items():
            if not new_model.state_dict()[n].equal(m):
                print(n)
                exit()

        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['label'].cuda()
            perm_labels = torch.tensor([perm_dict[l.item()] for l in labels]).cuda()
            labels = perm_labels
            outputs = new_model(pixel_values)
            loss = criterion(outputs.logits, labels)

            acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])


    model = ViTForImageClassification.from_pretrained(model_string, num_labels=7).cuda()
    for did, domain_name in enumerate(train_domains):
        train_dataset = dataset.filter(lambda x: x['domain'] == domain_name)['train']
        train_dataset = train_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
        train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        set_store_dict(model, train_domain_adaters[domain_name])

        domain_model = domain_models[domain_name]
        for n, m in model.state_dict().items():
            if not domain_model.state_dict()[n].equal(m):
                print(n)
                exit()

        pbar = Progbar(len(train_loader))
        for step, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['label'].cuda()
            perm_labels = torch.tensor([perm_dict[l.item()] for l in labels]).cuda()
            labels = perm_labels
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)

            acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])

torch.save(train_domain_adaters_list, "train_domain_adapters_list.pt")