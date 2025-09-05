from localdatasets import load_dataset
from transformers import ViTFeatureExtractor, ViTForImageClassification
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
test_domain = 'sketch'

train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)['train']
test_dataset = dataset.filter(lambda x: x['domain'] == test_domain)['train']

# Load the ViT model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

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


config =  LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
)

train_domain_adaters = {}
classifier = None

domain_models = {}

for did, domain_name in enumerate(train_domains):
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=7).cuda()
    model = get_peft_model(model, peft_config=config).cuda()

    train_dataset = dataset.filter(lambda x: x['domain'] == domain_name)['train']
    train_dataset = train_dataset.map(lambda x: feature_extractor(x['image']), batched=True)
    train_dataset.set_format(type='torch', columns=['pixel_values', 'label'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_trainable = 0
    num_total = 0
    trainable_params = []
    for name, param in model.named_parameters():
        num_total += param.numel()
        if "classifier" in name or "lora" in name:
            param.requires_grad = True
            trainable_params.append(param)
            num_trainable += param.numel()
        if "classifier" in name:
            print(name + ":", "Training classifier" if param.requires_grad else "Freezing classifier")        

    print(f"Trainable parameters: {num_trainable} / {num_total}")

    # Train the model
    optimizer = optim.AdamW(trainable_params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        pbar = Progbar(len(train_loader))
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            pixel_values = batch['pixel_values'].cuda()
            labels = batch['label'].cuda()
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
    
    new_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=7).cuda()
    new_model = get_peft_model(new_model, peft_config=config).cuda()
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
        outputs = new_model(pixel_values)
        loss = criterion(outputs.logits, labels)

        acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])


model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=7).cuda()
model = get_peft_model(model, peft_config=config).cuda()
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
        outputs = model(pixel_values)
        loss = criterion(outputs.logits, labels)

        acc = (outputs.logits.argmax(dim=1) == labels).float().mean().item()
        pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])

torch.save(train_domain_adaters, "train_domain_adapters.pt")