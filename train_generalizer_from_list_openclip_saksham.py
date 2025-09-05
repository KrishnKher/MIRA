from datasets import load_dataset, Dataset, DatasetDict
from transformers import ViTFeatureExtractor, AutoModel
from modeling_vit import ViTForImageClassification, ViTSelfAttention
from transformers import TrainingArguments, Trainer
from open_clip_vit import VisionTransformer, MultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR, DN4ILDataset, SplitDN4IL
from torch.utils.data import Subset

from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip
import argparse
import os
from localdatasets import make_VLCS, make_TI
import json


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="DN4IL")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument("--test_domain", type=str)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None, help='Random seed (default: None)')
args = parser.parse_args()

def project_onto_affine_hull(V, x, rcond=1e-10):
    V = np.asarray(V, dtype=float)
    x = np.asarray(x, dtype=float)
    n, d = V.shape
    if n == 0:
        raise ValueError("V must contain at least one vector.")
    if d != x.shape[-1]:
        raise ValueError("Dimension mismatch between V and x.")

    v0 = V[0]
    if n == 1:
        return v0.copy(), np.array([1.0])

    B = (V[1:] - v0).T
    u = x - v0
    w, *_ = np.linalg.lstsq(B, u, rcond=rcond)
    p = v0 + B @ w

    coeffs = np.empty(n)
    coeffs[1:] = w
    coeffs[0]  = 1.0 - w.sum()
    return p, coeffs

dataset_name = args.dataset
is_hf_dataset = True
if dataset_name == "PACS":
    dataset = load_dataset("flwrlabs/pacs")
    train_domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
elif dataset_name == "DomainNet":
    dataset = load_dataset("wltjr1007/DomainNet")
    train_domains = [0, 1, 2, 3, 4, 5]
    train_domains.remove(int(args.test_domain))
    test_domain = int(args.test_domain)
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DomainNet.pt')


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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_VLCS.pt')
elif dataset_name == "TI":
    try:
        dataset = load_dataset("ai22mtech12002/DG_TI")
    except:
        dataset = make_TI('data/terra_incognita')
    train_domains = ['location_38', 'location_43', 'location_46', 'location_100']
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_TI.pt')

elif dataset_name == "DN4IL":
    train_domains = ["real", "clipart", "infograph", "painting", "quickdraw", "sketch"] 
    try:
        dataset = load_dataset("ai22mtech12002/DN4IL")
    except:
        dataset = make_VLCS('data/DN4IL')
    train_domains.remove(args.test_domain)
    test_domain = args.test_domain
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DN4IL.pt')

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





    # adapter_list = torch.load(
    #     '/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_DN4IL.pt'
    # )
    
    

laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
vit = laion.visual.cuda()

# if args.dataset == "DN4IL":
#     # fixed_order = list(range(6))
#     # dataset = load_dataset('data/DN4IL')
#     # if args.seed is not None:
#     #     np.random.seed(args.seed)
#     #     fixed_order = np.random.permutation(fixed_order)

#     # num_classes = 100
#     # num_tasks   = 6
#     # train_domains = list(range(num_tasks))
#     # assert num_tasks == len(DN4ILDataset.domains), "Expect 6 DN4IL domains"

#     # # benchmark = SplitDN4IL(
#     # #     n_experiences       = num_tasks,
#     # #     return_task_id      = True,
#     # #     seed                = args.seed,
#     # #     fixed_domain_order  = fixed_order,
#     # #     shuffle             = True,
#     # #     train_transform     = preprocess_train,
#     # #     eval_transform      = preprocess_val,
#     # #     dataset_root        = '/data/ai22mtech12002/projects/WeightDG/data/DN4IL'
#     # # )


#     # is_hf_dataset = False
    

    
#     dn4il = DN4ILDataset(
#         root='/data/ai22mtech12002/projects/WeightDG/data/DN4IL',
#         transform=None  
#     )
    
    

#     # print(dn4il)
   
#     images, labels, domains = [], [], []
#     print('DN4IL length', len(dn4il))
#     for ex in dn4il:
#         # print(ex[0], ex[1], ex[2])
#         # print(ex)
#         # break
#         images.append(ex[0])
#         labels.append(ex[1])
#         domains.append(ex[2])
        
#         if len(images) > 100:
#             break
#         # if ex[2] not in domains:
#         #     domains.append(ex[2])
#         # print(type(ex))
#         # images.append[ex[0]]
#         # labels.append[ex[1]]
#         # domains.append(ex[2])
        
#     # print()
#     print(len(domains), len(images), len(labels))
        
#         # print(ex)
    
        

#     # Build a HF Dataset from these lists
#     hf = Dataset.from_dict({
#         'image':  images,
#         'label':  labels,
#         'domain': domains
#     })
#     # Wrap into a single‐split DatasetDict so printout shows all columns
#     dataset = DatasetDict({'train': hf})

#     # Now create the Avalanche continual‐learning benchmark as before
#     fixed_order = list(range(6))
#     if args.seed is not None:
#         np.random.seed(args.seed)
#         fixed_order = np.random.permutation(fixed_order)
#     num_tasks = 6
#     train_domains = list(range(num_tasks))
#     domain_names = ['sketch', 'clipart', 'infograph', 'painting', 'real', 'quickdraw']
#     # test_domains = args.test_domain
#     test_domains = ['infograph']
#     train_names = [n for n in domain_names if n not in test_domains]
#     train_domains = train_names
#     print('Train Names', train_names)
    

    

#     benchmark = SplitDN4IL(
#         n_experiences      = num_tasks,
#         return_task_id     = True,
#         seed               = args.seed,
#         fixed_domain_order = fixed_order,
#         shuffle            = True,
#         train_transform    = preprocess_train,
#         eval_transform     = preprocess_val,
#         dataset_root       = '/data/ai22mtech12002/projects/WeightDG/data/DN4IL'
#     )
#     is_hf_dataset = False


# if args.dataset == "DN4IL":
#     # fixed_order = list(range(6))
#     # if args.seed is not None:
#     #     np.random.seed(args.seed)
#     #     fixed_order = np.random.permutation(fixed_order)
    
#     # num_tasks = 6
#     # domain_names = ['sketch', 'clipart', 'infograph', 'painting', 'real', 'quickdraw']
#     # test_domains = args.test_domain
    
#     # train_domains = [n for n in domain_names if n not in test_domains]
    
    
#     # benchmark = SplitDN4IL(
#     #     n_experiences      = num_tasks,
#     #     return_task_id     = True,
#     #     seed               = args.seed,
#     #     fixed_domain_order = fixed_order,
#     #     shuffle            = True,
#     #     train_transform    = preprocess_train,
#     #     eval_transform     = preprocess_val,
#     #     dataset_root       = '/data/ai22mtech12002/projects/WeightDG/data/DN4IL'
#     # )
#     is_hf_dataset = False

# class DomainDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, preprocess):
#         self.dataset = dataset
#         self.preprocess = preprocess

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         return {'image': self.preprocess(item['image']), 'label': item['label']}
    
# # print(len(train_domains))
# # print('Dataset: properties', dataset)

# # print('Dataset: properties', dataset['train'])
# print(train_domains)


# train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)
# # print(len(train_dataset))
# try:
#     train_dataset = train_dataset['train']
# except KeyError:
#     pass    
# train_dataset = DomainDataset(train_dataset, preprocess_train)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

# print("Length", len(train_loader))

# #iterating over the dataloader
# for i, batch in enumerate(train_loader):
#     print('image', batch['image'])
#     print('label', batch['label'])
#     break

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
        print(len(adapter_list))
        temps = np.linspace(0.001, 1, 12)
        for adapters in adapter_list:
            for name, module in model.named_modules():
                if isinstance(module, MultiheadAttention):
                    if not module.use_hopfield:
                        # module.init_hopfield()
                        module.init_hopfield(0.5, hopfield_query_module)
                        temps = temps[1:]
                        module.hopfield_query_generator.cuda()
                        # train_module_params += list(module.hopfield_query_generator.parameters())
                    # keys = torch.randn(len(train_domains), 768).cuda()
                    keys = torch.ones(len(train_domains), 768).cuda() + torch.randn(len(train_domains), 768).cuda() * 1e-6
                    # keys = torch.randn(len(train_domains), 32).cuda()
                    keys = [key for key in keys]
                    for i, domain_name in enumerate(train_domains):
                        set_store_dict(model, adapters[domain_name])
                        keys[i].requires_grad = True
                        module.add_hopfield_element(keys[i])
                    all_keys += keys
        return all_keys, train_module_params

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

output_matrix = torch.randn(512, num_classes).cuda()
output_matrix.requires_grad = True
classifier = nn.Linear(512, num_classes, bias=False).cuda()
keys, train_module_params = make_hopfield(model)
total_params = 0
for p in keys + train_module_params:
    total_params += p.numel()
print(f"Total params: {total_params}")
opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()) + [output_matrix], lr=args.lr, weight_decay=1e-2)
# opt = optim.AdamW(keys + train_module_params + [output_matrix], lr=args.lr, weight_decay=1e-2)

# schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
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
    
print(len(train_domains))
train_dataset = dataset.filter(lambda x: x['domain'] in train_domains)
try:
    train_dataset = train_dataset['train']
except KeyError:
    pass    
train_dataset = DomainDataset(train_dataset, preprocess_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

test_dataset = dataset.filter(lambda x: x['domain'] == test_domain)
try:
    test_dataset = test_dataset['train']
except KeyError:
    pass
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

# def classifier(preds):
#     # preds = preds / torch.norm(preds, dim=-1, keepdim=True)
#     # ops = output_matrix / torch.norm(output_matrix, dim=-1, keepdim=True)
#     return (preds @ output_matrix)


@torch.no_grad()
def eval():
    model.eval()
    device = next(model.parameters()).device
    base_adapter = adapter_list[0]
    # with open('adapter_list_check.json', 'w') as f:
    #     json.dump(adapter_list, f)
    proj_dict = {}
    print(adapter_list)
    for i in range(12):
        # print("Length of adapter list: ", len(adapter_list))
        # print(adapter_list)
    
        V = [ base_adapter[d][i].cpu().numpy() for d in train_domains ]
        x = base_adapter[test_domain][i].cpu().numpy()
        p, _ = project_onto_affine_hull(V, x)
        proj_dict[i] = torch.from_numpy(p.astype(np.float32)).to(device)
    set_store_dict(model, proj_dict)
    
    for m in model.modules():
        if isinstance(m, MultiheadAttention):
            m.use_hopfield = False
            
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
            mixed_batch.append(mixup(pixel_values[idx], 1))
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

        loss.backward()
        opt.step()
        acc = (outputs.argmax(dim=1) == labels).float().mean().item()
        # pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        running_acc += acc
        if step % 20 == 0:
            print(f"Step: {step}, Running acc: {running_acc / (step+1)}, Loss: {loss.item()}")
        if step % args.infer_after == 0 and step > 0:
            eval()
            model.train()
    eval()
    # schd.step()
    
