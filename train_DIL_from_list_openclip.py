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
from torchvision.transforms import Compose, Resize, RandomCrop, RandomHorizontalFlip, Normalize

from progbar import Progbar
from copy import deepcopy
import numpy as np
import open_clip
import argparse
import os
from localdatasets import make_VLCS, make_TI
from itertools import chain
from avalanche.benchmarks.classic import SplitCIFAR100, SplitTinyImageNet, SplitCUB200, SplitImageNet
from custom_datasets import SplitImageNetR

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="OfficeHome")
parser.add_argument("--adapters_per_domain", type=int, default=10)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--later_epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--num_classes", type=int, default=65)
parser.add_argument("--base_model", type=str, default='laion', choices=['laion', 'vit-in21k'])
parser.add_argument("--name_tag", type=str, default='')
parser.add_argument("--infer_after", type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

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


laion, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
vit = laion.visual.cuda()

if args.base_model == 'laion':
    class CustomAttention(nn.Module):
        def __init__(
            self,
            embed_dim,
            num_heads,
            bias=True,
            batch_first=False,
            in_proj_weight=None,
            in_proj_bias=None,
            out_proj=None,
            **kwargs
        ):
            super().__init__()
            # print("REPLACING###########################")
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
            self.batch_first = batch_first


            self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
            self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

            # assert False
            assert in_proj_weight is not None
            if in_proj_weight is not None:
                assert in_proj_weight.shape == (3 * embed_dim, embed_dim)
                q_w, k_w, v_w = in_proj_weight.chunk(3, dim=0)
                # print("CALEED########################")
                self.q_proj.weight.data = q_w
                self.k_proj.weight.data = k_w
                self.v_proj.weight.data = v_w

            if in_proj_bias is not None:
                assert in_proj_bias.shape == (3 * embed_dim,)
                q_b, k_b, v_b = in_proj_bias.chunk(3, dim=0)
                self.q_proj.bias.data = q_b
                self.k_proj.bias.data = k_b
                self.v_proj.bias.data = v_b

            self.out_proj = out_proj if out_proj is not None else nn.Linear(embed_dim, embed_dim, bias=bias)
            
            
            #Add bias in forward also
            #FIrst use replace layers, then use peft model
            
            #USe method of replace layers to convert MHA to Custom attention
            
            # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        def forward(
            self,
            query=None,
            key=None,
            value=None,
            # q_x=None,
            # k_x=None,
            # v_x=None,
            q_bias: Optional[Tensor] = None,
            k_bias: Optional[Tensor] = None,
            v_bias: Optional[Tensor] = None,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal: bool = False,
            **kwargs
        ):
            query = query if query is not None else kwargs.get("q_x", None)
            key = key if key is not None else kwargs.get("k_x", None)
            value = value if value is not None else kwargs.get("v_x", None)

            if query is None or key is None or value is None:
                raise ValueError("CustomAttention expects either (query, key, value) or (q_x, k_x, v_x) to be passed.")
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

            if q_bias is not None:
                q = q + q_bias
            if k_bias is not None:
                k = k + k_bias
            if v_bias is not None:
                v = v + v_bias

            scale = self.head_dim ** -0.5
            h = self.num_heads
            d = self.head_dim

            if self.batch_first:
                # [B, S, E]
                B, S, E = q.shape
                q = q.view(B, S, h, d).transpose(1, 2)  
                k = k.view(B, -1, h, d).transpose(1, 2) 
                v = v.view(B, -1, h, d).transpose(1, 2) 

                attn = (q @ k.transpose(-2, -1)) * scale     
                attn = attn.softmax(dim=-1)
                out  = (attn @ v)                              
                out  = out.transpose(1, 2).contiguous().view(B, S, E)
            else:
                S, B, E = q.shape
                q = q.view(S, B, h, d).permute(1, 2, 0, 3)     
                k = k.view(-1, B, h, d).permute(1, 2, 0, 3)   
                v = v.view(-1, B, h, d).permute(1, 2, 0, 3)   

                attn = (q @ k.transpose(-2, -1)) * scale       
                attn = attn.softmax(dim=-1)
                out  = (attn @ v)                              
                out  = out.permute(2, 0, 1, 3).contiguous().view(S, B, E)
            # print(out.shape)

            return self.out_proj(out), attn
        
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
        print(len(adapter_list))
        for adapters in adapter_list[:2]:
            adapters = adapters[domain]
            for name, module in model.named_modules():
                if isinstance(module, MultiheadAttention):
                    if not module.use_hopfield:
                        module.init_hopfield(0.5, hopfield_query_module)
                        module.hopfield_query_generator.cuda()
                    keys = torch.ones(1, 768).cuda() + torch.randn(1, 768).cuda() * 1e-6
                    # keys =  torch.randn(1, 768).cuda()
                    # keys = keys / torch.norm(keys, dim=-1, keepdim=True)
                    keys = [key for key in keys]
                    set_store_dict(model, adapters)
                    for k in keys:
                        k.requires_grad = True
                        module.add_hopfield_element(k)
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
# opt = optim.AdamW(keys + train_module_params + [output_matrix], lr=args.lr, weight_decay=1e-2)

# schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()


# class DomainDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, preprocess):
#         self.dataset = dataset
#         self.preprocess = preprocess

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         return {'image': self.preprocess(item['image']), 'label': item['label']}
class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess):
        self.dataset = dataset
        self.preprocess = preprocess
        if not is_hf_dataset:
            if args.base_model == 'vit-in21k':
                self.vit_preprocess = Compose([
                        Resize(256),
                        RandomCrop(224),
                        RandomHorizontalFlip(0.5),
                        Normalize(preprocess.image_mean, preprocess.image_std)
                    ])
            else:
                self.preprocess = Compose([
                        Resize(256),
                        RandomCrop(224),
                        RandomHorizontalFlip(0.5),
                        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
                    ])
            

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
            image, label, _ = item
            item = {}
            if args.base_model == 'vit-in21k':
                item['image'] = self.vit_preprocess(image)
            else:
                item['image'] = self.preprocess(image)
            item['label'] = label
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
    for test_loader in test_domain_loaders:
        pbar = Progbar(len(test_loader))
        for step, batch in enumerate(test_loader):
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda()
            outputs = classifier(model(pixel_values))
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])


test_domain_loaders = []
for domain_idx, domain in enumerate(train_domains):
    if is_hf_dataset:
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
    total_params = 0
    for p in keys + train_module_params:
        total_params += p.numel()
    print(f"Total params: {total_params}")
    opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()), lr=args.lr, weight_decay=1e-2)
    if domain_idx > 0:
        base_classifier = deepcopy(classifier)

    epochs = args.epochs if (args.later_epochs is None or domain_idx == 0) else args.later_epochs
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
            pixel_values = torch.cat(mixed_batch, dim=0)
            new_labels = torch.cat(new_labels, dim=0)
            hopfield_masks = 1
            labels = new_labels
            outputs = classifier(model(pixel_values, hopfield_masks=hopfield_masks))
            loss = criterion(outputs, labels)

            # penalize change in classifier weights
            if domain_idx > 0:
                loss += F.mse_loss(classifier.weight, base_classifier.weight.detach()) * 10

            loss.backward()
            opt.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            running_acc += acc
            print(f"Step: {step}, Running acc: {running_acc / (step+1)}, Loss: {loss.item()}")
        eval()
        