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
import math
import random

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


class ReplayBuffer:
    """
    A fixed-size replay buffer that maintains a balanced number of samples per class.
    """

    def __init__(self, buffer_size):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of samples the buffer can hold.
        """
        self.buffer_size = buffer_size
        self.buffer = []  # List of (sample, label) tuples

    def update(self, dataloader, label_offset=0):
        """
        Updates the buffer with samples from a given dataloader.

        1. Collects all new samples from the dataloader.
        2. Computes the set of all classes (existing + new).
        3. Determines the desired number of samples per class.
        4. Randomly discards excess samples for each class.
        5. Randomly adds new samples to fill each class quota.

        Args:
            dataloader: An iterable yielding (inputs, labels) batches.
                        Inputs should be tensors; labels should be scalar or tensor.
        """
        # 1. Collect new samples
        new_samples = []
        for batch in dataloader:
            inputs = batch['image']
            labels = batch['label']
            for x, y in zip(inputs, labels):
                label = y.item() if hasattr(y, 'item') else y
                new_samples.append((x, label + label_offset))

        # 2. Determine all classes
        existing_labels = set(label for _, label in self.buffer)
        new_labels = set(label for _, label in new_samples)
        all_labels = list(existing_labels.union(new_labels))
        n_classes = len(all_labels)
        if n_classes == 0:
            return  # Nothing to do if no samples

        # 3. Compute per-class quota
        base_quota = self.buffer_size // n_classes
        remainder = self.buffer_size % n_classes
        per_class_quota = {cls: base_quota for cls in all_labels}
        # Distribute the remainder randomly
        extra_classes = random.sample(all_labels, remainder)
        for cls in extra_classes:
            per_class_quota[cls] += 1

        # 4. Keep/discard existing samples to meet the new quotas
        updated_buffer = []
        for cls in all_labels:
            existing_cls = [s for s in self.buffer if s[1] == cls]
            if len(existing_cls) > per_class_quota[cls]:
                existing_cls = random.sample(existing_cls, per_class_quota[cls])
            updated_buffer.extend(existing_cls)

        # 5. Fill up from new samples where needed
        for cls in all_labels:
            current_count = sum(1 for _, lbl in updated_buffer if lbl == cls)
            needed = per_class_quota[cls] - current_count
            if needed > 0:
                candidates = [s for s in new_samples if s[1] == cls]
                if len(candidates) > needed:
                    candidates = random.sample(candidates, needed)
                updated_buffer.extend(candidates)

        # Replace the buffer
        self.buffer = updated_buffer

    def sample(self, batch_size):
        """
        Randomly samples a batch of samples from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            List of (sample, label) tuples.
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        """
        Returns:
            int: Current number of samples in the buffer.
        """
        return len(self.buffer)


class GrowingLinearClassifier(nn.Module):
    """
    A linear classifier whose output dimension (number of classes) can grow.
    When new classes are added, the old weights/biases are zero‐padded
    so that any attached DualGPM memory can also be padded correspondingly.
    """
    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim:   dimensionality of feature vectors coming in
            num_classes: initial number of output classes
        """
        super().__init__()
        self.in_features   = input_dim
        self.num_classes = num_classes

        # Initialize weight [C × D] and bias [C]
        self.weight = nn.Parameter(torch.empty(num_classes, input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Same scheme as nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, input_dim]
        returns logits: [batch_size, num_classes]
        """
        return F.linear(x, self.weight)

    def add_classes(self, k: int):
        """
        Expand the classifier to handle k additional classes.

        After this call:
          • self.num_classes increases by k
          • self.weight and self.bias gain k new rows of zeros at the bottom
        """
        if k <= 0:
            return

        C_old = self.num_classes
        C_new = C_old + k

        # Extract old values
        w_old = self.weight.data
        device, dtype = w_old.device, w_old.dtype

        # Make new, zero‐initialized weight & bias
        w_new = torch.zeros(C_new, self.in_features, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(w_new, a=math.sqrt(5))

        # Copy old parameters into the top rows
        w_new[:C_old] = w_old

        # Reassign
        self.num_classes = C_new
        self.weight      = nn.Parameter(w_new)


class HopfieldClassifier(nn.Module):
    """
    A simple classifier augmented with a Hopfield-style memory for swapping entire linear layer weights.
    """
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Base linear classifier
        self.classifier = nn.Linear(input_dim, num_classes)

        # Hopfield memory components
        self.use_hopfield = False
        self.hopfield_keys = None       # shape: (key_dim, num_memories)
        self.hopfield_values = None     # shape: (num_memories, value_dim)
        self.temp = 1.0
        self.separation_function = lambda sim, dim: F.softmax(sim, dim=dim)

    def init_hopfield(self, temp: float = 0.1, separation_function=None):
        """
        Enable Hopfield memory with a given temperature and optional separation function.
        """
        self.use_hopfield = True
        self.temp = temp
        if separation_function is not None:
            self.separation_function = separation_function
        self.hopfield_keys = None
        self.hopfield_values = None

    def add_hopfield_element(self, key: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None):
        """
        Add a memory slot storing full classifier weights and bias.

        Args:
            key: Tensor of shape (key_dim,) used to retrieve this memory.
            weight: Optional Tensor of shape (num_classes, input_dim). If None, uses current classifier.weight.
            bias:   Optional Tensor of shape (num_classes,). If None, uses current classifier.bias.
        """
        # Detach provided or current parameters
        weight = weight.detach() if weight is not None else self.classifier.weight.detach()

        # Flatten and concatenate weight and bias
        val = weight.reshape(-1).unsqueeze(0)  # (1, total_dim)

        # Ensure key has shape (key_dim, 1)
        key = key.unsqueeze(0) if key.dim() == 1 else key

        if self.hopfield_keys is None:
            self.hopfield_keys = key
            self.hopfield_values = val
        else:
            self.hopfield_keys = torch.cat([self.hopfield_keys, key], dim=0)
            self.hopfield_values = torch.cat([self.hopfield_values, val], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: either use base classifier or retrieve full weights from Hopfield memory.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        # x of shape B, d
        if self.use_hopfield and self.hopfield_keys is not None:
            query = x / x.norm(dim=-1, keepdim=True)

            # Normalize memory keys
            hopfield_keys = self.hopfield_keys / self.hopfield_keys.norm(dim=-1, keepdim=True)
            hopfield_keys = hopfield_keys / torch.norm(hopfield_keys, dim=0, keepdim=True)

            weight_sim = self.separation_function((query @ hopfield_keys.T) * self.temp) # b, num_memories

            # Retrieve weighted memory values
            mem = weight_sim @ self.hopfield_values  # (total_dim,)
            weight = mem.view(x.shape[0], self.num_classes, self.input_dim) # B, num_classes, input_dim
            logits = torch.einsum('bci,bi->bc', weight, x)  # B, num_classes
            # print(logits.shape)
            # exit()
        else:
            weight = self.classifier.weight
            logits = F.linear(x, weight)
        return logits


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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_tinyimagenet.pt')
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
    adapter_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_adapters_list_laion_inetR_5tasks.pt')
    classifier_list = torch.load('/data/ai22mtech12002/projects/WeightDG/weights/train_domain_classifiers_laion_inetR_5tasks.pt', weights_only=False)
    is_hf_dataset = False


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
        for adapters in adapter_list[:2]:
            adapters = adapters[domain]
            for name, module in model.named_modules():
                if isinstance(module, MultiheadAttention):
                    if not module.use_hopfield:
                        module.init_hopfield(10, hopfield_query_module, separation_function=F.softmax)
                        module.hopfield_query_generator.cuda()
                    keys = torch.ones(1, 768).cuda() + torch.randn(1, 768).cuda() * 1e-20
                    # keys =  torch.randn(1, 768).cuda()
                    keys = keys / torch.norm(keys, dim=-1, keepdim=True)
                    keys = [key for key in keys]
                    set_store_dict(model, adapters)
                    for k in keys:
                        k.requires_grad = True
                        module.add_hopfield_element(k)
                    all_keys += keys
        if not classifier.use_hopfield:
            classifier.init_hopfield(10, separation_function=F.softmax)
        for cls in classifier_list:
            keys = torch.ones(1, 512).cuda() + torch.randn(1, 512).cuda() * 1e-20
            keys = keys / torch.norm(keys, dim=-1, keepdim=True)
            cls = cls[domain]
            classifier.add_hopfield_element(keys, cls.weight)
            for k in keys:
                k.requires_grad = True
                all_keys += [k]
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


# classifier = nn.Linear(512, num_classes).cuda()
# classifier = GrowingLinearClassifier(512, num_classes).cuda()
# classifier = classifier_list[0]
classifier = HopfieldClassifier(512, num_classes).cuda()
# schd = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss()


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
    for domain_idx, test_loader in enumerate(test_domain_loaders):
        pbar = Progbar(len(test_loader))
        for step, batch in enumerate(test_loader):
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda() # + args.num_classes * domain_idx
            outputs = classifier(model(pixel_values))
            loss = criterion(outputs, labels)

            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])


test_domain_loaders = []
model_hopfield_keys = []
buffer = ReplayBuffer(500)
preprocess_train = Compose([
    preprocess_train,
    RandomHorizontalFlip(0.5)
])

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
    
    for k in model_hopfield_keys:
        k.requires_grad = False
    model_hopfield_keys += keys
    total_params = 0
    for p in model_hopfield_keys + train_module_params:
        total_params += p.numel()
    print(f"Total params: {total_params}")
    # if domain_idx > 0:
    #     new_cls = classifier_list[domain_idx].cuda()
    #     classifier.weight = nn.Parameter(torch.cat([classifier.weight, new_cls.weight], dim=0))

    opt = optim.AdamW(keys + train_module_params, lr=args.lr, weight_decay=1e-4)
    # opt = optim.AdamW(keys + train_module_params + list(classifier.parameters()), lr=args.lr, weight_decay=1e-4)

    epochs = args.epochs if (args.later_epochs is None or domain_idx == 0) else args.later_epochs
    for epoch in range(args.epochs if domain_idx == 0 else args.later_epochs):
        model.train()
        running_acc = 0
        pbar = Progbar(len(train_loader))
        print(f"Epoch {epoch + 1}/{epochs} for domain {domain}")
        for step, batch in enumerate(train_loader):
            opt.zero_grad()
            pixel_values = batch['image'].cuda()
            labels = batch['label'].cuda() # + args.num_classes * domain_idx

            if domain_idx > 0:
                er_samples = buffer.sample(batch_size // 2)
                er_images = torch.stack([s[0] for s in er_samples]).cuda()
                er_labels = torch.tensor([s[1] for s in er_samples]).cuda()
                pixel_values = torch.cat([pixel_values, er_images], dim=0)
                labels = torch.cat([labels, er_labels], dim=0)
            
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
            # outputs[:, :-num_classes] = -1e6
            loss = criterion(outputs, labels)

            loss.backward()
            opt.step()
            acc = (outputs.argmax(dim=1) == labels).float().mean().item()
            pbar.update(step + 1, values=[("loss", loss.item()), ("acc", acc)])
        eval()
        print()
    buffer.update(train_loader, 0 * args.num_classes)
    # classifier.add_classes(num_classes)
