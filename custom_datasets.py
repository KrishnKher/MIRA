# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for Simple Continual Learning datasets
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------

import random
import os
from typing import Union, List, Tuple

import torch
from torch.utils.data.dataset import Subset, ConcatDataset
from torchvision import datasets, transforms

from timm.data import create_transform

import csv
from pathlib import Path
from typing import Union

from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor

from avalanche.benchmarks.datasets import SimpleDownloadableDataset, \
    default_dataset_location
from avalanche.benchmarks import nc_benchmark
from avalanche.benchmarks import ni_benchmark


_default_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),

])

_default_eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
])


class ImagenetR(SimpleDownloadableDataset):
    """Tiny Imagenet Pytorch Dataset"""

    filename = ('imagenet-r.zip',
                'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar')

    def __init__(
            self,
            root: Union[str, Path] = None,
            *,
            train: bool = True,
            transform=None,
            target_transform=None,
            loader=default_loader,
            download=True):
        """
        Creates an instance of the Tiny Imagenet dataset.

        :param root: folder in which to download dataset. Defaults to None,
            which means that the default location for 'tinyimagenet' will be
            used.
        :param train: True for training set, False for test set.
        :param transform: Pytorch transformation function for x.
        :param target_transform: Pytorch transformation function for y.
        :param loader: the procedure to load the instance from the storage.
        :param bool download: If True, the dataset will be  downloaded if
            needed.
        """

        if root is None:
            root = default_dataset_location('imagenet-r')

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader

        super(ImagenetR, self).__init__(
            root, self.filename[1], None, download=download, verbose=True)

        self._load_dataset()


    def _load_metadata(self) -> bool:
        self.data_folder = self.root / 'imagenet-r'

        self.label2id, self.id2label = ImagenetR.labels2dict(
            self.data_folder)
        self.data, self.targets = self.load_data()
        return True

    @staticmethod
    def labels2dict(data_folder: Path):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.

        :param data_folder: The root path of tiny imagenet
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        with open(str(data_folder / 'wnids.txt'), 'r') as f:

            reader = csv.reader(f, delimiter=' ')
            curr_idx = 0
            for ll in reader:
                if ll[0] not in label2id:
                    label2id[ll[0]] = curr_idx
                    id2label[curr_idx] = ll[0]
                    curr_idx += 1

        print(len(label2id), len(id2label))
        return label2id, id2label

    def load_data(self):
        """
        Load all images paths and targets.

        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data = [[], []]

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0] += X
            data[1] += Y

        return data

    def get_train_images_paths(self, class_name):
        """
        Gets the training set image paths.

        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        train_img_folder = self.data_folder / 'train' / class_name

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def get_test_images_paths(self, class_name):
        """
        Gets the test set image paths.

        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        train_img_folder = self.data_folder / 'test' / class_name

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def __len__(self):
        """ Returns the length of the set """
        return len(self.data)

    def __getitem__(self, index):
        """ Returns the index-th x, y pattern of the set """

        path, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CUB(SimpleDownloadableDataset):
    """Tiny Imagenet Pytorch Dataset"""

    filename = ('cub.zip',
                'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar')

    def __init__(
            self,
            root: Union[str, Path] = None,
            *,
            train: bool = True,
            transform=None,
            target_transform=None,
            loader=default_loader,
            download=False):
        
        if root is None:
            root = default_dataset_location('cub')

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader

        super(CUB, self).__init__(
            root, self.filename[1], None, download=download, verbose=True)

        self._load_dataset()


    def _load_metadata(self) -> bool:
        self.data_folder = self.root / 'cub'
        self.label2id, self.id2label = CUB.labels2dict(
            self.data_folder)
        self.data, self.targets = self.load_data()
        return True

    @staticmethod
    def labels2dict(data_folder: Path):
        """
        Returns dictionaries to convert class names into progressive ids
        and viceversa.

        :param data_folder: The root path of tiny imagenet
        :returns: label2id, id2label: two Python dictionaries.
        """

        label2id = {}
        id2label = {}

        folders = os.listdir(str(data_folder / 'train'))
        for i, f in enumerate(folders):
            id = int(f.split('.')[0]) - 1
            label2id[f] = id
            id2label[id] = f

        return label2id, id2label

    def load_data(self):
        """
        Load all images paths and targets.

        :return: train_set, test_set: (train_X_paths, train_y).
        """

        data = [[], []]

        classes = list(range(200))
        for class_id in classes:
            class_name = self.id2label[class_id]

            if self.train:
                X = self.get_train_images_paths(class_name)
                Y = [class_id] * len(X)
            else:
                # test set
                X = self.get_test_images_paths(class_name)
                Y = [class_id] * len(X)

            data[0] += X
            data[1] += Y

        return data

    def get_train_images_paths(self, class_name):
        """
        Gets the training set image paths.

        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        train_img_folder = self.data_folder / 'train' / class_name

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def get_test_images_paths(self, class_name):
        """
        Gets the test set image paths.

        :param class_name: names of the classes of the images to be
            collected.
        :returns img_paths: list of strings (paths)
        """
        train_img_folder = self.data_folder / 'test' / class_name

        img_paths = [f for f in train_img_folder.iterdir() if f.is_file()]

        return img_paths

    def __len__(self):
        """ Returns the length of the set """
        return len(self.data)

    def __getitem__(self, index):
        """ Returns the index-th x, y pattern of the set """

        path, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def SplitImageNetR(
        n_experiences=10,
        *,
        return_task_id=False,
        seed=0,
        fixed_class_order=None,
        shuffle: bool = True,
        train_transform = _default_train_transform,
        eval_transform = _default_eval_transform,
        dataset_root: Union[str, Path] = None):
    """
    Creates a CL benchmark using the Tiny ImageNet dataset.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences in the current benchmark.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset.
        Defaults to None, which means that the default location for
        'tinyimagenet' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    train_set, test_set = _get_imagenet_r_dataset(dataset_root)

    if return_task_id:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)


def SplitCUB(
        n_experiences=10,
        *,
        return_task_id=False,
        seed=0,
        fixed_class_order=None,
        shuffle: bool = True,
        train_transform = _default_train_transform,
        eval_transform = _default_eval_transform,
        dataset_root: Union[str, Path] = None):
    
    train_set, test_set = _get_cub_dataset(dataset_root)

    if return_task_id:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)


def _get_imagenet_r_dataset(dataset_root):
    train_set = ImagenetR(root=dataset_root, train=True)

    test_set = ImagenetR(root=dataset_root, train=False)

    return train_set, test_set


def _get_cub_dataset(dataset_root):
    train_set = CUB(root=dataset_root, train=True)

    test_set = CUB(root=dataset_root, train=False)

    return train_set, test_set






##################################Saksham's DN4IL Implementation######################


class DN4ILDataset(SimpleDownloadableDataset):

    filename = (
        "dn4il-main.zip",
        "https://github.com/NeurAI-Lab/DN4IL-dataset/archive/refs/heads/main.zip"
    )

    domains = ["real", "clipart", "infograph", "painting", "quickdraw", "sketch"]

    def __init__(
        self,
        root: Union[str, Path],
        *,
        train: bool = True,
        transform=None,
        target_transform=None,
        loader=default_loader,
        download: bool = True,
    ):

        self.root = Path(root)
        self.ann_folder = self.root / "DN4IL-dataset-main"
        print("folder", self.ann_folder)

        self.image_root = self.root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader


        super().__init__(
            root, DN4ILDataset.filename[1], None,
            download=download, verbose=True
        )

        self._load_dataset()

    def _load_metadata(self) -> bool:

        class_names = set()
        for domain in DN4ILDataset.domains:
            for split in ("train", "test"):
                f = self.ann_folder / f"{domain}_{split}.txt"
                if not f.exists():
                    raise FileNotFoundError(f"Missing annotation file: {f}")
                entries = f.read_text().split()
                
                # print(entries[0])
                # print(entries[0].split("/", 2)[1])
                class_names.update(e.split("/", 2)[1] for e in entries if len(e.split("/", 2)) > 1)
                # print(len(entries), len(class_names))
        self.label2id = {lbl: idx for idx, lbl in enumerate(sorted(class_names))}
        self.id2label = {v: k for k, v in self.label2id.items()}
        return True

    def load_data(self) -> Tuple[List[Path], List[int]]:
        images: List[Path] = []
        targets: List[int] = []
        domains: List[str] = []

        split = "train" if self.train else "test"
        for domain in DN4ILDataset.domains:
            # print("Domain", domain)
            ann_file = self.ann_folder / f"{domain}_{split}.txt"
            # print("ANN_FILE", ann_file)
            lines = ann_file.read_text().split('\n')
            for rel in lines:
                if len(rel) == 0:
                    continue
                # print(rel)
                rel_path = Path(rel)
                cls = rel_path.parts[-2]   
                fname = rel_path.name.strip().split(' ')[0]      
                img_path = self.image_root / domain / cls / fname
                images.append(img_path)
                targets.append(self.label2id[cls])
                domains.append(domain)
        return images, targets, domains

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        path, target = self.data[index], self.targets[index]
        domains = self.domains[index]
        # print('Domain name:', domains)
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target, domains

    def _load_dataset(self):

        self._load_metadata()
        self.data, self.targets, self.domains = self.load_data()


def SplitDN4IL(
        n_experiences=6,
        *,
        return_task_id=False,
        seed: int = None,
        fixed_domain_order=None,
        shuffle: bool = True,
        train_transform = None,
        eval_transform  = None,
        dataset_root: Union[str, Path] = None):



    train_set = DN4ILDataset(root=dataset_root, train=True,
                             transform=train_transform, download=True)
    test_set  = DN4ILDataset(root=dataset_root, train=False,
                             transform=eval_transform,  download=True)

    # split into one subset per domain
    domains = DN4ILDataset.domains
    train_subsets, test_subsets = [], []
    for d in domains:
        tr_idx = [i for i,p in enumerate(train_set.data) if p.parts[-3] == d]
        te_idx = [i for i,p in enumerate(test_set.data)  if p.parts[-3] == d]
        train_subsets.append(torch.utils.data.Subset(train_set, tr_idx))
        test_subsets.append( torch.utils.data.Subset(test_set,  te_idx))


    return ni_benchmark(
        train_dataset   = train_subsets,
        test_dataset    = test_subsets,
        n_experiences   = n_experiences,
        task_labels     = return_task_id,
        seed            = seed,
        shuffle         = shuffle,
        train_transform = None,  # already in the subsets
        eval_transform  = None
    )

