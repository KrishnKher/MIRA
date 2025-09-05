# from torch.utils.data import Dataset
from datasets import Dataset, Features, ClassLabel, Value, DatasetDict
import os
import datasets
from PIL import Image
from tqdm import tqdm
import glob

def make_VLCS(data_path):
    domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    classes = ['bird', 'car', 'chair', 'dog', 'person']
    data = {'image': [], 'label': [], 'domain': []}
    for domain in domains:
        for clas in classes:
            image_folder = os.path.join(data_path, domain, clas)
            for img in tqdm(os.listdir(image_folder)):   
                if img.endswith('.jpg'):
                    try:
                        img = Image.open(os.path.join(image_folder, img)).convert('RGB')
                        data['image'].append(img)
                        data['label'].append(classes.index(clas))
                        data['domain'].append(domain)
                    except:
                        print("pass")
                        pass
    print("Preparing data")
    # create a generator for the data
    def image_generator():
        for img, lab, dom in tqdm(zip(data['image'], data['label'], data['domain'])):
            yield {'image': img, 'label': lab, 'domain': dom}
    features = Features({
        'image': datasets.Image(decode=True),
        'label': ClassLabel(names=classes),
        'domain': Value(dtype='string')
    })
    # data = Dataset.from_dict(data, features=features)
    data = Dataset.from_generator(image_generator, features=features)
    data.push_to_hub("ai22mtech12002/DG_VLCS", private=True)
    # print("Formatting to torch")
    # data = data.with_format("torch")
    return data


def make_TI(data_path):
    domains = ['location_38', 'location_43', 'location_46', 'location_100']
    classes = ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon', 'squirrel']
    data = {'image': [], 'label': [], 'domain': []}
    for domain in domains:
        for clas in classes:
            image_folder = os.path.join(data_path, domain, clas)
            for img in tqdm(os.listdir(image_folder)):   
                if img.endswith('.jpg'):
                    try:
                        img = Image.open(os.path.join(image_folder, img)).convert('RGB')
                        data['image'].append(img)
                        data['label'].append(classes.index(clas))
                        data['domain'].append(domain)
                    except:
                        print("pass")
                        pass
    print("Preparing data")
    # create a generator for the data
    def image_generator():
        for img, lab, dom in tqdm(zip(data['image'], data['label'], data['domain'])):
            yield {'image': img, 'label': lab, 'domain': dom}
    features = Features({
        'image': datasets.Image(decode=True),
        'label': ClassLabel(names=classes),
        'domain': Value(dtype='string')
    })
    # data = Dataset.from_dict(data, features=features)
    data = Dataset.from_generator(image_generator, features=features)
    data.push_to_hub("ai22mtech12002/DG_TI", private=True)
    # print("Formatting to torch")
    # data = data.with_format("torch")
    return data


def make_DN4IL(data_path):
    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    clipart_train_file = '/data/ai22mtech12002/projects/WeightDG/data/DN4IL/DN4IL-dataset-main/clipart_train.txt'
    classes = set()
    for line in open(clipart_train_file).readlines():
        classes.add(line.strip().split(' ')[0].split('/')[1])
    classes = list(classes)
    data_train = {'image': [], 'label': [], 'domain': []}
    data_test = {'image': [], 'label': [], 'domain': []}

    for data, split in zip([data_train, data_test], ['_train', '_test']):
        for domain in domains:
            file = os.path.join(data_path, 'DN4IL-dataset-main', domain + split + '.txt')
            assert os.path.exists(file)
            
            for line in tqdm(open(file).readlines()):
                img, clas = line.strip().split(' ')
                if img.endswith('.jpg') or img.endswith('.png'):
                    assert os.path.exists(os.path.join(data_path, img))
                    img = Image.open(os.path.join(data_path, img)).convert('RGB')
                    data['image'].append(img)
                    data['label'].append(clas)
                    data['domain'].append(domain)
    print("Preparing data")

    # create a generator for the data
    def train_image_generator():
        for img, lab, dom in tqdm(zip(data_train['image'], data_train['label'], data_train['domain'])):
            yield {'image': img, 'label': lab, 'domain': dom}

    def test_image_generator():
        for img, lab, dom in tqdm(zip(data_test['image'], data_test['label'], data_test['domain'])):
            yield {'image': img, 'label': lab, 'domain': dom}

    features = Features({
        'image': datasets.Image(decode=True),
        'label': ClassLabel(names=classes),
        'domain': Value(dtype='string')
    })

    # data = Dataset.from_dict(data, features=features)
    print("Creating train and test datasets")
    data_train = Dataset.from_generator(train_image_generator, features=features)
    data_test = Dataset.from_generator(test_image_generator, features=features)
    data = DatasetDict({'train': data_train, 'test': data_test})

    print("Pushing to hub")
    data.push_to_hub("ai22mtech12002/DN4IL", private=True)
    return data


def make_CDDB(data_path):
    domains = ['gaugan', 'biggan' , 'wild', 'whichfaceisreal', 'san']
    classes = ['0_real', '1_fake']
    data_train = {'image': [], 'label': [], 'domain': []}
    data_test = {'image': [], 'label': [], 'domain': []}

    for data, split in zip([data_train, data_test], ['train', 'val']):
        for domain in domains:
            for clas in ['0_real', '1_fake']:
                imgs = glob.glob(os.path.join(data_path, domain, split, clas, '*'))
                for img in imgs:
                    if img.endswith('.jpeg') or img.endswith('.jpg') or img.endswith('.png'):
                        assert os.path.exists(img)
                        img = Image.open(img).convert('RGB')
                        data['image'].append(img)
                        data['label'].append(clas)
                        data['domain'].append(domain)
    print("Preparing data")

    # create a generator for the data
    def train_image_generator():
        for img, lab, dom in tqdm(zip(data_train['image'], data_train['label'], data_train['domain'])):
            yield {'image': img, 'label': lab, 'domain': dom}

    def test_image_generator():
        for img, lab, dom in tqdm(zip(data_test['image'], data_test['label'], data_test['domain'])):
            yield {'image': img, 'label': lab, 'domain': dom}

    features = Features({
        'image': datasets.Image(decode=True),
        'label': ClassLabel(names=classes),
        'domain': Value(dtype='string')
    })

    # data = Dataset.from_dict(data, features=features)
    print("Creating train and test datasets")
    data_train = Dataset.from_generator(train_image_generator, features=features)
    data_test = Dataset.from_generator(test_image_generator, features=features)
    data = DatasetDict({'train': data_train, 'test': data_test})

    print("Pushing to hub")
    data.push_to_hub("ai22mtech12002/CDDB-hard", private=True)
    return data
