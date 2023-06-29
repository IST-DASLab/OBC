import os
import sys
sys.path.append('yolov5')

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def random_subset(data, nsamples, seed):
    set_seed(seed)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return Subset(data, idx[:nsamples])


_IMAGENET_RGB_MEANS = (0.485, 0.456, 0.406)
_IMAGENET_RGB_STDS = (0.229, 0.224, 0.225)

def get_imagenet(path, noaug=False):
    img_size = 224  # standard
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])
    non_rand_resize_scale = 256.0 / 224.0  # standard
    test_transform = transforms.Compose([
        transforms.Resize(round(non_rand_resize_scale * img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_RGB_MEANS, std=_IMAGENET_RGB_STDS),
    ])

    train_dir = os.path.join(os.path.expanduser(path), 'train')
    test_dir = os.path.join(os.path.expanduser(path), 'val')

    if noaug:
        train_dataset = datasets.ImageFolder(train_dir, test_transform)
    else:
        train_dataset = datasets.ImageFolder(train_dir, train_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

class YOLOv5Wrapper(Dataset):
    def __init__(self, original):
        self.original = original
    def __len__(self):
        return len(self.original)
    def __getitem__(self, idx):
        tmp = list(self.original[idx])
        tmp[0] = tmp[0].float() / 255
        return tmp

def get_coco(path, batchsize):
    from yolov5.utils.datasets import LoadImagesAndLabels
    train_data = LoadImagesAndLabels(
        os.path.join(path, 'images/calib'), batch_size=batchsize
    )
    train_data = YOLOv5Wrapper(train_data)
    train_data.collate_fn = LoadImagesAndLabels.collate_fn
    test_data = LoadImagesAndLabels(
        os.path.join(path, 'images/val2017'), batch_size=batchsize, pad=.5
    )
    test_data = YOLOv5Wrapper(test_data)
    test_data.collate_fn = LoadImagesAndLabels.collate_fn
    return train_data, test_data


DEFAULT_PATHS = {
    'imagenet': [
        '../imagenet'
    ],
    'coco': [
        '../coco'
    ]
}

def get_loaders(
    name, path='', batchsize=-1, workers=8, nsamples=1024, seed=0,
    noaug=False
):
    if name == 'squad':
        if batchsize == -1:
            batchsize = 16
        import bertsquad
        set_seed(seed)
        return bertsquad.get_dataloader(batchsize, nsamples), None

    if not path:
        for path in DEFAULT_PATHS[name]:
            if os.path.exists(path):
                break

    if name == 'imagenet':
        if batchsize == -1:
            batchsize = 128
        train_data, test_data = get_imagenet(path, noaug=noaug)
        train_data = random_subset(train_data, nsamples, seed)
    if name == 'coco':
        if batchsize == -1:
            batchsize = 16
        train_data, test_data = get_coco(path, batchsize)

    collate_fn = train_data.collate_fn if hasattr(train_data, 'collate_fn') else None
    trainloader = DataLoader(
        train_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=True,
        collate_fn=collate_fn
    )
    collate_fn = test_data.collate_fn if hasattr(test_data, 'collate_fn') else None
    testloader = DataLoader(
        test_data, batch_size=batchsize, num_workers=workers, pin_memory=True, shuffle=False,
        collate_fn=collate_fn
    )

    return trainloader, testloader
