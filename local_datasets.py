import os
from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

class ToWeightedGrayscale:
    def __init__(self):
        self.rgb_weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)

    def __call__(self, tensor):
        if tensor.size(0) != 3:
            raise ValueError("Expected RGB image (3 channels), got shape: {}".format(tensor.shape))
        return (tensor * self.rgb_weights).sum(dim=0, keepdim=True)


def get_dataset(cfg, split='train'):
    """
    Dispatch to the appropriate dataset loader based on cfg['dataset']['dataset'] (lowercase).
    Returns (train_ds, val_ds) for 'train', or test_ds for 'test'.
    """
    ds_name = cfg['dataset']['dataset'].lower()
    if ds_name == 'lapa':
        return get_lapa_dataset(cfg, split)
    elif ds_name in ('sun', 'sun397'):
        return get_sun_dataset(cfg, split)
    else:
        raise ValueError(f"Unsupported dataset '{ds_name}'. Use 'lapa' or 'sun'.")


class _ImageDirDataset(Dataset):
    """
    Simple image folder loader: assumes a directory with all images in one folder.
    Returns dict with 'image' tensor on the configured device.
    """
    def __init__(self, image_dir, transform, device):
        self.paths = sorted(glob(os.path.join(image_dir, '*.jpg')))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        tensor = self.transform(img)
        return {'image': tensor.to(self.device)}


class _RecursiveImageDirDataset(Dataset):
    """
    Recursive image loader: finds all images under a directory tree.
    """
    def __init__(self, root_dir, transform, device):
        pattern = os.path.join(root_dir, '**', '*.jpg')
        self.paths = sorted(glob(pattern, recursive=True))
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        tensor = self.transform(img)
        return {'image': tensor.to(self.device)}


def get_lapa_dataset(cfg, split='train'):
    """
    Load LaPa images only (no labels) from <data_directory>/{split}/images/.
    Simplifies original LaPaDataset to just image tensors.
    """
    ds_cfg = cfg['dataset']['lapa']
    root = ds_cfg['data_directory']
    device = cfg['general']['device']
    imsize = cfg['dataset']['imsize']

    transform = T.Compose([
        T.Lambda(lambda img: F.center_crop(img, min(img.size))),
        T.Resize(imsize),
        T.ToTensor(),
    ])

    grayscale = cfg['dataset'].get('grayscale', True)

    if grayscale:
        transform = T.Compose([
            transform,
            ToWeightedGrayscale(),
        ])

    if split == 'train':
        train_dir = os.path.join(root, 'train', 'images')
        val_dir   = os.path.join(root, 'val',   'images')
        train_ds = _ImageDirDataset(train_dir, transform, device)
        val_ds   = _ImageDirDataset(val_dir,   transform, device)
        return train_ds, val_ds
    elif split == 'test':
        test_dir = os.path.join(root, 'test', 'images')
        return _ImageDirDataset(test_dir, transform, device)
    else:
        raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")


def get_sun_dataset(cfg, split='train'):
    """
    Load the SUN397 dataset images recursively from <data_directory>/SUN397/.
    Since SUN397 has no official train/val split, returns the full set for any split.
    """
    ds_cfg = cfg['dataset']['sun']
    root = ds_cfg['data_directory']
    device = cfg['general']['device']
    imsize = cfg['dataset']['imsize']

    transform = T.Compose([
        T.Lambda(lambda img: F.center_crop(img, min(img.size))),
        T.Resize(imsize),
        T.ToTensor(),
    ])

    grayscale = cfg['dataset'].get('grayscale', True)

    if grayscale:
        transform = T.Compose([
            transform,
            ToWeightedGrayscale(),
        ])

    dataset = _RecursiveImageDirDataset(root, transform, device)
    seed = 0
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    split_idx = int(0.8 * len(dataset))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:split_idx + int(0.1 * len(dataset))]
    test_indices = indices[split_idx + int(0.1 * len(dataset)):]

    if split == 'train' or split == 'val':
        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
        return train_ds, val_ds
    elif split == 'test':
        dataset = torch.utils.data.Subset(dataset, test_indices)
        return dataset
    else:
        raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'.")

