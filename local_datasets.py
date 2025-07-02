import os
from glob import glob

import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


class ToWeightedGrayscale:
    """
    Convert an RGB tensor [3,H,W] to weighted grayscale [1,H,W].
    """
    def __init__(self):
        self.rgb_weights = torch.tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)

    def __call__(self, tensor):
        if tensor.dim() != 3 or tensor.size(0) != 3:
            raise ValueError(f"Expected RGB image tensor of shape [3,H,W], got {tensor.shape}")
        gray = (tensor * self.rgb_weights).sum(dim=0, keepdim=True)
        return gray


def get_dataset(cfg, split='train'):
    """
    Load train/val/test datasets based on cfg['dataset']['dataset'].
    Returns (train_ds, val_ds) for 'train', or a single dataset for 'val'/'test'.
    """
    name = cfg['dataset']['dataset'].lower()
    if name == 'lapa':
        return get_lapa_dataset(cfg, split)
    elif name in ('sun', 'sun397'):
        return get_sun_dataset(cfg, split)
    elif name == 'coco':
        return get_coco_dataset(cfg, split)
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use 'lapa', 'sun', or 'coco'.")


class _CocoImageDataset(Dataset):
    """
    Wrap a torchvision.datasets.CocoDetection to return only image tensors in a dict.
    """
    def __init__(self, coco_ds):
        self.coco_ds = coco_ds

    def __len__(self):
        return len(self.coco_ds)

    def __getitem__(self, idx):
        img, _ = self.coco_ds[idx]
        return {'image': img}


def get_coco_dataset(cfg, split='train'):
    """
    Load COCO Detection dataset. Returns (train_ds, val_ds) for 'train', else single dataset.
    Wraps outputs so each sample is a dict with only 'image'.
    Expects cfg['dataset']['coco'] keys:
      - data_directory: path to COCO base folder
      - image_dir_<split>: subfolder for images (default <split>2017)
      - ann_file_<split>: annotation JSON (default annotations/instances_<split>2017.json)
      - grayscale: bool
      - imsize: resize size
    """
    from torchvision.datasets import CocoDetection

    coco_cfg = cfg['dataset']['coco']
    base = coco_cfg['data_directory'].rstrip('/')
    imsize = cfg['dataset']['imsize']
    grayscale = coco_cfg.get('grayscale', True)

    # build transforms for image
    transforms_list = [
        T.Lambda(lambda img: F.center_crop(img, min(img.size))),
        T.Resize(imsize),
        T.ToTensor(),
    ]
    if grayscale:
        transforms_list.append(ToWeightedGrayscale())
    transform = T.Compose(transforms_list)

    def make_base_ds(split_key):
        img_dir_name = coco_cfg.get(f'image_dir_{split_key}', f'{split_key}2017')
        ann_file_name = coco_cfg.get(
            f'ann_file_{split_key}', f'annotations/instances_{split_key}2017.json'
        )
        img_dir = os.path.join(base, img_dir_name)
        ann_fp = os.path.join(base, ann_file_name)
        return CocoDetection(root=img_dir, annFile=ann_fp, transform=transform)

    if split == 'train':
        base_train = make_base_ds('train')
        base_val = make_base_ds('val')
        return _CocoImageDataset(base_train), _CocoImageDataset(base_val)
    elif split == 'val':
        base_val = make_base_ds('val')
        return _CocoImageDataset(base_val)
    elif split == 'test':
        base_test = make_base_ds('test')
        return _CocoImageDataset(base_test)
    else:
        raise ValueError(f"Invalid split '{split}'. Use 'train', 'val', or 'test'.")


class _ImageDirDataset(Dataset):
    """
    Load all JPGs in a flat directory, returning {'image': tensor}.
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
    Recursively load all JPGs under root_dir, returning {'image': tensor}.
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
    ds_cfg = cfg['dataset']['lapa']
    base = ds_cfg['data_directory']
    device = cfg['general']['device']
    imsize = cfg['dataset']['imsize']
    grayscale = ds_cfg.get('grayscale', True)

    transforms_list = [
        T.Lambda(lambda img: F.center_crop(img, min(img.size))),
        T.Resize(imsize),
        T.ToTensor(),
    ]
    if grayscale:
        transforms_list.append(ToWeightedGrayscale())
    transform = T.Compose(transforms_list)

    train_dir = os.path.join(base, 'train', 'images')
    val_dir = os.path.join(base, 'val', 'images')
    test_dir = os.path.join(base, 'test', 'images')

    if split == 'train':
        return (_ImageDirDataset(train_dir, transform, device),
                _ImageDirDataset(val_dir, transform, device))
    elif split == 'test':
        return _ImageDirDataset(test_dir, transform, device)
    else:
        raise ValueError(f"Invalid split '{split}'. Use 'train' or 'test'.")


def get_sun_dataset(cfg, split='train'):
    ds_cfg = cfg['dataset']['sun']
    base = ds_cfg['data_directory']
    device = cfg['general']['device']
    imsize = cfg['dataset']['imsize']
    grayscale = ds_cfg.get('grayscale', True)

    transforms_list = [
        T.Lambda(lambda img: F.center_crop(img, min(img.size))),
        T.Resize(imsize),
        T.ToTensor(),
    ]
    if grayscale:
        transforms_list.append(ToWeightedGrayscale())
    transform = T.Compose(transforms_list)

    dataset = _RecursiveImageDirDataset(base, transform, device)
    total = len(dataset)
    indices = torch.randperm(total, generator=torch.Generator().manual_seed(0))
    train_end = int(0.8 * total)
    val_end = train_end + int(0.1 * total)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if split == 'train':
        return Subset(dataset, train_idx), Subset(dataset, val_idx)
    elif split == 'val':
        return Subset(dataset, val_idx)
    elif split == 'test':
        return Subset(dataset, test_idx)
    else:
        raise ValueError(f"Invalid split '{split}'. Use 'train', 'val', or 'test'.")
