# tests/test_local_datasets.py

import pytest
import torch
from PIL import Image

from local_datasets import LaPaDataset, get_lapa_dataset, create_circular_mask

# Helpers to create dummy images and labels
def make_dummy_image(path, size=(256, 256), color=(128, 128, 128)):
    img = Image.new('RGB', size, color)
    img.save(path)

def make_dummy_label(path, size=(256, 256), color=128):
    lbl = Image.new('L', size, color)
    lbl.save(path)

@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(exist_ok=True)
        for i in range(2):
            make_dummy_image(img_dir / f"{i}.jpg")
            make_dummy_label(lbl_dir / f"{i}.png")
    return str(root)

@pytest.fixture
def cfg(data_dir):
    return {
        "data_directory": data_dir,
        "device": "cpu",
        "imsize": (256, 256),
        "grayscale": False,
        "target": ["semantic", "boundary"],
        "debug_subset": None,
        "retinal_compression": False,
        "circular_mask": False,
        "fov": 30,
    }

def test_dataset_length_and_split(cfg):
    train_ds, val_ds = get_lapa_dataset(cfg, split='train')
    test_ds = get_lapa_dataset(cfg, split='test')
    assert len(train_ds) == 2
    assert len(val_ds) == 2
    assert len(test_ds) == 2

def test_len_and_getitem(cfg):
    ds = LaPaDataset(cfg, mode='train')
    assert len(ds) == 2
    sample = ds[0]
    # Keys and shapes
    assert set(sample.keys()) == {"image", "segmentation_maps", "contour"}
    assert sample["image"].shape == (3, 256, 256)
    assert sample["segmentation_maps"].shape == (256, 256)
    assert sample["contour"].shape == (1, 256, 256)

def test_circular_mask(cfg):
    cfg_mask = dict(cfg, circular_mask=True)
    ds = LaPaDataset(cfg_mask, mode='train')
    mask = create_circular_mask(256, 256).view(1, 256, 256)
    # last 2 dimensions should be the same
    assert ds[0]["image"].shape[-2:] == mask.shape[-2:]

def test_index_error(cfg):
    ds = LaPaDataset(cfg, mode='val')
    with pytest.raises(IndexError):
        _ = ds[10]


