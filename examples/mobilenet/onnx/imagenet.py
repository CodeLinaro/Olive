# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
from torch import from_numpy
from torch.utils.data import Dataset

from olive.data.registry import Registry

logger = getLogger(__name__)


class ImagenetDataset(Dataset):
    def __init__(self, data):
        self.images = from_numpy(data["images"])
        self.labels = from_numpy(data["labels"])

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx):
        return {"x": self.images[idx]}, self.labels[idx]


@Registry.register_post_process()
def imagenet_post_fun(output):
    return output.argmax(axis=1)


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@Registry.register_pre_process()
def dataset_pre_process(output_data, **kwargs):
    cache_key = kwargs.get("cache_key")
    cache_file = None
    if cache_key:
        cache_file = Path(f"./cache/data/{cache_key}.npz")
        if cache_file.exists():
            with np.load(Path(cache_file)) as data:
                return ImagenetDataset(data)

    size = kwargs.get("size", 256)
    labels = []
    images = []
    for i, sample in enumerate(output_data):
        if i >= size:
            break
        image = sample["image"]
        label = sample["label"]
        image = image.convert("RGB")
        image = preprocess(image)
        images.append(image)
        labels.append(label)

    result_data = ImagenetDataset({"images": np.array(images), "labels": np.array(labels)})

    if cache_file:
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, images=np.array(images), labels=np.array(labels))

    return result_data
