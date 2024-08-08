# ...
# Transform can be applied to PIL images, tensors, ndimensionalarrays, or custom dataset

# On Images
# --------
# CentreCrop , Grayscale ,Pad, RandomAffine
# Resize, Scale

# On Tensors
#  Linear Transformation

#  Conversion
# ToPILImages: from tensor or ndarray
# ToTensors : from numpy.ndarray or PILImages

# Generic
# --------
# Use Lambda

# Custom
#  -----------
# write own class

# Compose multiple Transforms
# -----------------------
# composed - transforms.Rescale(256)
# torchvision.transform.ToTensor()

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):
    def __init__(self, transform=None):
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # note that we do not convert to tensors here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=None)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))