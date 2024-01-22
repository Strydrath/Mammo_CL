import os
from torchvision.transforms import Compose, Grayscale, ToTensor, Resize
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from avalanche.benchmarks.utils import AvalancheTensorDataset
import numpy as np
from torch.utils.data import Dataset


class NumpyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


def get_datasets(path_to_data, train_ratio=0.8, batch_size=32):
    # Create ImageFolder dataset from directory
    dataset = ImageFolder(path_to_data, transform =  Compose([Resize(512),Grayscale(), ToTensor()]))
    
    # Split dataset into training and testing sets
    num_train = int(len(dataset) * train_ratio)
    num_test = int((len(dataset) - num_train)/2)
    num_val = len(dataset) - num_train - num_test
    train_set, test_set, val_set = random_split(dataset, [num_train, num_test, num_val])
    print(train_set[0][0])
    return train_set, test_set, val_set