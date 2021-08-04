import numpy as np
import pickle
import torch

from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self):
        self.tmp = []

    def __getitem__(self, idx):
        return self.tmp[idx]
    
    def __len__(self):
        return len(self.tmp)

