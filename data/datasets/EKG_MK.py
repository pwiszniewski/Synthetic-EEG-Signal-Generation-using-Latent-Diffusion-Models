from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class EKGMK(Dataset):
    def __init__(self,
                 dataset_dir,
                 *,
                 is_resample=False,
                 cache_dir=None,
                 use_cache=False,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        # open data mat file with shape (2109, 180)
        data = scipy.io.loadmat(Path(dataset_dir) / 'dane.mat')
        if is_resample:
            # resample to 256
            data = scipy.signal.resample(data['DANE'], 256, axis=1)
        else:
            # add zeros to the end of each row to make it 256
            data = np.concatenate((data['DANE'], np.zeros((2109, 76))), axis=1)
        # reshape to (2109, 1, 256)
        data = np.expand_dims(data, axis=1)

        # set labels as zeros
        targets = np.zeros((2109, 1))

        if self.target_transform:
            targets = self.target_transform(targets)

        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets).type(torch.LongTensor)

    def __len__(self):
        # return len(self.dataset)
        return len(self.targets)

    def __getitem__(self, idx):
        # data, label = self.dataset[idx]
        data, label = self.data[idx], self.targets[idx]
        return data, label





