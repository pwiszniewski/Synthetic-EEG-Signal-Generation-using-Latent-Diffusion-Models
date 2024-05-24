from typing import Union

import numpy as np
import random

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms


class SyntheticDataset(Dataset):

    def __init__(self,
                 dataset_size,
                 num_time_samples,
                 fs,
                 sin_freqs: Union[list, tuple],
                 phase: Union[float, str] = 'random',
                 num_channels=1,
                 *,
                 noise_std=0.0,
                 amplitude=1.0,
                 manual_seed=None,
                 cache_dir=None,
                 transform=None,
                 target_transform=None):

        self.transform = transform
        self.target_transform = target_transform

        if manual_seed:
            random.seed(manual_seed)
            torch.manual_seed(manual_seed)

        x = np.linspace(0, num_time_samples - 1, num_time_samples)
        # d = np.sin(2 * np.pi * x / fs * sin_freq)
        # data = np.tile(d.reshape(-1, 1), dataset_size).T
        data = []
        labels = []
        for i in range(dataset_size):
            current_phase = random.random() * 2 * np.pi if phase == 'random' else phase
            sin_freq = random.choice(sin_freqs)
            d = amplitude * np.sin(2 * np.pi * x / fs * sin_freq + current_phase)
            if noise_std > 0:
                d += np.random.normal(0, noise_std, num_time_samples)
            data.append(d)
            labels.append(sin_freqs.index(sin_freq))

        if num_channels == 1:
            data = np.expand_dims(data, 1)
        else:
            raise
            # TODO: add various channel number

        if transform:
            if isinstance(transform, transforms.Compose):
                transform_name = '_'.join([t.get_name() for t in transform.transforms])
            else:
                transform_name = transform.get_name()

            cache_transformed_path = cache_dir / f'data_{transform_name}.npz'

            data = self.transform(data)

            # TODO: saving and loading cache transformed data
            # if cache_transformed_path.exists():
            #     print('loading data raw cache files from', cache_transformed_path)
            #     npzfile = np.load(cache_transformed_path)
            #     keys = npzfile.files
            #     data = npzfile[keys[0]]
            # else:
            #     data = self.transform(data)
            #     if use_cache:
            #         print('saving data transformed cache to ', cache_transformed_path)
            #         np.savez(cache_transformed_path, data)

        tensor_x = torch.Tensor(data)
        targets = torch.Tensor(labels).long()

        if self.target_transform:
            targets = self.target_transform(targets)

        self.dataset = TensorDataset(tensor_x, targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
