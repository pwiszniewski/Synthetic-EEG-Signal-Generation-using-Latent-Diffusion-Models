from abc import ABC
from pathlib import Path

import numpy as np
import os
import scipy.io

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

"""" 
Base class for all EEG datasets 
data is loaded from files and cached in memory
transform and target_transform are applied to data and targets respectively
transformed data and targets are cached in memory
if use_cache is False, data is loaded from files every time
"""
class BaseEEGDataset(Dataset, ABC):
    channels = None
    fs = None

    def __init__(self,
                 dataset_dir,
                 *,
                 cache_dir=None,
                 use_cache=False,
                 ch_names=None,
                 indices=None,
                 transform=None,
                 target_transform=None):

        self.dataset_dir = Path(dataset_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.transform = transform
        self.target_transform = target_transform

        end, start = self._get_start_end()

        self.cache_dir = self._get_cache_dir_and_create_if_not_exist(self.cache_dir, self.dataset_dir)
        cache_transformed_path = self._get_transform_path(self.cache_dir, transform)

        # TODO: do not load raw data when transformed data exist
        files = self._get_file_list()
        data, targets = self._load_data_and_targets(self.cache_dir, self.dataset_dir, files, use_cache)

        ch_indices = self._get_channel_indices(ch_names)
        idx = self._get_indices(indices, data)
        data, targets = self._select_data_and_targets(ch_indices, data, end, idx, start, targets)

        data = self._apply_data_transform(cache_transformed_path, data, transform, use_cache)
        # TODO: if data is loaded from cache then we have all indices but should be subset from idx

        self.targets = self._apply_target_transform(targets)

        self._make_dataset(data, self.targets)

    def _load_data_raw(self, dataset_dir, files):
        """
        Load data from files
        It should be implemented in child class
        It should return data and targets
        """
        raise NotImplementedError

    def _get_start_end(self):
        return None, None

    def _get_file_list(self):
        return None

    def _select_data_and_targets(self, ch_indices, data, end, idx, start, targets):
        data = data[idx]
        data = data[:, ch_indices, start:end]
        targets = targets[idx]
        return data, targets

    def _make_dataset(self, data, targets):
        data = torch.Tensor(data)
        targets = torch.Tensor(targets).type(torch.LongTensor)
        self.dataset = TensorDataset(data, targets)

    def _apply_target_transform(self, targets):
        if self.target_transform:
            targets = self.target_transform.apply(targets)
        return targets

    def _get_cache_dir_and_create_if_not_exist(self, cache_dir, dataset_dir):
        cache_dir = dataset_dir / 'cache' if not cache_dir else cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _apply_data_transform(self, cache_transformed_path, data, transform, use_cache):
        if transform:
            if use_cache and cache_transformed_path.exists():
                print('loading data transformed cache files from', cache_transformed_path)
                npzfile = np.load(cache_transformed_path)
                keys = npzfile.files
                data = npzfile[keys[0]]
            else:
                data = self.transform(data)
                if use_cache:
                    print('saving data transformed cache to ', cache_transformed_path)
                    np.savez(cache_transformed_path, data)
        return data

    def _get_transform_path(self, cache_dir, transform):
        if isinstance(transform, transforms.Compose):
            transform_name = '_'.join([t.get_name() for t in transform.transforms])
        elif transform is None:
            transform_name = ''
        else:
            transform_name = transform.get_name()
        cache_transformed_path = cache_dir / f'data_{transform_name}.npz'
        return cache_transformed_path

    def _load_data_and_targets(self, cache_dir, dataset_dir, files, use_cache):
        cache_raw_path = cache_dir / 'data_raw.npz'
        if use_cache and cache_raw_path.exists():
            data, targets = self._load_data_raw_from_cache(cache_raw_path)
        else:
            data, targets = self._load_data_raw(dataset_dir, files)
            if use_cache:
                self._save_cache_raw(cache_raw_path, data, targets)
        return data, targets

    def _save_cache_raw(self, cache_raw_path, data, targets):
        print('saving data raw cache to ', cache_raw_path)
        np.savez(cache_raw_path, data, targets)

    def _load_data_raw_from_cache(self, cache_raw_path):
        print('loading data raw cache files from', cache_raw_path)
        npzfile = np.load(cache_raw_path)
        keys = npzfile.files
        data = npzfile[keys[0]]
        targets = npzfile[keys[1]]
        return data, targets

    def _get_channel_indices(self, ch_names):
        if ch_names:
            mask = np.in1d(np.char.lower(self.channels), np.char.lower(ch_names))
            ch_indices = np.argwhere(mask).flatten()
        else:
            ch_indices = list(range(len(self.channels)))
        return ch_indices

    def _get_indices(self, indices, data):
        if indices is None:
            idx = list(range(data.shape[0]))
        else:
            idx = indices
        return idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class BaseEEGDatasetSubjectsBlocks(BaseEEGDataset, ABC):
    """
    Base class for datasets with subjects, blocks and characters
    """
    target_freqs = None
    num_subjects = None
    num_blocks = None
    num_chars = None

    def __init__(self,
                 dataset_dir,
                 *,
                 cache_dir=None,
                 use_cache=False,
                 ch_names=None,
                 subjects=None,
                 transform=None,
                 target_transform=None):

        self.subjects = self._get_subject_indices(subjects)
        self.global_indices = self._get_global_indices()

        super().__init__(dataset_dir,
                        cache_dir=cache_dir,
                        use_cache=use_cache,
                        ch_names=ch_names,
                        indices=self.get_indices(subjects=self.subjects),
                        transform=transform,
                        target_transform=target_transform)

    def get_subject_indices(self, subjects):
        idx = self.global_indices[subjects, :, :].flatten()
        return idx

    def get_indices(self, subjects=None, rejected_subjects=None,
                    blocks=None, rejected_blocks=None,
                    chars=None, rejected_chars=None):
        subjects_mask = self._get_mask(len(self.subjects), subjects, rejected_subjects)
        block_mask = self._get_mask(self.num_blocks, blocks, rejected_blocks)
        chars_mask = self._get_mask(self.num_chars, chars, rejected_chars)
        return self.global_indices[subjects_mask][:, block_mask][:, :, chars_mask].flatten()

    def _get_mask(self, full_indices, indices, rejected_indices):
        if indices is not None:
            mask = np.zeros(full_indices, dtype=bool)
            mask[indices] = True
        else:
            mask = np.ones(full_indices, dtype=bool)
        if rejected_indices is not None:
            mask[rejected_indices] = False
        return mask

    def _get_global_indices(self):
        num_subjects = len(self.subjects)
        global_indices = np.zeros((num_subjects, self.num_blocks, self.num_chars), dtype=int)
        cnt = 0
        for i in range(num_subjects):
            for j in range(self.num_blocks):
                for k in range(self.num_chars):
                    global_indices[i, j, k] = cnt
                    cnt += 1
        return global_indices

    def _get_subject_indices(self, subjects):
        if subjects:
            if isinstance(subjects, int):
                sub_indices = list(range(subjects))
            else:
                sub_indices = list(range(subjects[0], subjects[1]))
        else:
            sub_indices = list(range(self.num_subjects))
        return sub_indices