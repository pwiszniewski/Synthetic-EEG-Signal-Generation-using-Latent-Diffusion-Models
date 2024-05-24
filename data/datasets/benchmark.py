from pathlib import Path

import numpy as np
import os
import scipy.io

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms


class BenchmarkDataset(Dataset):
    """http://bci.med.tsinghua.edu.cn/download.html"""
    channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
                'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    target_freqs = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2, 8.4, 9.4,
                    10.4, 11.4, 12.4, 13.4, 14.4, 15.4, 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6,	15.6, 8.8, 9.8, 10.8,
                    11.8, 12.8, 13.8, 14.8, 15.8]
    fs = 250
    num_subjects = 35
    num_blocks = 6
    num_chars = 40
    """
    :arg dataset_dir: path to the directory containing the dataset
    :arg cache_dir: path to the directory where the cache will be stored
    :arg use_cache: if True, the cache will be used if it exists, otherwise it will be created
    :arg ch_names: list of channel names to be used
    :arg subjects: list of subject indices to be used if subjects is int indices from 0 to subjects-1 will be used
    :arg transform: transform to be applied to the data
    :arg target_transform: transform to be applied to the target
    :arg signal_length: length of the signal in seconds
    """
    def __init__(self,
                 dataset_dir,
                 *,
                 cache_dir=None,
                 use_cache=False,
                 ch_names=None,
                 subjects_selected=None,
                 targets_selected=None,
                 transform=None,
                 target_transform=None,
                 signal_length=0.4,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        dataset_dir = Path(dataset_dir)
        files = [file for file in os.listdir(dataset_dir) if file.endswith('.mat')]

        visual_latency = 0.14
        visual_cue = 0.5
        # signal_length = 0.4
        start = int((visual_latency + visual_cue) * self.fs)
        end = start + int(signal_length * self.fs)

        cache_dir = self._get_cache_dir_and_create_if_not_exist(cache_dir, dataset_dir)
        cache_transformed_path = self._get_transform_path(cache_dir, transform)

        #TODO: do not load raw data when transformed data exist
        data, targets = self._load_data_and_targets(cache_dir, dataset_dir, files, use_cache)

        ch_indices = self._get_channel_indices(ch_names)
        subjects_indices = self._get_subjects_indices(subjects_selected)
        targets_indices = self._get_targets_indices(targets_selected)

        self.global_indices = self._get_global_indices()
        idx = self.get_indices(subjects=subjects_indices, chars=targets_indices)
        data, targets = self._select_data_and_targets(ch_indices, data, end, idx, start, targets)

        # plot sample data
        # import matplotlib.pyplot as plt
        # for i in range(10):
        #     ix = np.random.randint(0, len(data))
        #     plt.plot(data[ix, 0, :])
        #     plt.show()


        data = self._apply_data_transform(cache_transformed_path, data, transform, use_cache)
        targets = self._apply_target_transform(targets)

        self._make_dataset(data, targets)

    def _select_data_and_targets(self, ch_indices, data, end, idx, start, targets):
        data = data[idx]
        data = data[:, ch_indices, start:end]
        targets = targets[idx]
        return data, targets

    def get_subject_indices(self, subjects):
        idx = self.global_indices[subjects, :, :].flatten()
        return idx

    def get_indices(self, subjects=None, rejected_subjects=None,
                    blocks=None, rejected_blocks=None,
                    chars=None, rejected_chars=None):
        subjects_mask = self._get_mask(self.num_subjects, subjects, rejected_subjects)
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
        global_indices = np.zeros((self.num_subjects, self.num_blocks, self.num_chars), dtype=int)
        cnt = 0
        for i in range(self.num_subjects):
            for j in range(self.num_blocks):
                for k in range(self.num_chars):
                    global_indices[i, j, k] = cnt
                    cnt += 1
        return global_indices

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

    def _load_data_raw(self, dataset_dir, files):
        data = []
        targets = []
        print('reading benchmark files')
        for file in sorted(files, key=lambda e: int(e.split('.')[0][1:]))[0:self.num_subjects]:
            print(file)
            mat = scipy.io.loadmat(dataset_dir / file)
            arr = mat['data']
            arr = np.swapaxes(arr, 2, 3)
            arr = arr.reshape(*arr.shape[:-2], -1)
            # if arr.shape[1] == 750:
            #     arr = np.pad(arr, ((0, 0), (0, 250), (0, 0)))
            data.append(np.moveaxis(arr, -1, 0)[:, :, :])
            targets.extend(np.tile(range(len(self.target_freqs)), self.num_blocks))
        data = np.array(data)
        data = data.reshape(-1, *data.shape[2:])
        targets = np.array(targets)
        return data, targets

    def _load_data_raw_from_cache(self, cache_raw_path):
        print('loading data raw cache files from', cache_raw_path)
        npzfile = np.load(cache_raw_path)
        keys = npzfile.files
        data = npzfile[keys[0]]
        targets = npzfile[keys[1]]
        return data, targets

    def _get_subjects_indices(self, subjects):
        if subjects:
            if isinstance(subjects, int):
                sub_indices = list(range(subjects))
            elif isinstance(subjects, tuple) or isinstance(subjects, list): # list of subjects
                sub_indices = [i - 1 for i in subjects]
                # sub_indices = list(range(subjects[0], subjects[1]))
            else:
                raise ValueError('subjects should be int, tuple or list')
        else:
            sub_indices = list(range(self.num_subjects)) # all subjects
        return sub_indices

    def _get_targets_indices(self, targets):
        if targets:
            if isinstance(targets, tuple) or isinstance(targets, list): # list of targets
                targets_indices = [self.target_freqs.index(t) for t in targets]
            else:
                raise ValueError('targets should be int, tuple or list')
        else:
            targets_indices = list(range(len(self.target_freqs))) # all targets
        return targets_indices

    def _get_channel_indices(self, ch_names):
        if ch_names:
            mask = np.in1d(np.char.lower(self.channels), np.char.lower(ch_names))
            ch_indices = np.argwhere(mask).flatten()
        else:
            ch_indices = list(range(len(self.channels)))
        return ch_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]