import numpy as np
import os
import scipy.io

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms


class BETADataset(Dataset):
    """
    https://www.frontiersin.org/articles/10.3389/fnins.2020.00627/full
    """
    channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'PZ',
                'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    target_freqs = [8.6, 8.8, 9., 9.2, 9.4, 9.6, 9.8, 10., 10.2, 10.4, 10.6, 10.8, 11., 11.2, 11.4, 11.6, 11.8, 12.,
                    12.2, 12.4, 12.6, 12.8, 13., 13.2, 13.4, 13.6, 13.8, 14., 14.2, 14.4, 14.6, 14.8, 15., 15.2, 15.4,
                    15.6, 15.8, 8., 8.2, 8.4]
    fs = 250
    num_subjects = 70
    num_blocks = 4
    num_chars = 40

    def __init__(self,
                 dataset_dir,
                 *,
                 cache_dir=None,
                 use_cache=False,
                 ch_names=None,
                 subjects=None,
                 transform=None,
                 target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        files = [file for file in os.listdir(dataset_dir) if file.endswith('.mat')]

        start = int((0.13 + 0.5) * self.fs)
        end = start + int(0.4 * self.fs)

        cache_dir = self._get_cache_dir_and_create_if_not_exist(cache_dir, dataset_dir)
        cache_transformed_path = self._get_transform_path(cache_dir, transform)

        #TODO: do not load raw data when transformed data exist
        data, targets = self._load_data_and_targets(cache_dir, dataset_dir, files, use_cache)

        ch_indices = self._get_channel_indices(ch_names)
        subjects = self._get_subject_indices(subjects)

        self.global_indices = self._get_global_indices()
        idx = self.get_subject_indices(subjects)
        data, targets = self._select_data_and_targets(ch_indices, data, end, idx, start, targets)

        data = self._apply_data_transform(cache_transformed_path, data, transform, use_cache)
        #TODO: if data is loaded from cache then we have all indices but should be subset from idx

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
        print('reading BETA files')
        for file in sorted(files, key=lambda e: int(e.split('.')[0][1:]))[0:self.num_subjects]:
            print(file)
            mat = scipy.io.loadmat(dataset_dir / file)
            arr = mat['data']['EEG'][0][0]
            arr = arr.reshape(*arr.shape[:-2], -1)
            if arr.shape[1] == 750:
                arr = np.pad(arr, ((0, 0), (0, 250), (0, 0)))
            data.append(np.moveaxis(arr, -1, 0)[:, :, :])
            info = mat['data']['suppl_info'][0][0][0][0]
            targets.extend(np.tile(range(len(info['freqs'][0])), self.num_blocks))
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

    def _get_subject_indices(self, subjects):
        if subjects:
            if isinstance(subjects, int):
                sub_indices = list(range(subjects))
            else:
                sub_indices = list(range(subjects[0], subjects[1]))
        else:
            sub_indices = list(range(self.num_subjects))
        return sub_indices

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