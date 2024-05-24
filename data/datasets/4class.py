import numpy as np
import os
import scipy.io
import mne as mne

import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms


class BETADataset(Dataset):
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

        cache_dir = dataset_dir / 'cache' if not cache_dir else cache_dir
        #TODO check "cache_dir = cache_dir or dataset_dir / 'cache'" instead this above one
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_raw_path = cache_dir / 'data_raw.npz'

        if use_cache and cache_raw_path.exists():
            print('loading data raw cache files from', cache_raw_path)
            npzfile = np.load(cache_raw_path)
            keys = npzfile.files
            data = npzfile[keys[0]]
            targets = npzfile[keys[1]]
        else:
            data = []
            targets = []

            print('reading BETA files')
            for file in sorted(files, key=lambda e: int(e.split('.')[0][1:]))[0:self.num_subjects]:
                print(file)
                mat = scipy.io.loadmat(dataset_dir / file)
                arr = mat['data']['EEG'][0][0]
                arr = arr.reshape(*arr.shape[:-2], -1)
                # data.append(np.moveaxis(arr, -1, 0)[:, :, start:end])
                if arr.shape[1] == 750:
                    arr = np.pad(arr, ((0, 0), (0, 250), (0, 0)))
                data.append(np.moveaxis(arr, -1, 0)[:, :, :])
                info = mat['data']['suppl_info'][0][0][0][0]
                targets.extend(np.tile(range(len(info['freqs'][0])), 4))

            data = np.array(data)
            data = data.reshape(-1, *data.shape[2:])
            targets = np.array(targets)

            self.target_freqs = np.round(info['freqs'][0], 1)

            if use_cache:
                print('saving data raw cache to ', cache_raw_path)
                np.savez(cache_raw_path, data, targets)

        if ch_names:
            mask = np.in1d(np.char.lower(self.channels), np.char.lower(ch_names))
            ch_indices = np.argwhere(mask).flatten()
        else:
            ch_indices = list(range(len(self.channels)))

        if subjects:
            if isinstance(subjects, int):
                subjects = list(range(subjects))
            else:
                subjects = list(range(subjects[0], subjects[1]))
        else:
            subjects = list(range(self.num_subjects))

        global_indices = np.zeros((self.num_subjects, self.num_blocks, self.num_chars), dtype=int)
        cnt = 0
        for i in range(self.num_subjects):
            for j in range(self.num_blocks):
                for k in range(self.num_chars):
                    global_indices[i, j, k] = cnt
                    cnt += 1

        idx = global_indices[subjects, :, :].flatten()
        data = data[idx]
        data = data[:, ch_indices, start:end]
        targets = targets[idx]

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







# from typing import Optional
#
# from pytorch_lightning.core import LightningDataModule
# from torch.utils.data import DataLoader
#
#
# class BETADataModule(LightningDataModule):
#     def __init__(self, data_dir: str, batch_size: int = 16) -> None:
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#
#     def prepare_data(self):
#         # download, split, etc...
#         # only called on 1 GPU/TPU in distributed
#         print('prepare_data')
#
#     def setup(self, stage: Optional[str] = None) -> None:
#         # make assignments here (val/train/test split)
#         # called on every process in DDP
#         print('setup')
#
#     def train_dataloader(self) -> DataLoader:
#         # train_split = Dataset(...)
#         # return DataLoader(train_split)
#         return None
#
#     def val_dataloader(self) -> DataLoader:
#         # val_split = Dataset(...)
#         # return DataLoader(val_split)
#         return None
#
#     def test_dataloader(self) -> DataLoader:
#         # test_split = Dataset(...)
#         # return DataLoader(test_split)
#         return None
#
#     def teardown(self, stage: Optional[str] = None) -> None:
#         # clean up after fit or test
#         # called on every process in DDP
#         print('teardown')